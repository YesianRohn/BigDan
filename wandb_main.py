# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import operator
import torch.backends.cudnn as cudnn
import json
import wandb

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
import timm.models
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset, GroupedDataset
from engine import train_one_epoch, evaluate, test
from samplers import RASampler

import models
import utils
import random
import logging
import sys
from time import strftime, localtime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class CustomClassifier(torch.nn.Module):
    def __init__(self, model, input_dim, output_dim, model_type=None,
                    multi_dataset_classes=None, known_data_source=False):
        '''
        Custom classifier with a Norm layer followed by a Linear layer.
        '''
        super().__init__()
        self.backbone = model
        self.inner_dim = input_dim

        self.known_data_source = known_data_source
        self.multi_dataset_classes = multi_dataset_classes

        if model_type == 'efficientnet':
            self.inner_dim = 512
            self.channel_bn = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Linear(input_dim*7*7, self.inner_dim))
            # [batch_size * num_features * kernel_size * kernel_size] --> [batch_size, inner_dim]
        else:
            self.channel_bn = torch.nn.BatchNorm1d(  # if model_type == 'deit':
                input_dim,
                affine=False,
            )
            # [batch_size * num_features] --> [batch_size * inner_dim] (inner_dim = num_features]
        self.layers = torch.nn.Sequential(torch.nn.Linear(self.inner_dim, output_dim))

    def forward(self, img, dataset_id=None):
        # TODO: how to leverage dataset_source in training and infernece stage?
        pdtype = img.dtype
        # print(self.backbone.forward_features(img))
        if type(self.backbone.forward_features(img))==tuple:
            # print(torch.as_tensor(self.backbone.forward_features(img)))
            feature=torch.as_tensor(self.backbone.forward_features(img)).to(pdtype)
        else:
            feature = self.backbone.forward_features(img).to(pdtype)  # 把backbone模型作为特征抽取器
        #print(feature.shape)
        outputs = self.channel_bn(feature)
        outputs = self.layers(outputs)
        return outputs


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--known_data_source', action='store_true', dest='known_data_source', default=True)
    parser.add_argument('--unknown_data_source', action='store_false', dest='known_data_source', default=True)
    parser.add_argument('--dataset_list', type=str, nargs='+',
                            default=['10shot_cifar100_20200721', '10shot_country211_20210924', '10shot_food_101_20211007', '10shot_oxford_iiit_pets_20211007', '10shot_stanford_cars_20211007'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='deit', type=str, metavar='MODEL',
                        help='Type of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=0, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
        
    parser.add_argument('--src', action='store_true') #simple random crop
    parser.add_argument('--flip', type=float, default=None, metavar='PCT',
                        help='flip image, both VerticalFlip and HorizontalFlip')
    
    parser.add_argument('--rotation', type=int, default=None, metavar='PCT',
                        help='image Rotation')
    


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='../test_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# 参数打印
def _print_args(args):
    logger.info('> training arguments:')
    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

def get_grade(pred_path): 
    dataset=args.dataset_list

    with open(pred_path, "r") as f1:
        pred = json.load(f1)

    with open("../test_data/ans_all.json", "r") as f2:
        ans = json.load(f2)

    grade =0 

    for i in range (len(dataset)):
        match_count = 0
        total_count = len(ans[dataset[i]])

        for key, value in ans[dataset[i]].items():
            if key in pred[dataset[i]] and pred[dataset[i]][key] == value:
                match_count += 1

        match_rate = match_count / total_count
        print(f"Grade of {dataset[i]}:  {match_rate * 100}/100")
        grade += match_rate * 100 / len(dataset)
        
    n_param=int(pred["n_parameters"])
    grade= grade*np.exp(-np.log10(n_param / 1e8 + 1))
    print("Grade: "+ str(grade)+ "/100")
    wandb.log({"grade": grade})

def main(args):
    
    # print(args.dataset_list)
    utils.init_distributed_mode(args)
    
    log_file =  '{}/{}-{}.log'.format(args.output_dir, args.model, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    #print(args)
    logger.info('=======hyper-parameter used========')
    _print_args(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # 导入数据
    # args.nb_classes is the sum of number of classes for all datasets
    dataset_train, args.nb_classes, class_list = build_dataset(is_train=True, args=args)
    dataset_val, *_ = build_dataset(is_train=False, args=args)
    
    # 定义采样方式
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        # 是否分布式校验
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 训练集dataloader
    if args.known_data_source :
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_sampler=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=operator.itemgetter(0), 
        )
        # 知道数据来源，按照idx group
        data_loader_train = GroupedDataset(data_loader_train, args.batch_size, len(args.dataset_list))
    else :
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

    # 验证集dataloader
    data_loader_val_list = []
    dataset_val_total = dataset_val
    for dataset_val in dataset_val.dataset_list:
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(2 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_val_list.append(data_loader_val)

    logger.info(f"Creating model: {args.model}")
    
    # 定义模型
    # Create a model with the timm function; any other model pre-trained under ImageNet-1k is allowed.
    model = create_model(args.model, num_classes=args.nb_classes, pretrained=True)
    
    # number of classes for each dataset
    multi_dataset_classes = [len(x) for x in dataset_train.classes_list]

    # 根据backbone不同来定义Classifier
    if 'deit' in args.model:
        args.model_type = 'deit'
    elif 'efficientnet' in args.model:
        args.model_type = 'efficientnet'
    model = CustomClassifier(model, model.num_features, args.nb_classes, args.model_type, multi_dataset_classes=multi_dataset_classes, known_data_source=args.known_data_source)
                    
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info('number of params:{}'.format(n_parameters) )

    backbone_params = [x for name, x in model_without_ddp.named_parameters() if 'backbone' in name]
    custom_params = [x for name, x in model_without_ddp.named_parameters() if 'backbone' not in name]

    # use smaller lr for backbone params
    params = [
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': custom_params}
    ]
    # 定义优化器
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_scaler = NativeScaler()
    # 定义lr_schduler
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # loss设置
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    
    # 评估：测试
    if args.test_only:
        # the format of submitted json
        # {
        #   'n_parameters': n_parameters,
        #   'dataset_1': {id_1:pred_1, id_2:pred2, ...},
        #    ...,
        #   'dataset_n': ...,
        # }
        pred_path = str(output_dir) + "/" + "pred_all.json"
        result_list = {}
        result_list['n_parameters'] = n_parameters
        for dataset_id, data_loader_val in enumerate(data_loader_val_list):
            pred_json = test(data_loader_val, model, device, dataset_id, num_classes_list=multi_dataset_classes, str_classes_list=class_list,
                                know_dataset=args.known_data_source)
            result_list[args.dataset_list[dataset_id]] = pred_json
        with open(pred_path, 'w') as f:
            json.dump(result_list, f)

        get_grade(pred_path)
        return

    # 评估：验证
    if args.eval:
        for dataset_id, data_loader_val in enumerate(data_loader_val_list):
            test_stats = evaluate(data_loader_val, model, device, dataset_id)
            logger.info(f"Accuracy of the network on {args.dataset_list[dataset_id]} of {len(dataset_val_total.dataset_list[dataset_id])} "
                    f"test images: {test_stats['acc1']:.1f}%")

        return

    # 开始训练
    logger.info("===========start training: {} epochs===========".format(args.epochs))
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        # TODO: Consistent lr now
        # how to use a lr scheduler for better convergence.
        # checkpoint保存
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             
        # 训练过程中验证
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs :
            test_stats_total = {}
            test_stats_list = []
            for dataset_id, data_loader_val in enumerate(data_loader_val_list):
                test_stats = evaluate(data_loader_val, model, device, dataset_id)
                test_stats_list.append(test_stats)
                logger.info(f"Accuracy of the network on {args.dataset_list[dataset_id]} of {len(dataset_val_total.dataset_list[dataset_id])} test images: {test_stats['acc1']:.1f}%")
                test_stats_one = {}
                for k, v in test_stats.items():
                    test_stats_one[k] = v
                test_stats_total['{}'.format(args.dataset_list[dataset_id])] = test_stats_one

            sum_acc = sum([x['acc1'] for x in test_stats_list])
            if max_accuracy < sum_acc:
                max_accuracy = sum_acc
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
                
            logger.info(f'Maxsum accuracy: {max_accuracy:.2f}%')

            logger.info("{:-^100s}".format("Best current test accuracy"))
            #logger.info("test_dataset \t\t\t loss \t\t acc1 \t\t acc5")
            logger.info( "%-35s\t\t%s\t\t%s\t\t%s\n"%("test_dataset","loss","acc1","acc5")+"-" * 100)
            for k,test_stats in test_stats_total.items():
                logger.info("%-35s\t\t%.4f\t\t%.4f\t\t%.4f"%(k, test_stats["loss"], test_stats["acc1"], test_stats["acc5"]))
                #logger.info('>> test_{}:\t\t\t {:.4f} \t\t {:.4f} \t\t {:.4f}'.format(k, test_stats["loss"], test_stats["acc1"], test_stats["acc5"]))
            logger.info("-"*100)
            '''log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats_total.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")'''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="bigdan",
        
        # track hyperparameters and run metadata
        config={
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dataset": args.dataset_list,
        "epochs": args.epochs,
        "model": args.model,
        "input_size": args.input_size,
        "weight_decay": args.weight_decay,
        "color_jitter": args.color_jitter,
        "flip": args.flip,
        "rotation": args.rotation,
        "seed": args.seed,
        "known_data_source": args.known_data_source,
        }
    )
    main(args)



    args.test_only = True
    args.resume = '{}/best_checkpoint.pth'.format(args.output_dir)
    args.data_path = '../test_data/'

    main(args)

    wandb.finish()
    