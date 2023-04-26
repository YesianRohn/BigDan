# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, set_training_mode=True, args = None,
                    class_indicator=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets, dataset_ids = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        with torch.cuda.amp.autocast():
            outputs = model(samples, dataset_ids)
            if class_indicator is not None :
                mask = class_indicator[targets]
                outputs[~mask.bool()] = -1e2

            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dataset_id=None, dump_result=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    result_json = {}

    for data in metric_logger.log_every(data_loader, 10, header):
        images, target = data[:2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, dataset_id)

        if dump_result :
            file_ids = data[-1].tolist()
            pred_labels = output.max(-1)[1].tolist()
            for id, pred_id in zip(file_ids, pred_labels) :
                result_json[id] = pred_id
        else :
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if dump_result :
        return result_json
    else :
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(data_loader, model, device, dataset_id=None, num_classes_list=None, know_dataset=True):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    result_json = {}

    class_start_id_list = []
    start_id = 0
    for num_classes in num_classes_list:
        class_start_id_list.append(start_id)
        start_id += num_classes

    for data in metric_logger.log_every(data_loader, 10, header):
        images, target = data[:2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, dataset_id)
        file_ids = data[-1].tolist()

        if not know_dataset:
            pred_labels = output.max(-1)[1].tolist()
            # map the concated class_id into original class_id
            pred_labels = [x-class_start_id_list[dataset_id] for x in pred_labels]
        else :
            output = output[:, class_start_id_list[dataset_id]:class_start_id_list[dataset_id]+num_classes_list[dataset_id]]
            pred_labels = output.max(-1)[1].tolist()

        for id, pred_id in zip(file_ids, pred_labels) :
            result_json[id] = pred_id

    return result_json
