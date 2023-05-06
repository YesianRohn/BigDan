# BigDan
人工智能课程大炼丹炉

Artificial Intelligence Course (COMP130031.02) Project of Fudan University.
Competiton link: https://codalab.lisn.upsaclay.fr/competitions/12725?secret_key=64e1ce9f-10fd-4a9a-b07c-f94d81946931

## AiCourse-Baseline

### Installation 

Our code base is developed and tested with PyTorch 1.7.0, TorchVision 0.8.1, CUDA 10.2, and Python 3.7.

```Shell
conda create -n baseline python=3.7 -y
conda activate baseline
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt 
```

### Model

Loading pre-trained weights is allowed. You can use the pre-trained model under **ImageNet-1k**, while other datasets like ImageNet-21k, CC3M, LAION, etc., are not allowed.

### Datasets

Five datasets are given, which include:
'10shot_cifar100_20200721','10shot_country211_20210924','10shot_food_101_20211007','10shot_oxford_iiit_pets_20211007','10shot_stanford_cars_20211007'                        

### Run

The executable pretrained models are offered by ```timm```. You can check and use the offered pretrained timm models. 

```Shell
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/baseline --epochs 50 --lr 1e-4 --weight-decay 0.01
```

There are three modes to execute the code.

1. Operate on individual dataset seperately. You can change ```--dataset_list``` to achieve it.
2. Operate on known datasets. The dataset which given images belong to will be offered. You can check the ```--known_data_source``` option. 
3. Operate on unknown datasets. The dataset which given image belong to will not be offered. You should predict both **datasets that images belong to** and **images' corresponding labels**. You can check the ```--unknown_data_source``` option.

After obtaining the checkpoint of certain modes, you should operate ```--test_only``` to produce a prediction json file ```pred_all.json```. The file will be produced under your output directory. 

```Shell
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/baseline --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only
```

### Submit

You should submit a zip file containing the ```pred_all.json``` file into the colab website. 

