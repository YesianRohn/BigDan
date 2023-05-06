output_name="deit0501"
epochs="50"
lr="1e-4"
weight-decay="0.01"
model_name="efficientnet_b3"

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/cifar100 --dataset_list 10shot_cifar100_20200721 --model ${model_name}
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/cifar100 --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --dataset_list 10shot_cifar100_20200721 --resume output/${output_name}/cifar100/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/country211 --dataset_list 10shot_country211_20210924
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/country211 --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --dataset_list 10shot_country211_20210924 --resume output/${output_name}/country211/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/food --dataset_list 10shot_food_101_20211007
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/food --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --dataset_list 10shot_food_101_20211007 --resume output/${output_name}/food/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/oxford --dataset_list 10shot_oxford_iiit_pets_20211007
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/oxford --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --dataset_list 10shot_oxford_iiit_pets_20211007 --resume output/${output_name}/oxford/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/deit/stanford --dataset_list 10shot_stanford_cars_20211007
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/deit/stanford --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --dataset_list 10shot_stanford_cars_20211007 --resume output/${output_name}/stanford/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/known_data_source --known_data_source
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/known_data_source --epochs 50 --lr 1e-4 --weight-decay 0.01 --known_data_source --test_only --resume output/${output_name}/known_data_source/best_checkpoint.pth

python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final --epochs 50 --lr 1e-4 --weight-decay 0.01 --output_dir output/${output_name}/unknown_data_source --unknown_data_source
python main.py --batch-size 64 --data-path ../../share/course23/aicourse_dataset_final/ --output_dir output/${output_name}/unknown_data_source --epochs 50 --lr 1e-4 --weight-decay 0.01 --unknown_data_source --test_only --resume output/${output_name}/unknown_data_source/best_checkpoint.pth
