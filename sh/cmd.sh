# path
data_path="../../share/course23/aicourse_dataset_final"
dataset_list=("10shot_cifar100_20200721" "10shot_country211_20210924" "10shot_food_101_20211007" "10shot_oxford_iiit_pets_20211007" "10shot_stanford_cars_20211007")
model_list=("deit_tiny_patch16_224" "efficientnet_b3") # ADD ME
model="${model_list[1]}" # CHANGE ME
output_dir="output/${model}"
output_dir_task1=("${output_dir}/${dataset_list[0]}" "${output_dir}/${dataset_list[1]}" "${output_dir}/${dataset_list[2]}" "${output_dir}/${dataset_list[3]}" "${output_dir}/${dataset_list[4]}")
output_dir_task2="${output_dir}/known_data_source"
output_dir_task3="${output_dir}/unknown_data_source"
resume="best_checkpoint.pth"
resume_task1=("${output_dir_task1[0]}/${resume}" "${output_dir_task1[1]}/${resume}" "${output_dir_task1[2]}/${resume}" "${output_dir_task1[3]}/${resume}" "${output_dir_task1[4]}/${resume}")
resume_task2="${output_dir_task2}/${resume}"
resume_task3="${output_dir_task3}/${resume}"
stdout_log="stdout.log"

# hyper parameters
bach_size="64"
epochs="50"
lr="1e-4"
weight_decay="0.01"

# task1-1
python main.py --data-path ${data_path} --dataset_list ${dataset_list[0]} --model ${model} --output_dir ${output_dir_task1[0]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --dataset_list ${dataset_list[0]} --model ${model} --output_dir ${output_dir_task1[0]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[0]} > ${stdout_log}

# task1-2
python main.py --data-path ${data_path} --dataset_list ${dataset_list[1]} --model ${model} --output_dir ${output_dir_task1[1]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --dataset_list ${dataset_list[1]} --model ${model} --output_dir ${output_dir_task1[1]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[1]} > ${stdout_log}

# task1-3
python main.py --data-path ${data_path} --dataset_list ${dataset_list[2]} --model ${model} --output_dir ${output_dir_task1[2]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --dataset_list ${dataset_list[2]} --model ${model} --output_dir ${output_dir_task1[2]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[2]} > ${stdout_log}

# task1-4
python main.py --data-path ${data_path} --dataset_list ${dataset_list[3]} --model ${model} --output_dir ${output_dir_task1[3]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --dataset_list ${dataset_list[3]} --model ${model} --output_dir ${output_dir_task1[3]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[3]} > ${stdout_log}

# task1-5
python main.py --data-path ${data_path} --dataset_list ${dataset_list[4]} --model ${model} --output_dir ${output_dir_task1[4]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --dataset_list ${dataset_list[4]} --model ${model} --output_dir ${output_dir_task1[4]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[4]} > ${stdout_log}

# task2
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task2} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task2} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task2} > ${stdout_log}

# task3
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task3} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task3} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task3} > ${stdout_log}
