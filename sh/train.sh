source sh/env.sh

# model
model="${model_list[1]}" # CHANGE ME

# path
output_dir="output/${model}"
output_dir_task1=("${output_dir}/${dataset_list[0]}" "${output_dir}/${dataset_list[1]}" "${output_dir}/${dataset_list[2]}" "${output_dir}/${dataset_list[3]}" "${output_dir}/${dataset_list[4]}")
output_dir_task2="${output_dir}/known_data_source"
output_dir_task3="${output_dir}/unknown_data_source"
resume="best_checkpoint.pth"
resume_task1=("${output_dir_task1[0]}/${resume}" "${output_dir_task1[1]}/${resume}" "${output_dir_task1[2]}/${resume}" "${output_dir_task1[3]}/${resume}" "${output_dir_task1[4]}/${resume}")
resume_task2="${output_dir_task2}/${resume}"
resume_task3="${output_dir_task3}/${resume}"
stdout_log="output/train_stdout.log"

# hyper parameters
bach_size="64"
epochs="50"
lr="1e-4"
weight_decay="0.01"

# task 1
for ((i=4;i<5;i+=1))
do
python main.py --device ${train_device} --data-path ${train_data_path} --dataset_list ${dataset_list[i]} --model ${model} --output_dir ${output_dir_task1[i]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}
done

# task2
python main.py --device ${train_device} --data-path ${train_data_path} --model ${model} --output_dir ${output_dir_task2} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}

# task3
python main.py --device ${train_device} --data-path ${train_data_path} --model ${model} --output_dir ${output_dir_task3} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}

