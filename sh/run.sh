source sh/para.sh

# task2
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task2} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}

# task3
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task3} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} > ${stdout_log}

