source sh/para.sh

# task 1
for ((i=0;i<5;i+=1))
do
python main.py --data-path ${data_path} --dataset_list ${dataset_list[i]} --model ${model} --output_dir ${output_dir_task1[i]} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task1[i]} > ${stdout_log}
done

# task2
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task2} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task2} > ${stdout_log}

# task3
python main.py --data-path ${data_path} --model ${model} --output_dir ${output_dir_task3} --batch-size ${bach_size} --epochs ${epochs} --lr ${lr} --weight-decay ${weight_decay} --test_only --resume ${resume_task3} > ${stdout_log}
