# env
train_device="cuda:0"
test_device="cuda:1"
train_data_path="../../share/course23/aicourse_dataset_final"
test_data_path="../test_data/"
dataset_list=("10shot_cifar100_20200721" "10shot_country211_20210924" "10shot_food_101_20211007" "10shot_oxford_iiit_pets_20211007" "10shot_stanford_cars_20211007")
model_list=("deit_tiny_patch16_224" "efficientnet_b3") # ADD ME
