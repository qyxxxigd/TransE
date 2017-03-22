
# Copyright Mar 2017 YUE QI. All Rights Reserved.

# global parameters for training and testing

# select a dataset
dataset = [('WN11', 1), ('WN18', 0), ('FB13', 1), ('FB15k', 0)]
dataset_id = 1
dataset_name, dataset_label = dataset[dataset_id]
f_discard = 2

# training parameters
dict_paras = {'embedding_size': 20,
              'regular_type': 2,
              'learning_rate': 0.1,
              'num_steps': 200000,
              'batch_size': 60,
              'margin': 1}

num_steps_add = 5000  # retrain

# testing parameters
head_or_tail = 0  # link prediction
filt = 1

theta = 0.5  # triplet classification

num_to_test = 10  # total test num
