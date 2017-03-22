
# Copyright Mar 2017 YUE QI. All Rights Reserved.

from transE_base import *
from parameters import *
import os

# create a TransE object for training
transE = TransE()

# create or load dataset infomation
path = '.' + os.sep + 'data' + os.sep + dataset_name
str_file_dataset_info = path + os.sep + 'out' + os.sep + 'info_' + str(f_discard) + '.txt'
if os.path.exists(str_file_dataset_info):
    transE.load_dataset(str_file_dataset_info)
else:
    str_file_train = path + os.sep + 'train.txt'
    transE.open_train(str_file_train, f_discard=f_discard)
    transE.build_dataset()
    transE.save_dataset(str_file_dataset_info)

# set training parameters
transE.dict_paras = dict_paras

# training process
transE.train()

# save embeddings
str_paras = str(dict_paras['embedding_size']) + '_' + str(dict_paras['regular_type']) + '_' + \
            str(dict_paras['learning_rate']) + '_' + str(dict_paras['num_steps']) + '_' + \
            str(dict_paras['batch_size']) + '_' + str(dict_paras['margin'])
str_file_embedding = path + os.sep + 'out' + os.sep + str_paras + '.txt'
transE.save_embeddings(str_file_embedding)
