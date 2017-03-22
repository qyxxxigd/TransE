
# Copyright Mar 2017 YUE QI. All Rights Reserved.

from transE_base import *
from parameters import *
import os

# create a TransE object for training
transE = TransE()

# load dataset infomation
path = '.' + os.sep + 'data' + os.sep + dataset_name
str_file_dataset_info = path + os.sep + 'out' + os.sep + 'info_' + str(f_discard) + '.txt'
transE.load_dataset(str_file_dataset_info)

# load embeddings to initialize
str_paras = str(dict_paras['embedding_size']) + '_' + str(dict_paras['regular_type']) + '_' + \
            str(dict_paras['learning_rate']) + '_' + str(dict_paras['num_steps']) + '_' + \
            str(dict_paras['batch_size']) + '_' + str(dict_paras['margin'])
str_file_embedding_in = path + os.sep + 'out' + os.sep + str_paras + '.txt'
transE.load_embeddings(str_file_embedding_in)

# training process
transE.dict_paras = dict_paras
transE.dict_paras['num_steps'] = num_steps_add
transE.train(retrain=1)

# save embeddings
str_paras = str(dict_paras['embedding_size']) + '_' + str(dict_paras['regular_type']) + '_' + \
            str(dict_paras['learning_rate']) + '_' + str(dict_paras['num_steps']+num_steps_add) + '_' + \
            str(dict_paras['batch_size']) + '_' + str(dict_paras['margin'])
str_file_embedding_out = path + os.sep + 'out' + os.sep + str_paras + '.txt'
transE.save_embeddings(str_file_embedding_out)
