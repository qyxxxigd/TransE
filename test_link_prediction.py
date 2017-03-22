
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
str_file_embedding = path + os.sep + 'out' + os.sep + str_paras + '.txt'
transE.load_embeddings(str_file_embedding)

# load test data
str_file_test = path + os.sep + 'test.txt'
transE.open_test(str_file_test, flag_label=dataset_label, discard_negative=1)

# testing process
mean_rank, hit_10 = \
    transE.link_prediction(transE.lst_triplet_train_map, head_or_tail=head_or_tail, filt=filt, num_to_test=num_to_test)
print('mean_rank:{}'.format(mean_rank))
print('hit@10:{}'.format(hit_10))

# write result to file
str_file_result = path + os.sep + 'out' +os.sep + 'result_lp.txt'
with open(str_file_result, 'a') as f_write:
    f_write.write(str(dict_paras))
    f_write.write('head or tail:' + str(head_or_tail))
    f_write.write('filt:' + str(filt))
    f_write.write('num_to_test' + str(num_to_test))
    f_write.write('mean rank: ' + str(mean_rank))
    f_write.write('hits@10: ' + str(hit_10))
    f_write.write('\n')




