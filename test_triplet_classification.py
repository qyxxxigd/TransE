
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
accuracy = transE.triplet_classification(transE.lst_triplet_test_map, theta=theta, num_to_test=num_to_test)
print('accuracy rate:{}'.format(accuracy))

# write result to file
str_file_result = path + os.sep + 'out' + os.sep + 'result_tc.txt'
with open(str_file_result, 'a') as f_write:
    f_write.write(str(dict_paras))
    f_write.write('theta:' + str(theta))
    f_write.write('num_to_test' + str(num_to_test))
    f_write.write('accuracy rate: ' + str(accuracy))
    f_write.write('\n')




