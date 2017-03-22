
# Copyright Mar 2017 YUE QI. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import random
import operator
import collections
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class TransE:

    """ class for TransE model """

    def __init__(self):

        # inputs
        self.lst_entity = list()  # [e1, e2, ...]
        self.lst_relation = list()  # [r1, r2, ...]
        self.lst_triplet_train = list()  # [(h1,r1,t1), (h2,r2,t2), ...]
        self.lst_triplet_test = list()
        self.lst_label = list()

        # data mapping
        self.lst_entity_map = list()  # [id_e1, id_e2, ...]
        self.lst_relation_map = list()  # [id_r1, id_r2, ...]
        self.lst_triplet_train_map = list()  # [(id_h1,id_r1,id_t1), (id_h2,id_r2,id_t2), ...]
        self.lst_triplet_test_map = list()

        # lexicon
        self.dictionary_entity = dict()  # {e1:id_e1, e2:id_e2, ...}
        self.dictionary_relation = dict()  # {r1:id_r1, r2:id_r2, ...}
        self.lst_triplet_corrupted_head = dict()
        self.lst_triplet_corrupted_tail = dict()
        self.reverse_dictionary_entity = dict()  # {id_e1:e1, id_ew:e2, ...}
        self.reverse_dictionary_relation = dict()  # {id_e1:e1, id_ew:e2, ...}
        self.dict_tofh = dict()  # {r1:{h1:[t1, t2, ...], h2:{...}}, r2:{...}}
        self.dict_hoft = dict()  # {r1:{t1:[h1, h2, ...], t2:{...}}, r2:{...}}

        # size
        self.entity_size = 0
        self.relation_size = 0
        self.triplet_train_size = 0
        self.triplet_test_size = 0

        # model parameters
        self.embeddings_entity = list()  # embeddings for all entities, size of [entity_size+1, dimension]
        self.embeddings_relation = list()  # embeddings for all relations, size of [relation_size, dimension]
        self.dict_paras = dict()  # all parameters for training

        # self.data_index = 0  # global index for generating batch

    def build_dataset(self):

        """ build dictionary and map data """

        print('begin to build data set...')

        # build dictionary
        for e in self.lst_entity:
            self.dictionary_entity[e] = len(self.dictionary_entity)

        for r in self.lst_relation:
            self.dictionary_relation[r] = len(self.dictionary_relation)

        self.reverse_dictionary_entity = dict(zip(self.dictionary_entity.values(), self.dictionary_entity.keys()))
        self.reverse_dictionary_relation = dict(zip(self.dictionary_relation.values(), self.dictionary_relation.keys()))

        # build data map
        self.lst_entity_map = [self.dictionary_entity[e] for e in self.lst_entity]
        self.lst_relation_map = [self.dictionary_relation[r] for r in self.lst_relation]
        self.lst_triplet_train_map = self.map_triplet(self.lst_triplet_train)

        # build corrupted candidates for (h,r,~) and (~,r,t)
        for (h, r, t) in self.lst_triplet_train_map:
            if r not in self.dict_tofh:
                self.dict_tofh[r] = {h: [t]}
            else:
                if h not in self.dict_tofh[r]:
                    self.dict_tofh[r][h] = [t]
                else:
                    self.dict_tofh[r][h].append(t)

            if r not in self.dict_hoft:
                self.dict_hoft[r] = {t: [h]}
            else:
                if t not in self.dict_hoft[r]:
                    self.dict_hoft[r][t] = [h]
                else:
                    self.dict_hoft[r][t].append(h)

        for r in self.dict_tofh:
            self.lst_triplet_corrupted_tail[r] = dict()
            for h in self.dict_tofh[r]:
                set_tail_corrupted_all = set(self.lst_entity_map) - set(self.dict_tofh[r][h])
                lst_tail_corrupted_choose = random.sample(set_tail_corrupted_all, 5*len(self.dict_tofh[r][h]))
                self.lst_triplet_corrupted_tail[r][h] = lst_tail_corrupted_choose

        for r in self.dict_hoft:
            self.lst_triplet_corrupted_head[r] = dict()
            for t in self.dict_hoft[r]:
                lst_head_corrupted_all = set(self.lst_entity_map) - set(self.dict_hoft[r][t])
                lst_head_corrupted_choose = random.sample(lst_head_corrupted_all, 5*len(self.dict_hoft[r][t]))
                self.lst_triplet_corrupted_head[r][t] = lst_head_corrupted_choose

        print('data set has been built successfully!')

    def map_triplet(self, lst_triplet):
        lst_triplet_map = list()
        for (h, r, t) in lst_triplet:
            h_map = self.dictionary_entity[h]
            r_map = self.dictionary_relation[r]
            t_map = self.dictionary_entity[t]
            lst_triplet_map.append((h_map, r_map, t_map))
        return lst_triplet_map

    def get_corrupted_triplet(self, triplet):

        """ produce corrupted triplet by replacing head or tail with random entity """

        (h, r, t) = triplet

        # i = np.random.uniform(-0.5, 0.5)

        # if i < 0:  # corrupt head entity
        entity_temp = random.sample(self.lst_triplet_corrupted_head[r][t], 1)[0]
        corrupted_triplet1 = (entity_temp, r, t)
        # else:  # corrupt tail entity
        entity_temp = random.sample(self.lst_triplet_corrupted_tail[r][h], 1)[0]
        corrupted_triplet2 = (h, r, entity_temp)
        return corrupted_triplet1, corrupted_triplet2

    def generate_batch(self):

        """ generate batch for training """

        # sbatch = list()
        # tbatch = list()
        # for i in range(self.dict_paras['batch_size']):
        #     sbatch.append(self.lst_triplet_train_map[self.data_index])
        # self.data_index = (self.data_index + 1) % self.triplet_train_size

        sbatch = random.sample(self.lst_triplet_train_map, self.dict_paras['batch_size'])
        tbatch = list()

        for ele in sbatch:
            corrupted1, corrupted2 = self.get_corrupted_triplet(ele)
            tbatch.append((ele, corrupted1))
            tbatch.append((ele, corrupted2))
        return tbatch

    def distance(self, vec1, vec2):

        """ compute distance of two vectors in different ways """
        regular_type = self.dict_paras['regular_type']
        if regular_type == 1:  # L1-norm
            out = tf.reduce_sum(tf.abs(tf.sub(vec1, vec2)), 1)
        elif regular_type == 2:  # L2-norm
            out = tf.reduce_sum(tf.square(tf.sub(vec1, vec2)), 1)
        else:  # cosine similarity
            vec1_norm = tf.nn.l2_normalize(vec1, 1)
            vec2_norm = tf.nn.l2_normalize(vec2, 1)
            vec_mul = tf.mul(vec1_norm, vec2_norm)
            out = tf.reduce_sum(vec_mul, 1)
        return out

    def loss_function(self, train_head, train_tail, train_relation, train_head_corrupted, train_tail_corrupted):

        """ loss function for TransE """

        # train_head = tf.nn.l2_normalize(train_head, 1)
        # train_tail = tf.nn.l2_normalize(train_tail, 1)
        # train_head_corrupted = tf.nn.l2_normalize(train_head_corrupted, 1)
        # train_tail_corrupted = tf.nn.l2_normalize(train_tail_corrupted, 1)

        # loss = tf.reduce_mean(
        #     tf.maximum(self.dict_paras['margin']
        #                + self.distance(tf.add(train_head, train_relation), train_tail)
        #                - self.distance(tf.add(train_head_corrupted, train_relation), train_tail_corrupted), 0.))

        loss = tf.reduce_mean(self.distance(tf.add(train_head, train_relation), train_tail))

        return loss

    def train(self, retrain=0):

        """ run this module for training """

        if 'embedding_size' not in self.dict_paras:  # dimension
            self.dict_paras['embedding_size'] = 20
        embedding_size = self.dict_paras['embedding_size']

        if 'learning_rate' not in self.dict_paras:
            self.dict_paras['learning_rate'] = 0.1
        learning_rate = self.dict_paras['learning_rate']

        if 'num_steps' not in self.dict_paras:
            self.dict_paras['num_steps'] = 100000
        num_steps = self.dict_paras['num_steps']

        if 'num_loss_report' not in self.dict_paras:
            self.dict_paras['num_loss_report'] = 1000
        num_loss_report = self.dict_paras['num_loss_report']

        if 'batch_size' not in self.dict_paras:
            self.dict_paras['batch_size'] = 60
        batch_size = self.dict_paras['batch_size']

        # distance type {L1:1, L2:2, cos_similarity:3}
        if 'regular_type' not in self.dict_paras:
            self.dict_paras['regular_type'] = 2

        if 'margin' not in self.dict_paras:
            self.dict_paras['margin'] = 1

        # build graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0'):

                # Input data.
                train_head_id = tf.placeholder(tf.int32, shape=[2*batch_size])
                train_tail_id = tf.placeholder(tf.int32, shape=[2*batch_size])
                train_relation_id = tf.placeholder(tf.int32, shape=[2*batch_size])
                train_head_corrupted_id = tf.placeholder(tf.int32, shape=[2*batch_size])
                train_tail_corrupted_id = tf.placeholder(tf.int32, shape=[2*batch_size])

                # model parameters
                if not retrain:
                    bound = (6./embedding_size)**0.5
                    self.embeddings_entity = tf.Variable(
                        tf.random_uniform([self.entity_size, embedding_size], -bound, bound))
                    self.embeddings_relation = tf.Variable(tf.nn.l2_normalize(
                        tf.random_uniform([self.relation_size, embedding_size], -bound, bound), 1))
                else:
                    self.embeddings_entity = tf.Variable(self.embeddings_entity)
                    self.embeddings_relation = tf.Variable(self.embeddings_relation)

                # normalize all embeddings of entities.
                self.embeddings_entity = tf.nn.l2_normalize(self.embeddings_entity, 1)

                # look up embeddings for inputs

                embedding_head = tf.nn.embedding_lookup(self.embeddings_entity, train_head_id)
                embedding_tail = tf.nn.embedding_lookup(self.embeddings_entity, train_tail_id)
                embedding_relation = tf.nn.embedding_lookup(self.embeddings_relation, train_relation_id)
                embedding_head_corrupted = tf.nn.embedding_lookup(self.embeddings_entity, train_head_corrupted_id)
                embedding_tail_corrupted = tf.nn.embedding_lookup(self.embeddings_entity, train_tail_corrupted_id)

                # loss function
                loss = self.loss_function(embedding_head, embedding_tail, embedding_relation
                                          , embedding_head_corrupted, embedding_tail_corrupted)

                # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

                # Begin training.
                init = tf.initialize_all_variables()

                with tf.Session(graph=graph) as sess:
                    init.run()
                    print("Initialized")

                    average_loss = 0

                    for step in xrange(num_steps):

                        batch = self.generate_batch()

                        head_id = [ele[0][0] for ele in batch]
                        relation_id = [ele[0][1] for ele in batch]
                        tail_id = [ele[0][2] for ele in batch]
                        head_corrupted_id = [ele[1][0] for ele in batch]
                        tail_corrupted_id = [ele[1][2] for ele in batch]

                        feed_dict = {train_head_id: head_id, train_relation_id: relation_id, train_tail_id:tail_id,
                                     train_head_corrupted_id: head_corrupted_id, train_tail_corrupted_id: tail_corrupted_id}

                        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                        average_loss += loss_val

                        if step % num_loss_report == 0:
                            if step > 0:
                                average_loss /= num_loss_report
                                print("Average loss at step ", step, ": ", average_loss)
                                average_loss = 0

                    self.embeddings_entity = tf.nn.l2_normalize(self.embeddings_entity, 1).eval()
                    self.embeddings_relation = self.embeddings_relation.eval()

        print('Training completed!')

    def link_prediction(self, lst_triplet_test_map, head_or_tail=0, filt=1, num_to_test=1000):

        """ link prediction task for testing """

        # begin testing
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                num_hit_10 = 0
                lst_rank_all = list()
                num_tested = 0
                # for i in random.sample(range(self.triplet_test_size), num_to_test):
                #     (h, r, t) = lst_triplet_test_map[i]
                for (h, r, t) in lst_triplet_test_map[:num_to_test]:
                    if head_or_tail == 0:  # rank head
                        embedding_relation = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings_relation, r), 0)
                        embedding_tail = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings_entity, t),0)
                        temp_ones = tf.ones([self.entity_size, 1])
                        embedding_relation_mat = tf.matmul(temp_ones, embedding_relation)
                        embedding_tail_mat = tf.matmul(temp_ones, embedding_tail)
                        distance_vec = self.distance(tf.add(self.embeddings_entity, embedding_relation_mat),
                                                     embedding_tail_mat)
                        lst_entity_rank = sorted(enumerate(distance_vec.eval()), key=operator.itemgetter(1))

                        lst_entity_sorted = [ele[0] for ele in lst_entity_rank]
                        rank = lst_entity_sorted.index(h) + 1
                        if filt:
                            for i in range(rank-1):
                                entity = lst_entity_sorted[i]
                                if entity in self.dict_hoft[r][t]:
                                    rank -= 1

                    else:
                        embedding_head = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings_entity, h), 0)
                        embedding_relation = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings_relation, r), 0)
                        temp_ones = tf.ones([self.entity_size, 1])
                        embedding_head_mat = tf.matmul(temp_ones, embedding_head)
                        embedding_relation_mat = tf.matmul(temp_ones, embedding_relation)
                        distance_vec = self.distance(tf.add(embedding_head_mat, embedding_relation_mat),
                                                     self.embeddings_entity)
                        lst_entity_rank = sorted(enumerate(distance_vec.eval()), key=operator.itemgetter(1))

                        lst_entity_sorted = [ele[0] for ele in lst_entity_rank]
                        rank = lst_entity_sorted.index(t) + 1
                        if filt:
                            for i in range(rank-1):
                                entity = lst_entity_sorted[i]
                                if entity in self.dict_tofh[r][h]:
                                    rank -= 1

                    if rank <= 10:
                        num_hit_10 += 1

                    num_tested += 1
                    print('{} rank:{}, num_hit_10:{}'.format(num_tested, rank, num_hit_10))
                    lst_rank_all.append(rank)

        mean_rank = sum(lst_rank_all) / float(num_to_test)
        hit_10 = num_hit_10 / float(num_to_test) * 100.

        print('Testing completed!')

        return mean_rank, hit_10

    def triplet_classification(self, lst_triplet_test_map, theta, num_to_test=1000):

        num_correct = 0
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                num_tested = 0
                for i in random.sample(range(self.triplet_test_size), num_to_test):
                    (h, r, t) = lst_triplet_test_map[i]
                    label = self.lst_label[i]
                    embedding_head = tf.nn.embedding_lookup(self.embeddings_entity, h)
                    embedding_relation = tf.nn.embedding_lookup(self.embeddings_relation, r)
                    embedding_tail = tf.nn.embedding_lookup(self.embeddings_entity, t)
                    distance_vec = tf.sub(tf.add(embedding_head, embedding_relation), embedding_tail)
                    distance_eval = tf.sqrt(tf.reduce_sum(tf.square(distance_vec)))
                    if (distance_eval.eval() <= theta and label == 1) or \
                            (distance_eval.eval() > theta and label == -1):
                        num_correct += 1

                    num_tested += 1
                    print("{} distance:{}, label:{}".format(num_tested, distance_eval.eval(), label))

        accuracy = num_correct / num_to_test

        print('Testing completed!')

        return accuracy

    def save_dataset(self, str_file):

        """ save the dataset """

        with open(str_file, 'wb') as f_write:
            pickle.dump(self.lst_entity_map, f_write, True)
            pickle.dump(self.lst_relation_map, f_write, True)
            pickle.dump(self.lst_triplet_train_map, f_write, True)
            pickle.dump(self.dictionary_entity, f_write, True)
            pickle.dump(self.dictionary_relation, f_write, True)
            pickle.dump(self.lst_triplet_corrupted_head, f_write, True)
            pickle.dump(self.lst_triplet_corrupted_tail, f_write, True)
            pickle.dump(self.reverse_dictionary_entity, f_write, True)
            pickle.dump(self.reverse_dictionary_relation, f_write, True)
            pickle.dump(self.dict_tofh, f_write, True)
            pickle.dump(self.dict_hoft, f_write, True)
            pickle.dump(self.entity_size, f_write, True)
            pickle.dump(self.relation_size, f_write, True)

    def save_embeddings(self, str_file):

        """ save the embeddings and parameters """

        with open(str_file, 'wb') as f_write:
            pickle.dump(self.embeddings_entity, f_write, True)
            pickle.dump(self.embeddings_relation, f_write, True)
            pickle.dump(self.dict_paras, f_write, True)

    def load_dataset(self, str_file):

        """ load the dataset """

        with open(str_file, 'rb') as f_read:
            self.lst_entity_map = pickle.load(f_read)
            self.lst_relation_map = pickle.load(f_read)
            self.lst_triplet_train_map = pickle.load(f_read)
            self.dictionary_entity = pickle.load(f_read)
            self.dictionary_relation = pickle.load(f_read)
            self.lst_triplet_corrupted_head = pickle.load(f_read)
            self.lst_triplet_corrupted_tail = pickle.load(f_read)
            self.reverse_dictionary_entity = pickle.load(f_read)
            self.reverse_dictionary_relation = pickle.load(f_read)
            self.dict_tofh = pickle.load(f_read)
            self.dict_hoft = pickle.load(f_read)
            self.entity_size = pickle.load(f_read)
            self.relation_size = pickle.load(f_read)

    def load_embeddings(self, str_file):

        """ load the embeddings and parameters """

        with open(str_file, 'rb') as f_read:
            self.embeddings_entity = pickle.load(f_read)
            self.embeddings_relation = pickle.load(f_read)
            self.dict_paras = pickle.load(f_read)

    def open_train(self, str_path_train, f_discard=0):

        """ load training triplets, entity and relation lists from file"""

        lst_triplet_train = list()
        lst_entity = list()
        lst_relation = list()
        lst_infrequent = list()

        with open(str_path_train) as f_train:
            lines = f_train.readlines()
            for line in lines:
                triplet = line.strip().split()
                if len(triplet) < 3:
                    continue
                lst_triplet_train.append(tuple(triplet))
                lst_entity.extend([triplet[0], triplet[2]])
                lst_relation.append(triplet[1])

        self.lst_relation = list(set(lst_relation))

        if not f_discard:
            print(1)
            self.lst_entity = list(set(lst_entity))
            self.lst_triplet_train = lst_triplet_train
        else:
            count = collections.Counter(lst_entity)
            for ele in count:
                if count[ele] > f_discard:
                    self.lst_entity.append(ele)
                else:
                    lst_infrequent.append(ele)
            for (h, r, t) in lst_triplet_train:
                if h in lst_infrequent or t in lst_infrequent:
                    continue
                self.lst_triplet_train.append((h, r, t))

        self.entity_size = len(self.lst_entity)
        self.relation_size = len(self.lst_relation)
        self.triplet_train_size = len(self.lst_triplet_train)

        print('{} training triplets have been collected, including {} entities and {} relations'
              .format(self.triplet_train_size, self.entity_size, self.relation_size))

    def open_test(self, str_path_test, flag_label=0, discard_negative=0):

        """ load and map testing triplets """

        with open(str_path_test) as f_test:
            lines = f_test.readlines()
            for line in lines:
                triplet_with_label = line.strip().split()
                if len(triplet_with_label) < 3:
                    continue
                if (triplet_with_label[0] not in self.dictionary_entity) \
                        or (triplet_with_label[2] not in self.dictionary_entity) \
                        or (triplet_with_label[1] not in self.dictionary_relation):
                    continue
                if flag_label == 0:
                    self.lst_triplet_test.append(tuple(triplet_with_label[0:3]))
                elif flag_label == 1 and len(triplet_with_label) == 4:
                    label = int(triplet_with_label[3])
                    if discard_negative and label == -1:
                        continue
                    self.lst_triplet_test.append(tuple(triplet_with_label[0:3]))
                    self.lst_label.append(triplet_with_label[3])

        self.triplet_test_size = len(self.lst_triplet_test)
        self.lst_triplet_test_map = self.map_triplet(self.lst_triplet_test)

        print('{} testing triplets have been collected'.format(self.triplet_test_size))

