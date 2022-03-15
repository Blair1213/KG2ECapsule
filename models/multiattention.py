# -*- coding: utf-8 -*-
# @Time    : 2021-01-15 15:40
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : XML.py
# @Software : PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021-01-14 18:48
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : multiattention.py
# @Software : PyCharm
# -*- coding: utf-8 -*-


from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve,precision_score,recall_score
import sklearn.metrics as m
from layers import Aggregator
from callbacks import KGCNMetric
import tensorflow as tf
from models.base_model import BaseModel
from keras.engine.topology import Layer
epsilon = 1e-9




class MultiAttention(BaseModel):
    def __init__(self, config):
        super(MultiAttention, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            #drugID
            shape=(1, ), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            #drugID
            shape=(1, ), name='input_drug_two', dtype='int64')
        input_drug_relation = Input(
            # real relation
            shape=(1,), name='input_drug_relation', dtype='int64')

        #trainable parameter entity embedding and relation embedding
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')

        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')

        drug_one_embedding = entity_embedding(input_drug_one)
        ##multi attention
        receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x),
                                         name='receptive_filed_drug_one')(input_drug_one)

        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth + 1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth + 1:]

        ###embedding list
        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]
        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding_drug_one')

        for depth in range(self.config.n_depth):


            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}_drug_one'
            )

            next_neigh_ent_embed_list_drug_one = [drug_one_embedding]
            for hop in range(self.config.n_depth-depth):
                print(hop)
                neighbor_embed = neighbor_embedding([drug_one_embedding, neigh_rel_embed_list_drug_one[hop],
                                                         neigh_ent_embed_list_drug_one[hop + 1]])
                ##计算各自的attention即可
                #next_entity_embed = aggregator(
                #    [next_neigh_ent_embed_list_drug_one[hop-1], neighbor_embed])
                #next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
                next_neigh_ent_embed_list_drug_one.append(neighbor_embed)

            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one


        ##drug two
        drug_two_embedding = entity_embedding(input_drug_two)
        receptive_list = Lambda(lambda x: self.get_receptive_field(x),
                                name='receptive_filed')(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth + 1]
        neigh_rel_list = receptive_list[self.config.n_depth + 1:]

        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]
        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding')
        for depth in range(self.config.n_depth):


            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}_drug_one'
            )

            next_neigh_ent_embed_list = [drug_two_embedding]
            for hop in range(self.config.n_depth - depth):
                print(hop)
                neighbor_embed = neighbor_embedding([drug_two_embedding, neigh_rel_embed_list[hop],
                     neigh_ent_embed_list[hop + 1]])
                ##计算各自的attention即可
                #next_entity_embed = aggregator(
                #    [next_neigh_ent_embed_list[hop-1], neighbor_embed])
                #next_neigh_ent_embed_list.append(next_entity_embed)
                next_neigh_ent_embed_list.append(neighbor_embed)

            neigh_ent_embed_list = next_neigh_ent_embed_list

        ##drug 1 representation neigh[1]
        ##drug 2 representation neigh[2]
        drug_relation_embedding = relation_embedding(input_drug_relation)

        drug1_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list_drug_one[0])
        drug2_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list[0])
        #drug_drug_score = Lambda(
        #    lambda x: K.sigmoid(K.sum(x[0] * x[1], axis=-1, keepdims=True))
        #)([drug1_squeeze_embed, drug2_squeeze_embed])

        capsule_score = Lambda(lambda x: self.capsule(x[0], x[1], x[2]), name="capsule")(
            [drug1_squeeze_embed, drug2_squeeze_embed, drug_relation_embedding])

        drug_drug_score = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(capsule_score)

        model = Model([input_drug_one, input_drug_two, input_drug_relation], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])

        return model

    def get_receptive_field(self, entity):
        """Calculate receptive field for entity using adjacent matrix

        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list;

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param drug: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score

        ## drug_self_attention [batch_size, neigh_size**hop, 1]
        ## converting_info [batch_size,neigh_size**hop,embed_dim]
        print("drug")
        print(drug)
        self_attention_weight_drug = K.sum(drug * rel, axis=-1, keepdims=True)
        print(self_attention_weight_drug)
        converting_info = self_attention_weight_drug * drug
        print(converting_info)

        ##ent_self_attention [batch_size, neighbor_size**hop, 1]
        ##receiving_info [batch_size, neighbor_size**hop, embed_dim]
        self_attention_weight_ent = K.sum(rel * ent, axis = -1, keepdims= True)
        receiving_info = self_attention_weight_ent * ent
        print(self_attention_weight_ent)
        print(receiving_info)

        ##current_node_info [batch_size, neighbor_size**hop, embed_dim]
        current_info = receiving_info + converting_info
        print(current_info)

        sending_attention_weight = K.sum(drug * ent, axis = -1, keepdims=True)
        final_info = sending_attention_weight * current_info
        print(final_info)


        weighted_ent = K.reshape(current_info,
                                 (K.shape(current_info)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))
        print(weighted_ent)

        neighbor_embed = K.sum(weighted_ent, axis=2)
        print(neighbor_embed)

        return neighbor_embed

    def get_relation_embedding(self, relationid):


        ##relationid: a param of [,neighbor_number ** hop]

        ##relationid is a list
        print("getting relation embedding")
        print(relationid)
        neigh_embedding_list = []
        relation_matrix = K.variable(self.config.relation_vector,name="relation_vector",dtype='float32')


        for i in range(self.config.n_depth):
            ##choose relation type
            relation_number = relationid[i]
            relation_embedding = K.gather(relation_matrix,K.cast(relation_number,dtype='int64'))
            print("relation embedding")
            print(relation_embedding)
            neigh_embedding_list.append(relation_embedding)


        return neigh_embedding_list;

    def get_drug_relation(self, relationid):

        relation_matrix = K.variable(self.config.relation_vector,name="relation_vector",dtype='float32')
        relation_embedding = K.gather(relation_matrix, K.cast(relationid, dtype='int64'))

        return relation_embedding;

    def capsule(self,drug1,drug2,relation):

        ##drug: shape(?,?,32)
        ##relation: shape(?,1,32)
        print("drug")
        print(drug1)
        print(relation)
        drug1 = K.reshape(drug1,(-1,self.config.embed_dim))
        drug2 = K.reshape(drug2,(-1,self.config.embed_dim))
        relation = K.reshape(relation,(-1,self.config.embed_dim))


        ##shape[?,32]
        ##shape[?,32]
        print(relation*drug1)
        w_1 = K.reshape(K.sum(relation*drug1,axis=-1),(-1,1))
        w_2 = K.reshape(K.sum(relation*drug2,axis=-1),(-1,1))

        print(w_1)
        print(w_2)
        print("u_hat")
        u_hat = w_1*drug1 + w_2*drug2           ##shape[?,32]
        #u_hat = K.sum((w_1*drug1,w_2*drug2),axis=0)  ##shape[?,32]
        print(u_hat)

        ##routing algorithm
        ##capsule_number = self.config.embed_size
        ##interation step = self.config.iters
        capsule_output = self.routing_algorithm(u_hat)

        return capsule_output;

    def routing_algorithm(self,u_hat):

        #capsule_number = self.config.embed_dim
        ##u_hat = [?,32]
        interation = self.config.iters
        print("interation")
        print(interation)
        b = K.variable(self.config.B_matrix,name="B_matrix")   ##shape[32,1]
        #b = K.zeros_like(u_hat)
        #b = K.reshape(b,(-1,32,1))

        for iterations in range(0,interation):
            print(iterations)
            c = K.softmax(b) ##shape[32,1]
            print(c)
            S = K.dot(u_hat,c)  ##shape(?,1) 1
            print(S)
            V = self.squash(S)
            print(V)   ##shape(?,1)
            V_U_h = K.reshape(V*u_hat,(-1,self.config.embed_dim,1))  ##3
            print(V_U_h)    ##shape[?,32,1]
            b = b + V_U_h

        #V = K.reshape(V,(-1,1))
        return V;

    def squash(self,vector):

        ###vector = [?,1]

        vec_squared_norm = K.sum(K.square(vector), axis = -1, keepdims=True) + K.epsilon()
        print("vec_squared_norm")
        print(vec_squared_norm)  #shape[1,1]  should be [?,1]
        scalar_factor = K.sqrt(vec_squared_norm) / (0.5 + vec_squared_norm)
        print("scalar_factor")
        print(scalar_factor) ##shape[1,1]
        vec_squashed = scalar_factor * vector  # element-wise [?,1]  2
        #vec_squashed = K.squeeze(vec_squashed, axis=3)
        print(vec_squashed)

        return vec_squashed;


    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        #precision = precision_score()
        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, p,r, f1, aupr, fpr.tolist(), tpr.tolist()
