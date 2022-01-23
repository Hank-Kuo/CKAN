import collections
import os
import random
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(args):
    logging.info("================== preparing data ===================")
    max_user_history_item = args.history_item* args.sample_triplet
    train_data, test_data, user_init_entity_set, item_init_entity_set = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, max_user_history_item, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, max_user_history_item, False)

    return train_data, test_data, n_entity, n_relation, max_user_history_item, \
            kg, user_init_entity_set, item_init_entity_set, user_triple_sets, item_triple_sets


def load_rating(args):
    rating_file = './data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(args, rating_np)


def dataset_split(args, rating_np):
    # 處理 user history item 
    k = pd.DataFrame(rating_np, columns=['user', 'item', 'label'])
    value_counts = k[k['label']==1]['user'].value_counts()
    to_remove = value_counts[value_counts <= args.history_item].index
    df = k[~k['user'].isin(to_remove)]
    user_history = df[df['label']==1].groupby('user').apply(lambda x: x.sample(args.history_item,random_state=555))
    user_history_id = [y for x, y in user_history.index.tolist()]
    df = df.drop(user_history_id)

    user_history_dict = dict()
    item_neighbor_item_dict = dict()
    item_history_dict = dict()
    # user history  -> user 多少 seed 與 item 多少 seed
    for i, u in enumerate(list(set(user_history['user'].tolist()))):
        
        sub = user_history[user_history['user']==u]['item'].tolist()

        if u not in user_history_dict:
            user_history_dict[u] = []
        user_history_dict[u].extend(sub)

        for item in sub:
            if item not in item_history_dict:
                item_history_dict[item] = []
            item_history_dict[item].extend([u])

    # 拿到 user neibhbor 的 item
    for item in item_history_dict.keys():
        item_nerghbor_item = []
        for user in item_history_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(df['item'].tolist())
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]

    train_data, test_data = train_test_split(df.to_numpy(), train_size=0.8, random_state=42)

    return train_data, test_data, user_history_dict, item_neighbor_item_dict
    

def load_kg(args):
    kg_file = './data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        temp_set = set()
        for l in range(args.n_hop):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
                temp_set = set(entities)
            else:
                entities = triple_sets[obj][-1][2]
                temp_set = set.union(temp_set, set(entities))

            for entity in entities:
                old_triplets = kg[entity]
                triplets = []
                
                for i in old_triplets:
                    tail = i[0]
                    if tail not in temp_set:
                        triplets.append(i)

                if len(triplets) < args.sample_triplet:
                    sample_triplets = triplets
                else:
                    sample_triplets = random.sample(triplets, args.sample_triplet)
                
                for tail_and_relation in sample_triplets:
                    h.append(entity)
                    r.append(tail_and_relation[1])
                    t.append(tail_and_relation[0])
                
                    
            if len(h) == 0:
                triple_sets[obj].append(([0]*set_size, [0]*set_size, [0]*set_size))
            else:
                replace = len(h) > set_size
                if replace == True:
                    indices = np.random.choice(len(h), size=set_size, replace=False)
                else:
                    indices = np.random.choice(len(h), size=len(h), replace=False)
                
                # padding 
                l = max(0, set_size - len(h))
                h = [h[i] for i in indices] + [0]*l
                r = [r[i] for i in indices] + [0]*l
                t = [t[i] for i in indices] + [0]*l

                triple_sets[obj].append((h, r, t))
    return triple_sets