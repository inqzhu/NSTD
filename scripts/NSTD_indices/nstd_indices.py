#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
computing NSTD indices and relevant basic indices
"""

import json
import math
import time
import networkx as nx
import pandas as pd

#################################
# Basic Functions
#################################

def get_entropy(u_topic):
    # compute topic entropy
    H = 0.0
    for i in range(0, len(u_topic)):
        H = H - u_topic[i]*math.log(u_topic[i])
    return H

def compute_entropy():
    # output topic entropy for each users

    t0 = time.time()
    with open('nstd_user_topic', 'r') as f:
        lda_vec = json.loads(f.read())

    user_topic_entropy = {}
    for user, vec in lda_vec.items():
        user_topic_entropy[user] = get_entropy(vec)

    with open('nstd_user_topic_entropy', 'w') as fout:
        fout.write(json.dumps(user_topic_entropy))
    t1 = time.time()
    print("(%.2fs)" % (t1-t0))

def get_similarity(f_topic, l_topic):
    # obtain the similarity between two topic vectors
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    f_mean = float(sum(f_topic))/len(f_topic)
    l_mean = float(sum(l_topic))/len(l_topic)
    for i in range(0, len(f_topic)):
        tp_f = f_topic[i]-f_mean
        tp_l = l_topic[i]-l_mean
        sum1 = sum1 + tp_f*tp_l
        sum2 = sum2 + tp_f*tp_f
        sum3 = sum3 + tp_l*tp_l
    res = sum1/math.sqrt(sum2*sum3)
    return res

def filter_users(user_set, op):
    # extract users from a set 
    new_set = []
    if op == 'out':
        for t in user_set:
            new_set.append(t[1])
    elif op == 'in':
        for t in user_set:
            new_set.append(t[0])
    return new_set

def get_overlap(set_1, set_2):
    # obtain overlap users from two sets
    over_lap = []
    for i in range(0, len(set_1)):
        if set_1[i] in set_2:
            over_lap.append(set_1[i])
    return over_lap

def get_dict_value(dic, key):
    # extract value from a dictionary
    if dic.has_key(key):
        return dic[key]
    else:
        return 0

def get_topic(dic, uid):
    # obtain topic vector of a user
    sig = False
    u_topic = []
    if dic.has_key(uid):
        u_topic = dic[uid]
        sig = True
    return [sig, u_topic]

def set_remove(set_1, set_2):
    # remove elements of set_2 from set_1
    new_set = []
    for s in set_1:
        if s not in set_2:
            new_set.append(set_1)
    return new_set

#################################
# NSTD indice: Homophily-based index
#################################

def get_homophily_index(user_i, user_j, lda_vec):
    # compute NSTD index H_1
    
    # input: id of user_i, 
    #        id of user_j, 
    #        topic vectors
    # output: Homophily-based index
    vec_i = lda_vec[user_i]
    vec_j = lda_vec[user_j]
    h = get_similarity(vec_i, vec_j)
    return h

#################################
# NSTD indice: Transitivity-based indices
#################################

def get_transitivity_indices(user_i, user_j, user_topic_entropy, g, transitivity_type):
    # compute Transitivity-based indices
    
    # input: id of user_i, 
    #          id of user_j, 
    #          user topic entropies, 
    #          network (a networkx file), 
    #          transitivity_type ('type_1', ..., 'type_4', which refer to the 4 types of transitivity in Section 3)
    # output: Transitivity-based indices (the number of common friends, the mean and the variance of topic entropies of common friends)
    
    # obtain common friends
    if transitivity_type == 'type_1':
        cf_i = filter_users(g.out_edges(user_i), 'out')
        cf_j = filter_users(g.out_edges(user_j), 'out')
    elif transitivity_type == 'type_2':
        cf_i = filter_users(g.out_edges(user_i), 'out')
        cf_j = filter_users(g.in_edges(user_j), 'in')
    elif transitivity_type == 'type_3':
        cf_i = filter_users(g.in_edges(user_i), 'in')
        cf_j = filter_users(g.out_edges(user_j), 'out')        
    elif transitivity_type == 'type_4':
        cf_i = filter_users(g.in_edges(user_i), 'in')
        cf_j = filter_users(g.in_edges(user_j), 'in')   
        
    common_friends = overlap(cf_i, cf_j)
    
    # the number of common friends
    t1 = len(common_friends)
    
    topic_entropies = []
    for cf in common_friends:
        topic_entropies.append(get_dict_value(user_topic_entropy, str(cf)))
    topic_entropies = pd.Series(topic_entropies)
    
    if t1 > 0:
        # the mean of topic entropies of common friends
        t2 = topic_entropies.mean()
        if t1 > 1:
            # the variance of topic entropies of common friends
            t3 = topic_entropies.std()
        else:
            t3 = 0
    else:
        t2 = None
        t3 = None
    
    return [t1, t2, t3]

#################################
# NSTD indice: Clustering-based indices
#################################

def get_clustering_indices(user_i, user_j, lda_vec, g, relation_type):
    # compute Clustering-based indices
    
    # input: id of user_i, 
    #          id of user_j, 
    #          user topic vectors, 
    #          network (a networkx file), 
    #          relation_type ('mutual', 'onesided_from', 'onesided_to')
    # output: Clustering-based indices (the mean, maximum of topic similarities between user j and user i's close friends)
    
    # obtain common friends
    if relation_type == 'mutual':
        in_i = filter_users(g.in_edges(user_i), 'in')
        out_i = filter_users(g.out_edges(user_i), 'out')
        close_friends = overlap(in_i, out_i)
    elif relation_type == 'onesided_from':
        in_i = filter_users(g.in_edges(user_i), 'in')
        out_i = filter_users(g.out_edges(user_i), 'out')
        close_friends = set_remove(out_i, in_i)
    elif relation_type == 'onesided_to':
        in_i = filter_users(g.in_edges(user_i), 'in')
        out_i = filter_users(g.out_edges(user_i), 'out')
        close_friends = set_remove(in_i, out_i)
    
    topic_similarities = []
    [sig_j, user_j_topic] = get_topic(lda_vec, str(user_j))
    for cf in close_friends:
        [sig, u_topic] = get_topic(lda_vec, str(cf))
        if sig:
            topic_similarities.append(get_similarity(user_j_topic, u_topic))
    topic_similarities = pd.Series(topic_similarities)
    
    if len(topic_similarities) > 0:
        # the mean of topic similarities between user j and user i's close friends
        c1 = topic_similarities.mean()
        # the maximum of topic similarities between user j and user i's close friends
        c2 = topic_similarities.max()
    else:
        c1 = None
        c2 = None
    
    return [c1, c2]

#################################
# NSTD indice: Degree-heterogeneity-based indices
#################################

def get_degree_heterogeneity_indices(user_i, user_j, user_topic_entropy, lda_vec, g):
    # compute Degree-heterogeneity-based indices
    
    # input: id of user_i, 
    #          id of user_j, 
    #          user topic entropies, 
    #          user topic vectors,
    #          network (a networkx file)
    # output: Degree-heterogeneity-based indices (the mean and the variance of topic entropies of user j's followers)
    
    followers_j = filter_users(g.in_edges(user_j), 'in')
    topic_entropies = []
    for cf in followers_j:
        topic_entropies.append(get_dict_value(user_topic_entropy, str(cf)))
    topic_entropies = pd.Series(topic_entropies)   
    if len(followers_j) > 0:
        # the mean of topic entropies of user j's followers
        d_a1 = topic_entropies.mean()
        if len(followers_j) > 1:
            # the variance of topic entropies of user j's followers
            d_a2 = topic_entropies.std()
        else:
            d_a2 = None
    else:
        d_a1 = None
        d_d2 = None
        
    topic_similarities = []
    [sig_i, user_i_topic] = get_topic(lda_vec, str(user_i))
    for cf in followers_j:
        [sig, u_topic] = get_topic(lda_vec, str(cf))
        if sig:
            topic_similarities.append(get_similarity(user_i_topic, u_topic))
    topic_similarities = pd.Series(topic_similarities)
    
    if len(topic_similarities) > 0:
        # the mean of topic similarities between user i and user j's followers
        d_b1 = topic_similarities.mean()
        # the maximum of topic similarities between user i and user j's followers
        d_b2 = topic_similarities.max()
    else:
        d_b1 = None
        d_b2 = None
    
    return [d_a1, d_a2, d_b1, d_b2]

if __name__=='__main__':

    """
    preparation works
    """
    # load the topic vectors
    with open('nstd_user_topic', 'r') as f:
        lda_vec = json.loads(f.read())

    # load the network (a networkx file, the output of the script 'network_process.py' )
    g = nx.read_gml("nstd_nx.file")

    # output topic entropy for each users
    compute_entropy()

    # load topic entropies of users
    with open('nstd_user_topic_entropy', 'r') as f:
        user_topic_entropy = json.loads(f.read())

    """
    examples to compute NSTD indices
    """
    user_i = '210****495'
    user_j = '141****485'

    # 1. compute Homophily-based index
    h1 = get_homophily_index(user_i, user_j, lda_vec)

    # 2. compute Transitivity-based indices
    # 'type_1' - 'type_4' refer to the 4 types of transitivity:
    # 'type_1': i -> k, j -> k (here k denotes a kind of common friend for user i and user j)
    [t1_type_1, t2_type_1, t3_type_1] = get_transitivity_indices(user_i, user_j, user_topic_entropy, g, 'type_1')
    # 'type_2': i -> k, k -> j
    [t1_type_2, t2_type_2, t3_type_2] = get_transitivity_indices(user_i, user_j, user_topic_entropy, g, 'type_2')
    # 'type_3': k -> i, j -> k
    [t1_type_3, t2_type_3, t3_type_3] = get_transitivity_indices(user_i, user_j, user_topic_entropy, g, 'type_3')
    # 'type_4': k -> i, k -> j
    [t1_type_4, t2_type_4, t3_type_4] = get_transitivity_indices(user_i, user_j, user_topic_entropy, g, 'type_4')

    # 3. Clustering-based indices
    # 'murual', 'onesided_from' and 'onesided_to' refer to the 3 types of within-cluster relationships
    # mutual relationship: i -> k, k -> i
    [c1_mutual, c2_mutual] = get_clustering_indices(user_i, user_j, lda_vec, g, 'mutual')
    # onesided_from user i: i -> k, k -/-> i
    [c1_onesided_from, c2_onesided_from] = get_clustering_indices(user_i, user_j, lda_vec, g, 'onesided_from')
    # onesided_to user i: i -/-> k, k -> i
    [c1_onesided_to, c2_onesided_to] = get_clustering_indices(user_i, user_j, lda_vec, g, 'onesided_to')

    # 4. Degree-heterogeneity-based indices
    [d_a1, d_a2, d_b1, d_b2] = get_degree_heterogeneity_indices(user_i, user_j, user_topic_entropy, lda_vec, g)





