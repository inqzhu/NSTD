#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
text processing for the weibo data
"""

import jieba
import jieba.posseg 
import jieba.analyse
import json
import time
import re
import pandas as pd
import numpy as np
import lda

#################################
# Part-1: word segmentation
#################################

def text_clean(st):
    # conduct data cleanning for a given text
    _st = re.sub(r'http|\[|\]', '', st)
    return _st

def get_keywords(st):
    # extract keywords
    all_words = []
    sentence = st

    # word segmentation
    words = jieba.posseg.cut(text_clean(sentence))

    for word in words:
        # keep only meaningful words, thus clear out neglectable words
        if word.flag in ['n', 'ns', 'nh','nr', 'nt', 'nz', 'ni', 'nl', 'nd', 'v', 'vd', 'vn']:
            if word.word not in all_words:
                all_words.append(word.word)

    return all_words

def drop_na(s):
    # drop duplicated items
    pds = pd.Series(s)
    pds = pds.drop_duplicates()
    pds = list(pds)
    return pds

def word_count_stat(word_list):
    # count each word
    aw = pd.Series(word_list)
    vc = aw.value_counts()
    return [list(vc.index), list(vc)]

def work_segmentation(weibo_data_path):
    #  segment all text to words

    jieba.initialize()    
    jieba.enable_parallel(4)

    # global vocabulary
    global_words = []

    # word counts for each user
    word_count_user = {}

    t0 = time.time()
    with open(weibo_data_path, 'r') as f:
        # input the weibo data
        weibo_js = json.loads(f.read())

        user_weibo = weibo_js.items()
        n_users = len(user_weibo)
        cnt = 0

        # for each user, all published texts are stored in a list.
        for user, weibo_list in user_weibo:
            # store all words of the UGC
            user_words = []
            #print len(weibo_list)

            # for each text, conduct data cleanning and word segmentation
            for weibo in weibo_list:
                user_words = user_words + get_keywords(weibo)
                
            # for each user, record the count of each word he or she has mentioned
            [user_words, word_counts]= word_count_stat(user_words)
            word_count_user[user] = [user_words, word_counts]
            global_words = global_words + user_words

            cnt += 1
            if cnt % 1000 == 0:
                _t1 = time.time()
                print "** processing %.2f %% (%.2fs)" % (100*cnt/float(n_users), _t1-t0)
                #break
        
    t1 = time.time()
    print "(%.2fs)" % (t1-t0)

    # drop duplicated words for the global vocabulary
    global_words = drop_na(global_words)
    print "global vocabulary: %d" % len(global_words)

    with open('nstd_word_count', 'w') as fout:
        fout.write(json.dumps(word_count_user))

    with open('nstd_global_vocabulary', 'w') as fout:
        fout.write(json.dumps({'words': global_words}))   

#################################
# Part-2: generate word vectors
#################################

def vec(sample, all_words):
    # generate a vector of counts of words
    v = np.zeros(len(all_words), dtype=int)
    for i in range(0, len(sample[0])):
        if sample[0][i] in all_words:
            v[all_words.index(sample[0][i])] = sample[1][i]
    #print v.sum(), len(sample[0])
    #print v
    #print v[:100]
    return v

def word_to_vector():
    # transform all word_count to vector
    t0 = time.time()

    with open('nstd_global_vocabulary', 'r') as f:
        global_words = json.loads(f.read())['words']

    with open('nstd_word_count', 'r') as f:
        word_count_user_json = json.loads(f.read())

        word_count_user = word_count_user_json.items()
        n_users = len(word_count_user)
        
        # record word vectors
        vecs = {}

        cnt = 0
        for user, count in word_count_user:
            _vec = vec(count, global_words)
            if _vec.sum() > 0:
                vecs[user] = list(_vec)
            cnt += 1

            if cnt % 10 == 0:
                _t1 = time.time()
                print "** processing %.2f %% (%.2fs)" % (100*cnt/float(n_users), _t1-t0)
                #break

    t1 = time.time()
    print "(%.2fs)" % (t1-t0)

    with open('nstd_word_vector', 'w') as fout:
        fout.write(json.dumps(vecs))

#################################
# Part-3: LDA model
#################################

def weibo_LDA():
    # conduct LDA

    train_name = []
    train_X = []

    t0 = time.time()
    with open('nstd_word_vector', 'r') as f:
        vecs = json.loads(f.read())
        for user in vecs:
            train_name.append(user)
            train_X.append(vecs[user])   
    t1 = time.time()        
    print "reading vectors OK. (%.2fs)" % (t1-t0)

    train_X = np.asarray(train_X)
    train_X = train_X.astype('int64')

    model = lda.LDA(n_topics=100, n_iter=200, random_state=1)
    model.fit(train_X)

    # output the distribution on K topics for each user
    doc_topic = model.doc_topic_

    res_dic = {}
    for i in range(0, len(train_name)):
        res_dic[train_name[i]] = list(doc_topic[i])

    with open('nstd_user_topic', 'w') as fout:
        fout.write(json.dumps(res_dic))
    t2 = time.time()
    print "(%.2fs)" % (t2-t1)

if __name__=='__main__':
    
    input_path = 'nstd_weibo'
    # the input is the path of the ugc data (such as weibo)
    # the input textual data are organized as a [diction] (i.e. JSON file), where the key
    # is [user id], the value is [ the list of his or her published contents ]. 
    # For example: {'1001':[ 'aaa', 'bbb', ''', 'ccc']}

    # segment all text to words
    work_segmentation(input_path)

    # transform all word_count to vector
    word_to_vector()

    # conduct LDA for weibo data
    weibo_LDA()

    