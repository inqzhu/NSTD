#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
processing network data
"""

import networkx as nx
import re
import time

def to_id(_user):
    uid = re.sub('\n', '', _user)
    return uid

def toNX(input_path):
    # initialize a network
    g = nx.DiGraph()

    line = 'OK'
    _cnt = 0
    
    t0 = time.time()
    with open(input_path, 'r') as f:
        while line:
            line = f.readline()
            ns = line.split(',')
            if len(ns) == 2 :
                _from = to_id(ns[0])
                _to = to_id(ns[1])
                e = (_from, _to)
                g.add_edge(*e)

                _cnt = _cnt + 1

            if _cnt % 10000 == 0:
                _t1 = time.time()
                print "** %d, %.2fs" % (_cnt, _t1-t0)
               	#break
        
    nx.write_gml(g, "nstd_nx.file")

    t1 = time.time()
    print "%.2fs" % (t1-t0)


if __name__=='__main__':

    input_path = 'nstd_network'
    # the input is the path of the network data 
    # here the raw data consist of directed links
    # each row refers to a link. 
    # For example, a row "1001,1002" denotes a link between user 1001 and user 1002 
    # (user 1001 is following user 1002)

    # generate a networkx object according to the input network
    # the network will be organized as networkx object for further use
    toNX(input_path)
