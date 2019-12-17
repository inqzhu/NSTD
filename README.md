# Indices combining network structure and topic distribution (NSTD)

NSTD indices are proposed for link prediction on directed networks. First, characteristics of social networks, including homophily, transitivity, clustering and degree heterogeneity are considered for directed networks. Second, those indices extend previous similarity measurements for link prediction by integrating users' topic distributions on user generated contents (UGCs). The UGCs are analyzed based on the Latent Dirichlet Allocation (LDA) model. 



`ugc_process.py` is used for the preprocessing of UGCs. The input textual data are recommended to be organized as a dictionary (such as JSON file), where each tuple is (user id, the list of his or her published contents). 

  
  > For example: {'1001':[ 'aaa', 'bbb', ''', 'ccc']}  
  > { <br/>
  >   '1001':[ 'aaa', 'bbb', ..., 'ccc'], <br/>
  >   '1002':[ 'aaa', 'ccc', ..., 'ddd'], <br/>
  >    ...<br/>
  > } <br/>
  
The output is a dictionary of topic vectors based on the LDA model. In the output, each tuple is (user_id, topic_vector).

`network_process.py` is used for the preprocessing of network structural data. The input are recommended to be organized as a list of directed links. Each raw of the data refers to an existing link.

   > For example, a row "1001,1002" denotes a link between user 1001 and user 1002 (user 1001 is following user 1002)
   > 1001, 1002 <br/>
   > 1002, 1003 <br/>
   > 1001, 1004 <br/>
   > ...
 
The output is a network file (a networkx object).

`nstd_indices.py` is used for the computation of NSTD indices. The input consists of the dictionary of topic vectors (the output of `ugc_process.py`) and the networkx object (the output of `network_process.py`). The script nstd_indices.py provides functions to compute each NSTD indices. Examples of the useage of the indices are given in `nstd_indices.py`.
