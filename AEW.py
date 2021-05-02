import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

from pandas import DataFrame
from numpy import linalg as LA
from scipy import sparse
from networkx.algorithms.bipartite import biadjacency_matrix
from networkx.algorithms import bipartite

def generate_W(players, lineups, ratings):
    """
    This function generates sparse matrix represantation of bipartite graph
    for players and linups
    Input:
      players - pandas series
      lineups - pandas series
      ratings - pandas series
    Output:
      sparse biadjacency matrix W 
    """
    # generate graph
    BG = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    BG.add_nodes_from(lineups, bipartite=0)
    BG.add_nodes_from(players, bipartite=1)

    # add weighted edges
    for row in players:
        for row2 in lineups:  
            if row in row2:
                index = lineups[lineups.isin([row2])].index[0]
                BG.add_edges_from([(row, row2)])
                BG[row][row2]['weight'] = ratings[index]


    # Biadjacency matrix - for bipartite network
    W = biadjacency_matrix(BG, players, lineups).toarray()

    # sparce form of Biadjacency matrix
    W = sparse.csr_matrix(W)
    print('Shape of W: '+str(W.shape))

    return W

def AEW(W, u0=None, v0=None,
        alpha=0.5, beta=0.5, max_iter=200, tol=1.0e-4, verbose=False):
  
    """
    This function calculates AEW ranking vectors based on
    Input:
      W - sparse weighted biadjacency matrix of graph
      u0, v0 - normalized initial scoring vetors values 
      alpha, beta - impact of graph structure 
      max_iter =  maximum number of iteration 
      tol = covergence criterua
      verbose = displaying algorithm progress 
    Output:
         u, v::numpy.ndarray:The BiRank for rows and columns     
    """
    # default initial vectors: 1/|U|, 1/|V|
    if u0 is None:
        u0=np.repeat(1 / W.shape[0], W.shape[0])
    if v0 is None:
        v0=np.repeat(1 / W.shape[1], W.shape[1])
    

    W = W.astype('float', copy=False)
    WT = W.T

    # Calculate Su, Sv
    Ku = np.count_nonzero(W.toarray(), axis = 1)
    Kv = np.count_nonzero(W.toarray(), axis = 0)
    Ku_ = sparse.diags(1/Ku)
    Kv_ = sparse.diags(1/Kv)
    Su = Ku_.dot(W)
    Sv = Kv_.dot(W.T)


    u_last = u0.copy()
    v_last = v0.copy()

    for i in range(max_iter):

        # calculate update
        u_temp = Su.dot(v_last)
        v_temp = Sv.dot(u_last)

        # normalize
        u_temp = u_temp / u_temp.sum()
        v_temp = v_temp / v_temp.sum()

        # consider query vector
        u = alpha * u_temp + (1-alpha) * u0
        v = beta * v_temp + (1-beta) * v0
        
        # calculate change from i-1
        err_u = np.absolute(u - u_last).sum()
        err_v = np.absolute(v - v_last).sum()       

        if verbose:
            print(
                "Iteration : {}; top error: {}; bottom error: {}".format(
                    i, err_u, err_v
                )
            )

        if err_v < tol and err_u < tol:
            break
        u_last = u
        v_last = v

    return u, v

def AEW_hist(W, u0=None, v0=None,
        alpha=0.5, beta=0.5, max_iter=200, tol=1.0e-4, verbose=False):
  
    """
    This function return whole history of AEW ranking vectors
    throughout iterations based on
    Input:
      W - sparse weighted biadjacency matrix of graph
      u0, v0 - normalized initial scoring vetors values 
      alpha, beta - impact of graph structure 
      max_iter =  maximum number of iteration 
      tol = covergence criterua
      verbose = displaying algorithm progress 
    Output:
         U_hist, V_hist::numpy.ndarray:The BiRank for rows and columns     
    """
    # default initial vectors: 1/|U|, 1/|V|
    if u0 is None:
        u0=np.repeat(1 / W.shape[0], W.shape[0])
    if v0 is None:
        v0=np.repeat(1 / W.shape[1], W.shape[1])
    

    W = W.astype('float', copy=False)
    WT = W.T

    # Calculate Su, Sv
    Ku = np.count_nonzero(W.toarray(), axis = 1)
    Kv = np.count_nonzero(W.toarray(), axis = 0)
    Ku_ = sparse.diags(1/Ku)
    Kv_ = sparse.diags(1/Kv)
    Su = Ku_.dot(W)
    Sv = Kv_.dot(W.T)


    u_last = u0.copy()
    v_last = v0.copy()
    U_hist = u_last
    V_hist = v_last

    for i in range(max_iter):

        # calculate update
        u_temp = Su.dot(v_last)
        v_temp = Sv.dot(u_last)

        # normalize
        u_temp = u_temp / u_temp.sum()
        v_temp = v_temp / v_temp.sum()

        # consider query vector
        u = alpha * u_temp + (1-alpha) * u0
        v = beta * v_temp + (1-beta) * v0
        
        # calculate change from i-1
        err_u = np.absolute(u - u_last).sum()
        err_v = np.absolute(v - v_last).sum()       

        if verbose:
            print(
                "Iteration : {}; top error: {}; bottom error: {}".format(
                    i, err_u, err_v
                )
            )
        # if err_v < tol and err_u < tol:
        #     break
        
        U_hist = np.vstack((U_hist,u))
        V_hist = np.vstack((V_hist,v))

        u_last = u
        v_last = v

    return U_hist, V_hist