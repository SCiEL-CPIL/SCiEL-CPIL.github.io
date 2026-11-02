import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    def add_rank1_2x2x1(j_start, k_val):
        # Adds 1 term for 2x2x1 block at j in {j_start, j_start+1}, k=k_val
        # T_{i,j,k_val} = 1 for i in {0,1}, j in {j_start, j_start+1}
        u = {(0, j_start): 1, (1, j_start+1): 1}
        v = {(j_start, k_val): 1, (j_start+1, k_val): 1}
        w = {(k_val, 0): 1, (k_val, 1): 1}
        terms.append((u, v, w))

    def add_rank1_2x1x5(j_val):
        # Adds 1 term for 2x1x5 block at j=j_val, k in {0..4}
        # T_{i,j_val,k} = 1 for i in {0,1}, k in {0..4}
        u = {(0, j_val): 1, (1, j_val): 1}
        v = {}
        for k in range(5):
            v[(j_val, k)] = 1
        w = {}
        for k in range(5):
            for i in range(2):
                w[(k, i)] = 1
        terms.append((u, v, w))

    # Decomposition plan:
    # 1. j in {0, 1}
    #    1a. k in {0, 1}: Strassen(0, 0, 0)
    add_strassen(0, 0, 0)
    #    1b. k in {2, 3}: Strassen(0, 0, 2)
    add_strassen(0, 0, 2)
    #    1c. k in {4}: Rank1 2x2x1
    add_rank1_2x2x1(0, 4)
    
    # 2. j in {2, 3}
    #    2a. k in {0, 1}: Strassen(0, 2, 0)
    add_strassen(0, 2, 0)
    #    2b. k in {2, 3}: Strassen(0, 2, 2)
    add_strassen(0, 2, 2)
    #    2c. k in {4}: Rank1 2x2x1
    add_rank1_2x2x1(2, 4)
    
    # 3. j in {4}
    #    3a. k in {0..4}: Rank1 2x1x5
    add_rank1_2x1x5(4)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
