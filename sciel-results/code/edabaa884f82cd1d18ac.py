import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    # Partition:
    # i: 0,1 (size 2), 2 (size 1)
    # j: 0,1 (size 2), 2,3 (size 2)
    # k: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a block
    def add_block(i_start, j_start, k_start, i_size, j_size, k_size, type_block):
        nonlocal col
        if type_block == "strassen":
            # 2x2x2 Strassen
            # U indices: i_start..i_start+1, j_start..j_start+1
            # V indices: j_start..j_start+1, k_start..k_start+1
            # W indices: k_start..k_start+1, i_start..i_start+1
            
            # M1
            U[i_start*m + j_start, col] += 1
            U[i_start*m + j_start+1, col] += 1
            U[i_start+1*m + j_start, col] += 1
            U[i_start+1*m + j_start+1, col] += 1
            V[j_start*p + k_start, col] += 1
            V[j_start*p + k_start+1, col] += 1
            V[j_start+1*p + k_start, col] += 1
            V[j_start+1*p + k_start+1, col] += 1
            W[k_start*n + i_start, col] += 1
            W[k_start*n + i_start+1, col] += 1
            W[k_start+1*n + i_start, col] += 1
            W[k_start+1*n + i_start+1, col] += 1
            col += 1
            
            # M2
            U[i_start+1*m + j_start, col] += 1
            U[i_start+1*m + j_start+1, col] += 1
            V[j_start*p + k_start, col] += 1
            W[k_start*n + i_start+1, col] += 1
            W[k_start+1*n + i_start+1, col] += 1
            col += 1
            
            # M3
            U[i_start*m + j_start, col] += 1
            V[j_start+1*p + k_start, col] += 1
            V[j_start+1*p + k_start+1, col] -= 1
            W[k_start*n + i_start, col] += 1
            W[k_start+1*n + i_start, col] -= 1
            col += 1
            
            # M4
            U[i_start+1*m + j_start+1, col] += 1
            V[j_start*p + k_start, col] += 1
            V[j_start*p + k_start+1, col] -= 1
            W[k_start*n + i_start, col] -= 1
            W[k_start+1*n + i_start, col] -= 1
            col += 1
            
            # M5
            U[i_start*m + j_start, col] += 1
            U[i_start*m + j_start+1, col] += 1
            V[j_start+1*p + k_start+1, col] += 1
            W[k_start*n + i_start, col] -= 1
            W[k_start+1*n + i_start, col] += 1
            col += 1
            
            # M6
            U[i_start+1*m + j_start, col] += 1
            U[i_start*m + j_start, col] -= 1
            V[j_start*p + k_start, col] += 1
            V[j_start*p + k_start+1, col] += 1
            W[k_start*n + i_start+1, col] += 1
            W[k_start+1*n + i_start+1, col] += 1
            col += 1
            
            # M7
            U[i_start*m + j_start+1, col] += 1
            U[i_start+1*m + j_start+1, col] -= 1
            V[j_start+1*p + k_start, col] += 1
            V[j_start+1*p + k_start+1, col] += 1
            W[k_start*n + i_start, col] -= 1
            W[k_start+1*n + i_start, col] += 1
            col += 1
        elif type_block == "schoolbook_2x2x1":
            # 2x2x1
            for ii in range(2):
                for jj in range(2):
                    U[(i_start+ii)*m + (j_start+jj), col] = 1
                    V[(j_start+jj)*p + k_start, col] = 1
                    W[k_start*n + (i_start+ii), col] = 1
                    col += 1
        elif type_block == "schoolbook_1x2x2":
            # 1x2x2
            for jj in range(2):
                for kk in range(2):
                    U[i_start*m + (j_start+jj), col] = 1
                    V[(j_start+jj)*p + (k_start+kk), col] = 1
                    W[(k_start+kk)*n + i_start, col] = 1
                    col += 1
        elif type_block == "schoolbook_1x2x1":
            # 1x2x1
            for jj in range(2):
                U[i_start*m + (j_start+jj), col] = 1
                V[(j_start+jj)*p + k_start, col] = 1
                W[k_start*n + i_start, col] = 1
                col += 1

    # i=0,1 (size 2)
    # j=0,1 (size 2), k=0,1 (size 2) -> Strassen
    add_block(0, 0, 0, 2, 2, 2, "strassen")
    # j=2,3 (size 2), k=0,1 (size 2) -> Strassen
    add_block(0, 2, 0, 2, 2, 2, "strassen")
    # j=0,1 (size 2), k=2,3 (size 2) -> Strassen
    add_block(0, 0, 2, 2, 2, 2, "strassen")
    # j=2,3 (size 2), k=2,3 (size 2) -> Strassen
    add_block(0, 2, 2, 2, 2, 2, "strassen")
    # j=0,1 (size 2), k=4 (size 1) -> Schoolbook 2x2x1
    add_block(0, 0, 4, 2, 2, 1, "schoolbook_2x2x1")
    # j=2,3 (size 2), k=4 (size 1) -> Schoolbook 2x2x1
    add_block(0, 2, 4, 2, 2, 1, "schoolbook_2x2x1")
    
    # i=2 (size 1)
    # j=0,1 (size 2), k=0,1 (size 2) -> Schoolbook 1x2x2
    add_block(2, 0, 0, 1, 2, 2, "schoolbook_1x2x2")
    # j=2,3 (size 2), k=0,1 (size 2) -> Schoolbook 1x2x2
    add_block(2, 2, 0, 1, 2, 2, "schoolbook_1x2x2")
    # j=0,1 (size 2), k=2,3 (size 2) -> Schoolbook 1x2x2
    add_block(2, 0, 2, 1, 2, 2, "schoolbook_1x2x2")
    # j=2,3 (size 2), k=2,3 (size 2) -> Schoolbook 1x2x2
    add_block(2, 2, 2, 1, 2, 2, "schoolbook_1x2x2")
    # j=0,1 (size 2), k=4 (size 1) -> Schoolbook 1x2x1
    add_block(2, 0, 4, 1, 2, 1, "schoolbook_1x2x1")
    # j=2,3 (size 2), k=4 (size 1) -> Schoolbook 1x2x1
    add_block(2, 2, 4, 1, 2, 1, "schoolbook_1x2x1")
    
    return {"U": U, "V": V, "W": W}

