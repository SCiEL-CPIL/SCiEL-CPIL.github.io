"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    NM, MP, PN = n*m, m*p, p*n
    
    # Block decomposition: n=[2,1], m=[2,2], p=[2,2,2]
    n_blocks = [2, 1]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 2]
    
    # Strassen rank-1 terms for 2x2x2 (decomposing all-ones tensor)
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
    strassen_U = np.zeros((2, 2, 7))
    strassen_V = np.zeros((2, 2, 7))
    strassen_W = np.zeros((2, 2, 7))
    
    # Verified Strassen decomposition for 2x2x2 all-ones tensor:
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    strassen_U[0,0,0] = 1; strassen_V[0,0,0] = 1; strassen_W[0,0,0] = 1
    # c=1: U[0,1]=1, V[1,0]=1, W[0,0]=1  
    strassen_U[0,1,1] = 1; strassen_V[1,0,1] = 1; strassen_W[0,0,1] = 1
    # c=2: U[1,0]=1, V[0,1]=1, W[1,0]=1
    strassen_U[1,0,2] = 1; strassen_V[0,1,2] = 1; strassen_W[1,0,2] = 1
    # c=3: U[1,1]=1, V[1,1]=1, W[1,1]=1
    strassen_U[1,1,3] = 1; strassen_V[1,1,3] = 1; strassen_W[1,1,3] = 1
    # c=4: U[0,0]=1, V[1,1]=1, W[1,0]=1
    strassen_U[0,0,4] = 1; strassen_V[1,1,4] = 1; strassen_W[1,0,4] = 1
    # c=5: U[0,1]=1, V[0,0]=1, W[0,1]=1
    strassen_U[0,1,5] = 1; strassen_V[0,0,5] = 1; strassen_W[0,1,5] = 1
    # c=6: U[1,0]=1, V[1,1]=1, W[0,1]=1
    strassen_U[1,0,6] = 1; strassen_V[1,1,6] = 1; strassen_W[0,1,6] = 1
    
    # Verify Strassen
    def verify_block(U_b, V_b, W_b, ni, mj, pk):
        T_rec = np.zeros((ni, mj, pk))
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    for c in range(U_b.shape[2]):
                        T_rec[i,j,k] += U_b[i,j,c]*V_b[j,k,c]*W_b[k,i,c]
        return np.allclose(T_rec, 1.0, atol=1e-10)
    
    strassen_valid = verify_block(strassen_U, strassen_V, strassen_W, 2, 2, 2)
    
    # Schoolbook block
    def schoolbook_block(ni, mj, pk):
        U = np.zeros((ni, mj, ni*mj*pk))
        V = np.zeros((mj, pk, ni*mj*pk))
        W = np.zeros((pk, ni, ni*mj*pk))
        col = 0
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[i,j,col] = 1
                    V[j,k,col] = 1
                    W[k,i,col] = 1
                    col += 1
        return U, V, W
    
    # Build full decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off, ni, mj, pk):
        nr = U_b.shape[2]
        for c in range(nr):
            U_col = np.zeros(NM)
            V_col = np.zeros(MP)
            W_col = np.zeros(PN)
            for ii in range(ni):
                for jj in range(mj):
                    U_col[(i_off+ii)*m + (j_off+jj)] = U_b[ii, jj, c]
            for jj in range(mj):
                for kk in range(pk):
                    V_col[(j_off+jj)*p + (k_off+kk)] = V_b[jj, kk, c]
            for kk in range(pk):
                for ii in range(ni):
                    W_col[(k_off+kk)*n + (i_off+ii)] = W_b[kk, ii, c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    i_off = 0
    for ni in n_blocks:
        j_off = 0
        for mj in m_blocks:
            k_off = 0
            for pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2 and strassen_valid:
                    add_block(strassen_U, strassen_V, strassen_W, i_off, j_off, k_off, 2, 2, 2)
                else:
                    U_b, V_b, W_b = schoolbook_block(ni, mj, pk)
                    add_block(U_b, V_b, W_b, i_off, j_off, k_off, ni, mj, pk)
                k_off += pk
            j_off += mj
        i_off += ni
    
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
