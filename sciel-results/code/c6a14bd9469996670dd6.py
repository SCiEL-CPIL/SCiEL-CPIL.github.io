import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    U_list = []
    V_list = []
    W_list = []
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[u_idx(i, j)] = 1
                    v[v_idx(j, k)] = 1
                    w[w_idx(k, i)] = 1
                    add_term(u, v, w)

    def add_strassen_2x2x2(i0, j0, k0):
        # 7 terms for 2x2x2 Strassen
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0, k0)] = 1
        w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1
        v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
        w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0+1)] = 1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
        w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
        v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
        v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
        w[w_idx(k0+1, i0+1)] = 1
        add_term(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
        v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
        w[w_idx(k0, i0)] = 1
        add_term(u, v, w)

    # Strategy: Try multiple block decompositions and pick the best rank
    # Split 5 into different partitions
    
    partitions = [
        [[0], [1, 2], [3, 4]],       # 1+2+2
        [[0], [1], [2, 3, 4]],       # 1+1+3
        [[0, 1], [2, 3], [4]],       # 2+2+1
        [[0, 1], [2], [3, 4]],       # 2+1+2
        [[0], [1, 2, 3], [4]],       # 1+3+1
        [[0, 1, 2], [3], [4]],       # 3+1+1
    ]
    
    best_rank = float('inf')
    best_U = best_V = best_W = None
    
    for blocks in partitions:
        if time.time() > deadline:
            break
            
        U_list.clear()
        V_list.clear()
        W_list.clear()
        
        for ib in blocks:
            for jb in blocks:
                for kb in blocks:
                    if time.time() > deadline:
                        break
                    
                    i_start, i_end = min(ib), max(ib)
                    j_start, j_end = min(jb), max(jb)
                    k_start, k_end = min(kb), max(kb)
                    
                    i_size = len(ib)
                    j_size = len(jb)
                    k_size = len(kb)
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        add_strassen_2x2x2(i_start, j_start, k_start)
                    else:
                        add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end)
        
        rank = len(U_list)
        
        if rank < best_rank:
            best_rank = rank
            best_U = [row[:] for row in U_list]
            best_V = [row[:] for row in V_list]
            best_W = [row[:] for row in W_list]
    
    # If we haven't found anything, use schoolbook
    if best_U is None:
        U_list.clear()
        V_list.clear()
        W_list.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[u_idx(i, j)] = 1
                    v[v_idx(j, k)] = 1
                    w[w_idx(k, i)] = 1
                    add_term(u, v, w)
        best_rank = len(U_list)
        best_U = U_list
        best_V = V_list
        best_W = W_list
    
    rank = best_rank
    
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            if best_U[r][i] != 0:
                U_mat[i][r] = best_U[r][i]
        for j in range(m*p):
            if best_V[r][j] != 0:
                V_mat[j][r] = best_V[r][j]
        for k in range(p*n):
            if best_W[r][k] != 0:
                W_mat[k][r] = best_W[r][k]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
