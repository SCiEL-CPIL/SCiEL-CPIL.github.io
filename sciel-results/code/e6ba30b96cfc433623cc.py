"""Tensor decomposition for 3x5x6 matrix multiplication - block decomposition."""
import time

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1
    for r in range(len(U[0])):
        for a in range(len(U)):
            if U[a][r] != 0:
                for b in range(len(V)):
                    if V[b][r] != 0:
                        for c in range(len(W)):
                            if W[c][r] != 0:
                                key = (a, b, c)
                                T[key] = T.get(key, 0) - U[a][r] * V[b][r] * W[c][r]
    for val in T.values():
        if abs(val) > 1e-9:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Schoolbook fallback
    U_sb = [[0.0]*rank_sb for _ in range(n*m)]
    V_sb = [[0.0]*rank_sb for _ in range(m*p)]
    W_sb = [[0.0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1.0
                V_sb[j*p+k][col] = 1.0
                W_sb[k*n+i][col] = 1.0
                col += 1
    
    # Block decomposition: split m=5 into [0,1] and [2,3,4], p=6 into [0,1,2] and [3,4,5]
    U_block = []
    V_block = []
    W_block = []
    
    def add_term(u_idx, v_idx, w_idx):
        U_row = [0.0]*(n*m)
        U_row[u_idx] = 1.0
        V_row = [0.0]*(m*p)
        V_row[v_idx] = 1.0
        W_row = [0.0]*(p*n)
        W_row[w_idx] = 1.0
        U_block.append(U_row)
        V_block.append(V_row)
        W_block.append(W_row)
    
    # Block 1: j in [0,1], k in [0,2]: A[i,j]*B[j,k] -> C[i,k]
    for i in range(n):
        for j in range(2):
            for k in range(3):
                add_term(i*m+j, j*p+k, k*n+i)
    
    # Block 2: j in [0,1], k in [3,5]: A[i,j]*B[j,k] -> C[i,k]
    for i in range(n):
        for j in range(2):
            for k in range(3):
                add_term(i*m+j, j*p+(k+3), (k+3)*n+i)
    
    # Block 3: j in [2,4], k in [0,2]: A[i,j]*B[j,k] -> C[i,k]
    for i in range(n):
        for j in range(3):
            for k in range(3):
                add_term(i*m+(j+2), (j+2)*p+k, k*n+i)
    
    # Block 4: j in [2,4], k in [3,5]: A[i,j]*B[j,k] -> C[i,k]
    for i in range(n):
        for j in range(3):
            for k in range(3):
                add_term(i*m+(j+2), (j+2)*p+(k+3), (k+3)*n+i)
    
    rank_block = len(U_block)
    
    if _verify(U_block, V_block, W_block, n, m, p):
        return {"U": U_block, "V": V_block, "W": W_block}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
