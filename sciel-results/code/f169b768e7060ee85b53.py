import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    c = 0
    
    # Strassen terms
    # M1
    U[u_idx(0,0)][c] = 1; U[u_idx(1,1)][c] = 1
    V[v_idx(0,0)][c] = 1; V[v_idx(1,1)][c] = 1
    W[w_idx(0,0)][c] = 1; W[w_idx(1,1)][c] = -1
    c += 1
    
    # M2
    U[u_idx(1,0)][c] = 1; U[u_idx(1,1)][c] = 1
    V[v_idx(0,0)][c] = 1
    W[w_idx(0,1)][c] = 1
    c += 1
    
    # M3
    U[u_idx(0,0)][c] = 1
    V[v_idx(0,1)][c] = 1; V[v_idx(1,1)][c] = -1
    W[w_idx(1,0)][c] = 1
    c += 1
    
    # M4
    U[u_idx(1,1)][c] = 1
    V[v_idx(1,0)][c] = 1; V[v_idx(0,0)][c] = -1
    W[w_idx(0,1)][c] = 1; W[w_idx(0,0)][c] = -1
    c += 1
    
    # M5
    U[u_idx(0,0)][c] = 1; U[u_idx(0,1)][c] = 1
    V[v_idx(1,1)][c] = 1
    W[w_idx(1,0)][c] = 1
    c += 1
    
    # M6
    U[u_idx(1,0)][c] = 1; U[u_idx(0,0)][c] = -1
    V[v_idx(0,0)][c] = 1; V[v_idx(0,1)][c] = 1
    W[w_idx(0,0)][c] = 1; W[w_idx(1,1)][c] = -1
    c += 1
    
    # M7
    U[u_idx(0,1)][c] = 1; U[u_idx(1,1)][c] = -1
    V[v_idx(1,0)][c] = 1; V[v_idx(1,1)][c] = 1
    W[w_idx(0,0)][c] = -1
    c += 1
    
    # Group 2: j in {0,1}, k=2
    for i in range(2):
        for j in range(2):
            U[u_idx(i,j)][c] = 1
            V[v_idx(j,2)][c] = 1
            W[w_idx(2,i)][c] = 1
            c += 1
            
    # Group 3: j=2, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            U[u_idx(i,2)][c] = 1
            V[v_idx(2,k)][c] = 1
            W[w_idx(k,i)][c] = 1
            c += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
