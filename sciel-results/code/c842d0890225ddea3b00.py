import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook algorithm for n x m * m x p
    # Rank = n * m * p = 3 * 3 * 5 = 45
    
    R = n * m * p
    
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Index mapping:
    # A is n x m. Index u = i * m + k. (i: 0..n-1, k: 0..m-1)
    # B is m x p. Index v = k * p + j. (k: 0..m-1, j: 0..p-1)
    # C is n x p. Index w = i * p + j. (i: 0..n-1, j: 0..p-1)
    # BUT the prompt says W[k*n+i, r].
    # n=3.
    # If C is n x p, and index is k*n + i, then k is col, i is row.
    # So w_idx = j * n + i.
    
    # Let's verify the prompt's W convention again.
    # "W[k*n+i,r]"
    # If n=3, index = 3k + i.
    # If C is 3x5, i (row) 0-2, k (col) 0-4.
    # Max index = 3*4 + 2 = 14. Correct.
    
    # So for C_{i, j}, the index in W is j * n + i.
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index for A[i, k]
                u_idx = i * m + k
                # V index for B[k, j]
                v_idx = k * p + j
                # W index for C[i, j]
                w_idx = j * n + i
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
