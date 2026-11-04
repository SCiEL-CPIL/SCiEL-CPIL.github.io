import numpy as np

def construct():
    n, m, p = 3, 4, 6
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We iterate over each scalar multiplication term in the schoolbook algorithm
    # Term r corresponds to indices i, j, k
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: A is n x m, flattened row-major? 
                # The interface specifies U has shape (n*m, rank). 
                # Standard flattening for A_{ij} is i*m + j.
                u_idx = i * m + j
                
                # Index in V: B is m x p. Flattened row-major: j*p + k.
                v_idx = j * p + k
                
                # Index in W: C is n x p. 
                # The task specifies: "output factor follows the official trace convention W[k*n+i,r]"
                # This means the index for C_{ik} in the flattened W vector is k*n + i.
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
