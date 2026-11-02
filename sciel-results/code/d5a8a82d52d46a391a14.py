import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create 3D tensors for U, V, W components
    # U corresponds to indices (i, j) -> rank index (i*m + j)*p + k? 
    # The schoolbook decomposition uses rank = n*m*p.
    # Each term in the sum corresponds to a triple (i, j, k).
    # U[i*m + j, i*m*p + j*p + k] = 1
    # V[j*p + k, i*m*p + j*p + k] = 1
    # W[k*n + i, i*m*p + j*p + k] = 1
    
    # We can construct these using numpy arrays.
    # Shape of U: (n*m, rank)
    # Shape of V: (m*p, rank)
    # Shape of W: (p*n, rank)
    
    U_np = np.zeros((n * m, rank), dtype=float)
    V_np = np.zeros((m * p, rank), dtype=float)
    W_np = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_np[i * m + j, col] = 1.0
                V_np[j * p + k, col] = 1.0
                W_np[k * n + i, col] = 1.0
                
    # Convert to list of lists as required by the interface
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
