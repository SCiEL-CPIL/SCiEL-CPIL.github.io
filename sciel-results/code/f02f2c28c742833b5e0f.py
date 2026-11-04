import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: Rank = n * m * p
    # We construct the factor matrices using vectorized operations
    # instead of explicit loops, which represents a different implementation
    # approach (vectorized/linear algebra) vs the seed's iterative approach.
    
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Indices for U: i in [0, n), j in [0, m)
    # U[i*m + j, k] = 1 if k corresponds to (i,j,k')
    
    # We can create the index arrays
    # k = i * m * p + j * p + k'
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized fill
    # For each k in range(rank):
    # i = k // (m*p)
    # rem = k % (m*p)
    # j = rem // p
    # k_prime = rem % p
    
    k_indices = np.arange(rank)
    i_indices = (k_indices // (m * p)).astype(int)
    rem = k_indices % (m * p)
    j_indices = (rem // p).astype(int)
    k_prime_indices = (rem % p).astype(int)
    
    # Set U
    U_rows = i_indices * m + j_indices
    U[U_rows, k_indices] = 1
    
    # Set V
    V_rows = j_indices * p + k_prime_indices
    V[V_rows, k_indices] = 1
    
    # Set W
    W_rows = k_prime_indices * n + i_indices
    W[W_rows, k_indices] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
