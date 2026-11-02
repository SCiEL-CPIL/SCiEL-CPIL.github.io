import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    margin = 5.0
    
    # Target tensor T[i, j, k] = 1 if i < n, j < m, k < p and corresponds to A[i,j]B[j,k]
    # Actually, the tensor is defined by T[i, j, k] = delta_{j, j'}? No.
    # The multiplication tensor M has M[i, j, k] = 1 if the input A[i, j'] and B[j', k] are multiplied.
    # In the standard basis, M is a 3-tensor of size n x m x p.
    # M[i, j, k] is 1 if j is the shared index? No.
    # The tensor T is such that C[i, k] = sum_j A[i, j] B[j, k].
    # This is a bilinear map. The tensor T has indices i, j, k.
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] exists.
    # So T[i, j, k] = 1 for all i, j, k? No.
    # For a fixed j, A[i, j] and B[j, k] are multiplied.
    # So the tensor has non-zero entries at (i, j, k) for all i, j, k?
    # Wait. A is n x m. B is m x p.
    # The term A[i, j] * B[j, k] contributes to C[i, k].
    # So the tensor T has T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Yes, the schoolbook decomposition has rank n*m*p, and each rank-1 tensor is E_i \otimes E_j \otimes E_k.
    
    # We want to find U, V, W such that sum_c U[:, c] \otimes V[:, c] \otimes W[:, c] = T.
    # T[i, j, k] = 1 for all i, j, k.
    
    # Try to find a rank 18 solution.
    # We will use a randomized search.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, best_rank))
    V_sb = np.zeros((m * p, best_rank))
    W_sb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Search for lower rank
    # We will try ranks 18 to 23
    for rank in range(18, 24):
        if time.time() > deadline - margin:
            break
            
        # Try multiple random restarts
        for _ in range(100):
            if time.time() > deadline - margin:
                break
                
            # Initialize random sparse matrices
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Fill with random non-zeros
            for c in range(rank):
                # Choose 2-3 random indices for U
                u_idx = np.random.choice(n * m, size=2, replace=False)
                v_idx = np.random.choice(m * p, size=2, replace=False)
                w_idx = np.random.choice(p * n, size=2, replace=False)
                
                U[u_idx, c] = np.random.choice([-1, 1], size=2)
                V[v_idx, c] = np.random.choice([-1, 1], size=2)
                W[w_idx, c] = np.random.choice([-1, 1], size=2)
            
            # Check if this is a valid decomposition
            # T_reconstructed[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
            # We can compute this using tensor contractions
            # T_reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
            # This might be slow for large rank, but rank is small.
            
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            # Target T is all ones
            T_target = np.ones((n, m, p))
            
            if np.allclose(T_recon, T_target):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == rank:
            break
    
    # Convert to lists of integers/half-integers
    # The problem asks for integer/half-integer entries.
    # Our random search uses integers.
    
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
