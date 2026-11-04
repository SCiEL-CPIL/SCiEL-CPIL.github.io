"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T of shape (n*p, n*m, m*p)
    # T[c, a, b] = 1 if c = i*p+k, a = i*m+j, b = j*p+k
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * p + k
                a = i * m + j
                b = j * p + k
                T[c, a, b] = 1.0
    
    # Reconstruct from decomposition: T[c,a,b] = sum_r U[a,r] * V[b,r] * W[c,r]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Use einsum: T_reconstructed[c,a,b] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is U[a,r] * V[b,r] * W[c,r] summed over r
    T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = np.max(np.abs(T_rec - T))
    return max_err < tol, max_err

def _random_decomposition(rank, n, m, p, seed=None):
    """Generate a random initial decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # The minimum rank is at least max(n*m, m*p, n*p) = max(8, 32, 16) = 32
    # But actually, lower bounds are more complex. Let's try ranks from 32 upward.
    
    for rank in range(32, min(rank_schoolbook, 50)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U_r, V_r, W_r = _random_decomposition(rank, n, m, p, seed=attempt * 1000 + rank)
            
            # Simple local search: try to minimize the error
            # Use a simple iterative approach
            U_arr = np.array(U_r, dtype=float)
            V_arr = np.array(V_r, dtype=float)
            W_arr = np.array(W_r, dtype=float)
            
            # Build target tensor
            T = np.zeros((n * p, n * m, m * p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * p + k, i * m + j, j * p + k] = 1.0
            
            # Iteratively update factors to reduce error
            lr = 0.01
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Compute current reconstruction
                T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
                error = T_rec - T
                
                # Update U: gradient wrt U[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', error, V_arr, W_arr)
                U_arr -= lr * grad_U
                
                # Update V
                grad_V = np.einsum('abc,ar,cr->br', error, U_arr, W_arr)
                V_arr -= lr * grad_V
                
                # Update W
                grad_W = np.einsum('abc,ar,br->cr', error, U_arr, V_arr)
                W_arr -= lr * grad_W
                
                # Check convergence
                if it % 10 == 9:
                    T_rec_check = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
                    max_err = np.max(np.abs(T_rec_check - T))
                    if max_err < 1e-6:
                        # Found a valid decomposition
                        best_rank = rank
                        best_U = U_arr.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                        break
            
            if best_rank == rank:
                break
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
