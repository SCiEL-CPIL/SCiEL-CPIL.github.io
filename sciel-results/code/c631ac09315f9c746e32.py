"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 30
    
    # Build the multiplication tensor T[i,j,k] = 1 for valid (i,j,k)
    # T has shape (n, m, p) and T[i,j,k] = 1 iff we need the product A[i,j]*B[j,k]
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Symmetry-aware search: try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But we can do better with structure
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try to find a decomposition with rank < schoolbook
    # Use random restarts with symmetry exploitation
    
    def try_random_decomposition(rank, max_iters=5000):
        """Try to find a rank-r decomposition via random search."""
        # Initialize with random values in {-1, 0, 1}
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with structure
            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))
            W = np.random.choice([-1, 0, 1], size=(pn, rank))
            
            # Check if this decomposes the tensor
            # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.dot(U[idx_u, :], V[idx_v, :])
                        val = val * W[idx_w, :]  # This is wrong, need element-wise
                        # Actually: sum_r U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                        val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                        if val != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                return U, V, W
        return None
    
    # Try various ranks
    for rank in range(15, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
        
        # Try random search
        result = try_random_decomposition(rank, max_iters=1000)
        if result is not None:
            U, V, W = result
            best_result = (U.tolist(), V.tolist(), W.tolist())
            best_rank = rank
            break
    
    # If no better found, use schoolbook
    if best_result is None:
        U = np.zeros((nm, rank_schoolbook), dtype=int)
        V = np.zeros((mp, rank_schoolbook), dtype=int)
        W = np.zeros((pn, rank_schoolbook), dtype=int)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    U[idx_u, r] = 1
                    V[idx_v, r] = 1
                    W[idx_w, r] = 1
                    r += 1
        
        best_result = (U.tolist(), V.tolist(), W.tolist())
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

# EVOLVE-BLOCK-END
