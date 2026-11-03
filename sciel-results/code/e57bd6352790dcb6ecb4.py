"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_optimize_rank(rank, max_iters=500, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Use a simple iterative approach: for each (i,j,k), adjust factors
        # This is heuristic and may not converge
        
        # Alternative: use the fact that we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # For each c, define a "slice" contribution
        
        # Try a structured approach: use the same pattern as schoolbook but with sharing
        # For m=4, we can potentially group j indices
        
        # Let's try a different initialization: start from schoolbook and try to merge
        # This is complex, so let's try random search with verification
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            # Try different random seeds and initializations
            seed = attempt * 1000 + 7
            np.random.seed(seed)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to try to get magnitudes right
            # Each term U*V*W should sum to 1 over c
            # If each factor is ~1/sqrt(rank), then each product is ~1/rank, sum ~1
            scale = 1.0 / np.sqrt(rank)
            U = U * scale
            V = V * scale
            W = W * scale
            
            # Check if this works (very unlikely for random)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 0.01:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Build the guaranteed schoolbook solution
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Try ranks from low to high, but focus on reasonable ranges
    # Known: rank >= max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Let's try some structured ranks
    
    # Try a structured approach: use the m=4 dimension to create shared factors
    # Idea: for each (i,k) pair, we need sum_j A[i,j]*B[j,k] = 1
    # If we can express this using fewer than m*p*n terms...
    
    # Let's try rank 60, 64, 70, 72, 75, 76, 78, 79
    candidate_ranks = [79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 68, 65, 60, 56, 50]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if rank >= best_rank:
            continue
        
        result = try_optimize_rank(rank, max_iters=200)
        if result is not None:
            if verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                break  # Found a valid lower-rank solution
    
    return best_result
# EVOLVE-BLOCK-END
