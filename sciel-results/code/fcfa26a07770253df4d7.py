"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition is exact using vectorized numpy."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use the structure of 3x3x6 as two 3x3x3 blocks
    # The output C[i,k] = sum_j A[i,j]*B[j,k]
    # For k in {0,1,2}: uses B[:,0:3]
    # For k in {3,4,5}: uses B[:,3:6]
    
    # We can try to find a decomposition that exploits this block structure
    # by using a rank-r decomposition for 3x3x3 and combining two copies
    
    # First, let's try to find a good 3x3x3 decomposition
    # Then combine two such decompositions (one for each block of B)
    
    # For now, let's do a more systematic search with the block structure in mind
    
    # Try ranks from 50 down to 30
    for rank in range(50, 25, -1):
        if time.time() > deadline - 15:
            break
            
        # Try several random initializations
        max_attempts = 5 if rank > 40 else 15
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(attempt * 7919 + rank * 31)
            
            # Use sparse random matrices with values from {-1, 0, 1}
            density = 0.35 if rank > 40 else 0.45
            U = rng.choice([-1, 0, 1], size=(n*m, rank))
            V = rng.choice([-1, 0, 1], size=(m*p, rank))
            W = rng.choice([-1, 0, 1], size=(p*n, rank))
            
            # Apply sparsity masks
            mask_u = rng.random((n*m, rank)) < density
            mask_v = rng.random((m*p, rank)) < density
            mask_w = rng.random((p*n, rank)) < density
            U = U * mask_u
            V = V * mask_v
            W = W * mask_w
            
            # Verify
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        # If we found a valid decomposition at this rank, try lower
        if best_rank == rank:
            continue
    
    # Strategy: Try half-integer values for more flexibility
    if time.time() < deadline - 30:
        for rank in [42, 40, 38, 36, 34, 32]:
            if time.time() > deadline - 10:
                break
                
            for attempt in range(8):
                if time.time() > deadline - 5:
                    break
                    
                rng = np.random.RandomState(attempt * 104729 + rank * 13)
                
                # Use values from {-1, -0.5, 0, 0.5, 1}
                vals = [-1, -0.5, 0, 0, 0, 0.5, 1]
                U = rng.choice(vals, size=(n*m, rank))
                V = rng.choice(vals, size=(m*p, rank))
                W = rng.choice(vals, size=(p*n, rank))
                
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                    break
    
    # Strategy: Local search - perturb and try to reduce rank
    # Start from a valid decomposition and try to merge terms
    
    # Strategy: Try a structured approach based on the block structure
    # C[i,k] for k=0,1,2 depends on B[j,0:3]
    # C[i,k] for k=3,4,5 depends on B[j,3:6]
    
    # We can write:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #        = sum_j A[i,j]*B[j,k mod 3 + 3*floor(k/3)]
    
    # If we define B1 = B[:,0:3] and B2 = B[:,3:6], then:
    # C[:,0:3] = A * B1
    # C[:,3:6] = A * B2
    
    # So we need a decomposition that handles both A*B1 and A*B2
    # This is equivalent to multiplying A by [B1 | B2] = B
    
    # One approach: use a rank-r decomposition for 3x3x3, giving:
    # (A*B1)[i,k] = sum_{c=1}^r U[i,k,c] * V[k,c] * W[i,c]  (for k in 0,1,2)
    # (A*B2)[i,k] = sum_{c=1}^r U[i,k,c] * V'[k,c] * W[i,c]  (for k in 3,4,5)
    
    # But this doesn't quite work because the indices don't align nicely.
    
    # Let's try a different approach: use the fact that we can share some terms
    
    # For now, return the best we've found
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
