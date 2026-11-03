"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import deque

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each output (i, k), the sum over j of A[i,j]*B[j,k] should be recoverable
    # The tensor T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    # rank_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank = n*m*p = 54)
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    current_rank = rank0
    best_U, best_V, best_W = U, V, W
    
    # Strategy: try to reduce rank by combining terms
    # Idea: if we can find linear dependencies among the rank-1 tensors, we can reduce rank
    # Specifically, we try random combinations and see if we can merge terms
    
    # A simpler approach: try rank-deficient constructions
    # For 2x3 x 3x9, known lower bounds suggest rank >= 24 (from de Groote bounds)
    # But let's try to find any improvement over 54
    
    # Approach: randomized rank reduction
    # Try to represent the same tensor with fewer rank-1 terms
    
    # Use the identity: if we can find coefficients such that
    # sum_r u_r^T (v_r * w_r) = sum_r' u'_r'^T (v'_r' * w'_r')
    # with fewer terms
    
    # Practical approach: try small perturbations and Gaussian elimination
    # on the coefficient space
    
    # Alternative: exploit structure
    # A (2x3) times B (3x9): C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has n*m*p = 54 non-zero entries (all 1s in the appropriate positions)
    
    # Try a block decomposition approach
    # Group the 3 middle dimension into sub-groups
    
    # For now, try a random search for lower rank decompositions
    random.seed(42)
    
    def try_rank_reduction(target_rank, max_iters=2000):
        """Try to find a decomposition with the given target rank."""
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_r = random.randint(-3, 3) if random.random() < 0.5 else 1
            V_r = random.randint(-3, 3) if random.random() < 0.5 else 1
            W_r = random.randint(-3, 3) if random.random() < 0.5 else 1
            
            U_try = [[random.choice([0, 1, -1, 2, -2]) for _ in range(target_rank)] 
                     for _ in range(n * m)]
            V_try = [[random.choice([0, 1, -1, 2, -2]) for _ in range(target_rank)] 
                     for _ in range(m * p)]
            W_try = [[random.choice([0, 1, -1, 2, -2]) for _ in range(target_rank)] 
                     for _ in range(p * n)]
            
            if _check_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                return U_try, V_try, W_try
        return None
    
    # Try progressively lower ranks
    for target_rank in range(rank0 - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank_reduction(target_rank, max_iters=500)
        if result is not None:
            U_try, V_try, W_try = result
            if _check_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                current_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Also try a structured approach: exploit the 2x3 structure
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..8}, j in {0,1,2}
    # We can write: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #               C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Try a rank construction that shares more structure
    # For each j, we have 2*9 = 18 outputs that depend on B[j,:]
    # This is like 3 separate 2x1 x 1x9 multiplications
    
    # A better structured approach:
    # Use the fact that we only need to track which (i,j,k) triples are active
    
    # Let's try a more systematic search with small random perturbations
    if time.time() < deadline - 20:
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            # Start from schoolbook and try to merge random pairs
            U_merge = [row[:] for row in best_U]
            V_merge = [row[:] for row in best_V]
            W_merge = [row[:] for row in best_W]
            
            # Try to set one column to a sum of two others (if it results in zero contribution)
            # This is complex; instead try random linear combinations
            
            r1 = random.randint(0, len(best_U[0]) - 1)
            r2 = random.randint(0, len(best_U[0]) - 1)
            if r1 == r2:
                continue
            alpha = random.choice([1, -1, 2, -2])
            
            # Create new column as sum, then check if we can eliminate one
            for idx in range(len(U_merge)):
                U_merge[idx][r2] = U_merge[idx][r2] + alpha * U_merge[idx][r1]
                V_merge[idx][r2] = V_merge[idx][r2] + alpha * V_merge[idx][r1]
                W_merge[idx][r2] = W_merge[idx][r2] + alpha * W_merge[idx][r1]
            
            # Now try to zero out column r1 by subtracting column r2
            for idx in range(len(U_merge)):
                U_merge[idx][r1] = U_merge[idx][r1] - U_merge[idx][r2]
                V_merge[idx][r1] = V_merge[idx][r1] - V_merge[idx][r2]
                W_merge[idx][r1] = W_merge[idx][r1] - W_merge[idx][r2]
            
            # Check if r1 is now zero
            is_zero = all(U_merge[idx][r1] == 0 for idx in range(n*m)) and \
                      all(V_merge[idx][r1] == 0 for idx in range(m*p)) and \
                      all(W_merge[idx][r1] == 0 for idx in range(p*n))
            
            if is_zero:
                # Remove column r1
                U_new = [row[:r1] + row[r1+1:] for row in U_merge]
                V_new = [row[:r1] + row[r1+1:] for row in V_merge]
                W_new = [row[:r1] + row[r1+1:] for row in W_merge]
                
                new_rank = len(U_new[0])
                if new_rank < current_rank:
                    if _check_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                        current_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
    
    if time.time() > deadline - 2:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
