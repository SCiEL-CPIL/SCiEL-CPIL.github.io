"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find lower rank via structured search
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # Try rank = 47, 46, ... down to some limit
    # Use a greedy approach: start with schoolbook, try to eliminate one term
    # by adjusting others
    
    # For each term to eliminate, try to express it as a combination of others
    # This is hard in general, so use a heuristic
    
    # Alternative: use the fact that for fixed i, the 4x4 sub-tensor is all-ones
    # which has rank 1. So we might be able to share terms across different i values.
    
    # Let's try a decomposition where for each j,k, we have a term that works for all i
    # Term for (j,k): U[i,j,r] = 1 for all i, V[j,k,r] = 1, W[k,i,r] = 1 for all i
    # Then sum over r: for each (j,k), we get sum_i 1*1*1 = 3, not 1.
    # So we need to scale: U[i,j,r] = 1/3, then sum = 1. But we need integer/half-integer.
    
    # Try: for each (j,k), use 3 terms, one for each i, but with shared structure
    # This doesn't help.
    
    # Let's try: rank = n * m * p - something
    # The key insight: if we can find linear dependencies in the decomposition matrix
    
    # Convert to numpy for efficient computation
    U_np = np.array(best_U, dtype=float)
    V_np = np.array(best_V, dtype=float)
    W_np = np.array(best_W, dtype=float)
    
    # Build the tensor verification matrix
    # T_target[i,j,k] = 1 for all i,j,k
    # T_computed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # For each column r, check if it can be expressed as a linear combination of others
    # This is equivalent to checking if the decomposition matrix has rank < 48
    
    # The decomposition matrix D has entries D[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, D is identity, so rank = 48
    
    # To get lower rank, we need a different decomposition where D has lower rank
    
    # Let's try a parametric search: allow U, V, W to have small integer/half-integer values
    # and search for rank < 48
    
    # Given time constraints, try a limited search
    best_found = None
    
    # Try ranks from 47 down to 40
    for target_rank in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random integer/half-integer initialization
            U_try = np.zeros((n*m, target_rank))
            V_try = np.zeros((m*p, target_rank))
            W_try = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Assign random values from {-1, -0.5, 0, 0.5, 1}
                for idx in range(n*m):
                    U_try[idx, r] = random.choice([-1, -0.5, 0, 0.5, 1])
                for idx in range(m*p):
                    V_try[idx, r] = random.choice([-1, -0.5, 0, 0.5, 1])
                for idx in range(p*n):
                    W_try[idx, r] = random.choice([-1, -0.5, 0, 0.5, 1])
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r] for r in range(target_rank))
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_found = (target_rank, U_try, V_try, W_try)
                break
        if best_found:
            break
    
    if best_found:
        rank, U_f, V_f, W_f = best_found
        # Convert to list of lists with integer/half-integer values
        U_list = [[int(U_f[i,j]) if U_f[i,j] == int(U_f[i,j]) else float(U_f[i,j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[int(V_f[i,j]) if V_f[i,j] == int(V_f[i,j]) else float(V_f[i,j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[int(W_f[i,j]) if W_f[i,j] == int(W_f[i,j]) else float(W_f[i,j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook as fallback
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
