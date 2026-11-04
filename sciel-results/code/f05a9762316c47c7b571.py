"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * p  # 90
    
    # Generate schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_decomp = (U_school, V_school, W_school)
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition using randomized search
    # We'll try ranks from schoolbook_rank-1 down to a lower bound
    start_time = time.time()
    time_limit = 60  # seconds
    
    # Lower bound: for matrix multiplication, rank is at least max(n*m, m*p, p*n)
    # but this is not tight. A better lower bound is from Strassen for 2x2x2.
    # For now, we'll try down to 50.
    lower_bound = 50
    
    for r in range(schoolbook_rank - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Check if this decomposition is exact
            # We need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute the sum for this (i,j,k)
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_decomp = (U, V, W)
                best_rank = r
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to list format as required
    def to_list(mat):
        return mat.tolist()
    
    U, V, W = best_decomp
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
