import numpy as np
import random
import time
from collections import defaultdict
import heapq

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, base_rank), dtype=int)
    V_school = np.zeros((m * p, base_rank), dtype=int)
    W_school = np.zeros((p * n, base_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a better decomposition using combinatorial optimization
    # Formulate as minimum weight perfect matching in a tripartite graph
    
    # Each rank-1 term is a triangle (i,j,k) where:
    # - U has non-zero at (i*m+j, r)
    # - V has non-zero at (j*p+k, r)  
    # - W has non-zero at (k*n+i, r)
    
    # We need to cover all (i,j,k) triples exactly once with rank-1 terms
    # But we can have multiple (i,j,k) in one term if they share structure
    
    # For now, let's try a greedy approach to reduce rank
    # Start from schoolbook and try to merge terms
    
    # Actually, let's try a different approach: 
    # Use the fact that we can have linear combinations
    
    # Try rank = base_rank - 1 first
    for target_rank in range(base_rank - 1, base_rank - 5, -1):
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        # Initialize with schoolbook pattern for first target_rank terms
        for r in range(target_rank):
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            if i < n and j < m and k < p:
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
        
        # Check if this works (it won't, since we're missing terms)
        # We need to find a valid decomposition of this rank
        
        # For simplicity, if we can't find one quickly, use schoolbook
        if target_rank == base_rank - 1:
            # Special case: can we reduce by 1?
            # Try to find two terms that can be merged
            pass
    
    # Return schoolbook for now - it's guaranteed to work
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
