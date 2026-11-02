"""
Rank-minimal exact tensor decomposition for 3x4 @ 4x6 matrix multiplication.
We use a randomized search for a rank < 72 decomposition over {0, ±1/2, ±1, ±3/2, ±2}.
The tensor entries are 1 at T[i,j,k] and 0 otherwise, with T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
We search for a low-rank factorization.
"""

import numpy as np
import itertools
import random
import time

DIMENSIONS = [3, 4, 6]
VALUES = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _check_solution(U, V, W, n, m, p):
    num_U = n * m
    num_V = m * p
    num_W = p * n
    rank = len(U[0])
    
    # Pre-compute indices for faster access
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can vectorize this check
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_arr = np.array(U).reshape(n, m, rank)
    V_arr = np.array(V).reshape(m, p, rank)
    W_arr = np.array(W).reshape(p, n, rank)
    
    # We need to compute S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a 3-way tensor contraction.
    # We can iterate over r and accumulate.
    T = np.zeros((n, m, p))
    for r in range(rank):
        # u_r is (n, m)
        # v_r is (m, p)
        # w_r is (p, n)
        u_r = U_arr[:, :, r]
        v_r = V_arr[:, :, r]
        w_r = W_arr[:, :, r]
        
        # For each i, j, k:
        # T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # We can do this by:
        # temp[i,j,k] = u_r[i,j] * v_r[j,k]
        # Then T[i,j,k] += temp[i,j,k] * w_r[k,i]
        
        # Vectorized way:
        # temp[i,j,k] = np.einsum('ij,jk->ijk', u_r, v_r)
        # T += np.einsum('ijk,ki->ijk', temp, w_r)
        
        # Since dimensions are small, we can just use loops or einsum
        # Let's use einsum for speed
        # u_r: (n, m)
        # v_r: (m, p)
        # w_r: (p, n)
        
        # We want to add to T[i,j,k] the product u_r[i,j] * v_r[j,k] * w_r[k,i]
        # This is equivalent to:
        # T += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        
        T += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        
    # Check if T is all 1s
    if not np.allclose(T, 1.0):
        return False
    return True

def _random_matrix(rows, cols, values):
    return [[random.choice(values) for _ in range(cols)] for _ in range(rows)]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    best_rank = n * m * p
    best_solution = _construct_schoolbook()
    
    # Try to find a lower rank
    # We search R from 72 down to 60 (or lower if time permits)
    # For each R, we try random restarts
    
    # To make search effective, we can restrict the value set.
    # Common values in tensor decompositions are 0, 1, -1, 0.5, -0.5.
    # Let's use a smaller set for speed: {0, 1, -1, 0.5, -0.5}
    search_values = [0, 1, -1, 0.5, -0.5]
    
    # We will try to find a solution with rank R < 72.
    # Since the space is large, we rely on the fact that if a solution exists,
    # random sampling might find it if we are lucky, or we use a smarter heuristic.
    # A smarter heuristic: Fix U to be sparse (e.g., each row has 1 non-zero) and solve for V, W?
    # But V and W are coupled.
    
    # Let's try a simple random search first.
    # If it fails, we return schoolbook.
    
    # Optimization: If we find a solution with rank R, we can stop.
    # We iterate R from 71 down to 50.
    
    for R in range(71, 49, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for this rank
        num_restarts = 10000
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            U = _random_matrix(num_U, R, search_values)
            V = _random_matrix(num_V, R, search_values)
            W = _random_matrix(num_W, R, search_values)
            
            if _check_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    return best_solution

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
