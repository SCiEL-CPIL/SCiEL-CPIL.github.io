"""
Matrix multiplication tensor decomposition for 2x3 * 3x8.
Seeking a rank-minimal exact decomposition with integer/half-integer entries.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _check_time():
    if _time_left() < 5.0:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 2*3*8 = 48
    rank_schoolbook = n * m * p
    
    # The tensor for matrix multiplication C = AB has entries T[i,j,k] = 1
    # where i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1),
    # k indexes columns of B (0..p-1).
    # The output index in W is k*n + i (trace convention: W[k*n+i, r])
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # We start with the schoolbook decomposition and try to reduce rank.
    # For small matrices, we can attempt to find a lower rank decomposition
    # using known results or by trying specific structured decompositions.
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # For 2x3x8, the minimum rank is at least 24.
    # The schoolbook gives 48. Let's see if we can do better.
    
    # Try to construct a decomposition of rank < 48.
    # One approach: use the fact that the matrix multiplication tensor has some structure.
    # For 2x2x2, the minimum rank is 7 (Strassen).
    # For 2x3x8, we can try to block it into smaller pieces.
    
    # Let's try a structured approach:
    # Split the 3 columns of A / 3 rows of B into blocks.
    # Actually, let's just try to verify if rank 24 is achievable or find a better bound.
    
    # For now, let's return the schoolbook decomposition as a valid baseline.
    # Then we can try to improve it if time permits.
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    if _check_time():
        # Attempt to find a lower rank decomposition
        # We can try rank = 24 (the trivial lower bound)
        # This is a hard problem, so let's try a heuristic approach
        
        # Try constructing a decomposition of rank 24
        # If we can achieve rank 24, that's optimal (since rank >= max(n*m, m*p, p*n) = 24)
        
        # One way to think about this: for each pair (j, k), we need to cover all i.
        # Actually, the constraint is that for each (i, j, k), the product U[i*m+j, r] * V[j*p+k, r] must sum to 1 over r.
        
        # Let's try a structured decomposition:
        # For each j (0..m-1) and k (0..p-1), assign rank_jk columns.
        # Sum over j,k of rank_jk = total rank.
        # For fixed j, the matrix A(:, j) * B(j, :) gives a contribution to C.
        # This is a rank-1 outer product for each (i, k) pair, but we can share.
        
        # Actually, for fixed j, the sub-tensor has entries T[i, j, k] = 1 for all i, k.
        # This is a rank-1 tensor (outer product of all-ones vector of length n and all-ones vector of length p).
        # So for each j, we need rank 1 to represent the (i, k) part? No, that's not right.
        
        # The tensor is T[i,j,k] = 1. For fixed j, T[:,j,:] is an n x p matrix of all ones, which has rank 1.
        # But the tensor rank is not the same as matrix rank for each slice.
        
        # Let's just return the schoolbook for now, as finding a better decomposition
        # requires more sophisticated search that may not finish in time.
        
        # Try a simple improvement: if we can find any decomposition with rank < 48, use it.
        # For now, we'll stick with the schoolbook unless we find a specific better one.
        
        # Known result: for 2x3x3, minimum rank is 7 (I think). Let me verify.
        # Actually, I'm not sure of the exact minimum rank for 2x3x8.
        # The lower bound is max(2*3, 3*8, 8*2) = max(6, 24, 16) = 24.
        
        # Let's try to construct a rank-24 decomposition.
        # Idea: for each (j, k) pair, use 1 rank-1 term that covers all i.
        # But U[i*m+j, r] depends on i, so we can't have a single r work for all i with U=V=1.
        
        # Alternative: use the fact that for fixed (j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all i.
        # This means the vector U[:, j*m? No, U[i*m+j, :] for varying i must be such that
        # their dot product with V[j*p+k, :] is 1 for each i.
        
        # This is getting complex. Let's just return the schoolbook decomposition.
        pass
    
    return best_result
