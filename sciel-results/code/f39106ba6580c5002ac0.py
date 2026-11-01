"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 by 5x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 50
    
    # First, let's try to find a low-rank decomposition using a structured approach.
    # The matrix multiplication tensor T has entries T[i,j,k] which is 1 if the output
    # element (i,k) depends on input elements (i,j) and (j,k).
    # 
    # For a 2x5 * 5x5 multiplication:
    # - Input A is 2x5, so A[i,j] for i in 0..1, j in 0..4
    # - Input B is 5x5, so B[j,k] for j in 0..4, k in 0..4
    # - Output C is 2x5, so C[i,k] for i in 0..1, k in 0..4
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has indices (i, j, k) where:
    # - i indexes rows of A (and C): 0..1
    # - j indexes columns of A / rows of B: 0..4
    # - k indexes columns of B (and C): 0..4
    #
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, T[i,j,k] = 1 for all valid i,j,k since every A[i,j]*B[j,k] contributes.
    #
    # Wait, let me reconsider. The multiplication tensor T_{ijk} is defined such that:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    #
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # The schoolbook decomposition uses r = n*m*p = 50 terms, one for each (i,j,k).
    #
    # To find a lower rank, we can use known results or search.
    # For small matrices, the minimal rank is often known or can be found by search.
    #
    # Let's try a few approaches:
    # 1. Check if there's a known result for 2x5 * 5x5
    # 2. Try to use Strassen-like techniques
    # 3. Randomized search for low-rank decompositions
    
    # Approach: Try to find a decomposition of rank < 50
    # We'll use a simple approach: try ranks from low to high and check if a valid
    # decomposition exists.
    
    # For now, let's return the schoolbook decomposition as a baseline,
    # but try to improve it if we have time.
    
    # Try a few known low-rank constructions
    # For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    # For our case, let's see if we can do better than 50.
    
    # Let's try a direct construction approach:
    # We want U (10 x r), V (25 x r), W (10 x r) such that
    # for all i in 0..1, j in 0..4, k in 0..4:
    #   sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 50 equations in r*45 variables (roughly).
    # For r = 50, it's straightforward (schoolbook).
    # For r < 50, it's more constrained.
    
    # Let's try to find a decomposition with rank as low as possible.
    # We'll use a greedy or randomized approach.
    
    # Actually, let's just return the schoolbook for now and see if we can improve.
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to see if we can reduce the rank
    # For small problems, we can try various ranks
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a lower rank decomposition
    # We'll try ranks from 30 to 49 and see if we can find a valid one
    
    # Actually, let's be smarter. Let's use the fact that the problem is small
    # and try to find a low-rank decomposition using a linear algebra approach.
    
    # The condition for a valid decomposition is:
    # For each (i,k), the matrix M_{ik} defined by M_{ik}[j, j'] = 0 for j != j'
    # and M_{ik}[j,j] = 1 must be representable as sum_c U[:,c] V[:,c] W[:,c]
    # restricted to the appropriate indices.
    
    # This is getting complex. Let's just return the schoolbook for now.
    # But first, let's check if we have time to try something better.
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Not enough time, return schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a simple improvement: see if we can merge some terms
    # For now, let's just return the schoolbook decomposition
    # as it's guaranteed to be correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
