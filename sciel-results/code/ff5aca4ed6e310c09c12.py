"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # For each (i,j,k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Actually the tensor decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor.
    # Wait - the matmul tensor T[i,j,k] has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k}... no.
    # Actually for matrix multiplication A(3x3) * B(3x4) -> C(3x4):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait - the tensor for matrix multiplication has T[i,j,k] = 1 for all valid combinations.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] appears exactly once).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = Fraction(1)
                V[j*p+k][col] = Fraction(1)
                W[k*n+i][col] = Fraction(1)
                col += 1
    return U, V, W

def _try_rank_reduction(n, m, p, max_rank, time_budget):
    """Try to find a decomposition with rank < max_rank using random structured search."""
    deadline = time.time() + time_budget
    best_rank = max_rank
    best_decomp = None
    
    # Start from schoolbook and try to eliminate terms
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    
    # Try random linear combinations to reduce rank
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        if rank <= best_rank:
            break
        
        # Try to find a linear dependency among the rank-1 terms
        # Pick two random terms and try to combine them
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        # Try to see if we can merge terms c1 and c2
        # This is a heuristic - just try random perturbations
        # Actually, let's try a different approach: random search in low-rank space
        
    # Structured approach: try known patterns for small matrix multiplication
    # For 3x3x4, try rank 24, 25, 26, ...
    
    for target_rank in range(24, min(36, max_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try random structured decompositions
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse U, V, W with half-integer entries
            U_try = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V_try = [[Fraction(0)] * target_rank for _ in range(m * p)]
            W_try = [[Fraction(0)] * target_rank for _ in range(p * n)]
            
            # Fill with random small integers/half-integers
            for i in range(n * m):
                for c in range(target_rank):
                    if random.random() < 0.5:
                        U_try[i][c] = Fraction(random.randint(-2, 2), 1)
            for j in range(m * p):
                for c in range(target_rank):
                    if random.random() < 0.5:
                        V_try[j][c] = Fraction(random.randint(-2, 2), 1)
            for k in range(p * n):
                for c in range(target_rank):
                    if random.random() < 0.5:
                        W_try[k][c] = Fraction(random.randint(-2, 2), 1)
            
            if _check_decomposition(U_try, V_try, W_try, n, m, p):
                return U_try, V_try, W_try, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find better decomposition
    result = _try_rank_reduction(n, m, p, schoolbook_rank, deadline - start_time - 5)
    
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
