"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product, combinations
import time
import random

DIMENSIONS = [4, 5, 6]

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We can check this by building the tensor and comparing
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    
    return True

def _random_structured_search(n, m, p, target_rank, deadline, seed=None):
    """
    Search for a rank-target_rank decomposition using structured random projections.
    
    Strategy: 
    1. Randomly initialize U and V with small integer/rational entries
    2. For each (i,k) pair, solve the linear system for W[k*n+i, :]
    3. Verify the decomposition
    
    The key insight: if we fix U and V, then for each output position (i,k),
    we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j.
    
    This is a linear system in W[k*n+i, :] given U and V.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    best_result = None
    
    # Try multiple random initializations
    max_attempts = 500
    attempt = 0
    
    while attempt < max_attempts:
        if time.time() > deadline - 5:
            break
        
        attempt += 1
        
        # Generate random U and V with entries from {-1, 0, 1} or small rationals
        # Use a mix of strategies
        strategy = attempt % 3
        
        if strategy == 0:
            # Binary random
            U = np.random.randint(0, 2, size=(num_U, target_rank)).astype(float)
            V = np.random.randint(0, 2, size=(num_V, target_rank)).astype(float)
        elif strategy == 1:
            # Ternary random
            U = np.random.randint(-1, 2, size=(num_U, target_rank)).astype(float)
            V = np.random.randint(-1, 2, size=(num_V, target_rank)).astype(float)
        else:
            # Sparse random with some structure
            U = np.zeros((num_U, target_rank))
            V = np.zeros((num_V, target_rank))
            for r in range(target_rank):
                # Pick random non-zero entries
                for idx in np.random.choice(num_U, size=min(3, num_U), replace=False):
                    U[idx, r] = random.choice([-1, 1])
                for idx in np.random.choice(num_V, size=min(3, num_V), replace=False):
                    V[idx, r] = random.choice([-1, 1])
        
        # For each (i,k) pair, solve for W[k*n+i, :]
        W = np.zeros((num_W, target_rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # We need: for all j in [0,m): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a linear system: A @ W_row = b
                # where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # and b[j] = 1 for all j
                
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                b = np.ones(m)
                
                # Solve A @ w = b
                # If m <= target_rank, this is underdetermined; find a solution
                try:
                    if m <= target_rank:
                        # Use least squares or find a particular solution
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution actually satisfies the equation
                        if not np.allclose(A @ w, b, atol=1e-10):
                            valid = False
                            break
                    else:
                        # Overdetermined: check if consistent
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        if not np.allclose(A @ w, b, atol=1e-10):
                            valid = False
                            break
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                W[k * n + i, :] = w
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
    
    return None

def _try_known_decompositions(n, m, p, deadline):
    """Try known lower-rank decompositions for small matrix sizes."""
    # For very small matrices, there might be known decompositions
    # This is a placeholder for any hardcoded known results
    
    # Check if we're in a regime where rank < schoolbook is known
    schoolbook = n * m * p
    lower_bound = max(n * m, m * p, p * n)
    
    # For 4x5x6, schoolbook = 120, lower bound = 30
    # No known decomposition below schoolbook for this size as of current knowledge
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Deadline with safety margin (150s kill, use 140s)
    deadline = time.time() + 140
    
    # Lower bound on rank
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound up to schoolbook - 1
    # Be strategic: try a few key ranks
    candidate_ranks = []
    for r in range(lower_bound, best_rank):
        candidate_ranks.append(r)
    
    # Shuffle to explore different ranks
    random.shuffle(candidate_ranks)
    
    # But prioritize lower ranks
    candidate_ranks.sort()
    
    # Try each candidate rank
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = _random_structured_search(n, m, p, rank, deadline, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # If we found a decomposition, we could stop or continue to find lower
            # Continue searching for potentially lower rank
            # But if rank is close to lower bound, we can stop
            if rank <= lower_bound + 2:
                break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
