"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W give a valid tensor decomposition of matrix multiplication."""
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, r) or Va.shape != (m*p, r) or Wa.shape != (p*n, r):
        return False
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = Ua[:, c].reshape(n, m)
        vc = Va[:, c].reshape(m, p)
        wc = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_strassen_2x2x2():
    """
    Try to find a rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    if _verify(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    return None

def _build_block_decomposition(n, m, p, block_sizes):
    """
    Build a block decomposition for n x m x p tensor.
    block_sizes: list of (n_i, m_i, p_i) tuples that partition the dimensions.
    For each block, use a known decomposition.
    """
    # This is a placeholder - we'll use a different approach
    pass

def _random_search_decomposition(n, m, p, target_rank, num_trials=1000, deadline=None):
    """
    Random search for a rank-r decomposition.
    Uses random sparse patterns and solves for the remaining degrees of freedom.
    """
    if deadline is None:
        deadline = time.time() + 115
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Generate random sparse patterns for U, V, W
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # For each rank component, randomly choose which entries are non-zero
        for c in range(target_rank):
            # Choose random support for U
            num_nonzero_U = random.randint(1, min(4, n*m))
            indices_U = random.sample(range(n*m), num_nonzero_U)
            for idx in indices_U:
                U[idx][c] = random.choice([-1, 1])
            
            num_nonzero_V = random.randint(1, min(4, m*p))
            indices_V = random.sample(range(m*p), num_nonzero_V)
            for idx in indices_V:
                V[idx][c] = random.choice([-1, 1])
            
            num_nonzero_W = random.randint(1, min(4, p*n))
            indices_W = random.sample(range(p*n), num_nonzero_W)
            for idx in indices_W:
                W[idx][c] = random.choice([-1, 1])
        
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None

def _constructive_low_rank(n, m, p):
    """
    Try to construct a low-rank decomposition using known techniques.
    For 3x4x5, we can try various block decompositions.
    """
    # Strategy: Use the fact that rank(3x4x5) <= rank(2x2x2) + rank(2x2x3) + ...
    # Let's try a systematic approach using known bounds.
    
    # One approach: decompose using Strassen-like patterns on sub-blocks
    # A is 3x4, B is 4x5
    # Split rows of A: [1, 2] and columns of B: [2, 3]
    # C = A*B where A = [A1; A2] (1x4, 2x4) and B = [B1, B2] (4x2, 4x3)
    # C11 = A1*B1 (1x2), C12 = A1*B2 (1x3)
    # C21 = A2*B1 (2x2), C22 = A2*B2 (2x3)
    
    # Rank contributions:
    # C11: 1x2x1 tensor -> rank 1 (outer product)
    # C12: 1x3x1 tensor -> rank 1
    # C21: 2x2x2 tensor -> rank 7 (Strassen)
    # C22: 2x3x2 tensor -> need to find rank
    
    # Total so far: 1 + 1 + 7 = 9, plus rank of 2x3x2
    # Known: rank(2x3x2) = rank(2x2x3) by symmetry
    # rank(2x2x3) is at most 10, possibly 9
    
    # Let's try to construct a rank-9 or rank-10 decomposition for 2x2x3
    # Then total would be 9 + 9 = 18 or 9 + 10 = 19
    
    # For now, return None and use random search
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(U_school, V_school, W_school, n, m, p):
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decompositions
    # Strategy 1: Try known low-rank constructions
    # Strategy 2: Random search for lower ranks
    
    # Try ranks from low to high
    min_possible = max(n, m, p)  # Lower bound
    max_to_try = best_rank - 1
    
    # First, try a few known constructions
    # Try Strassen-based block decomposition
    # For 3x4x5, try splitting into blocks
    
    # Try random search for various ranks
    for target_rank in range(40, 60):  # Try ranks from 40 to 59
        if time.time() > deadline - 5:
            break
        
        result = _random_search_decomposition(n, m, p, target_rank, num_trials=500, deadline=deadline - 5)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                # Found a valid decomposition, try to improve
                break
    
    # If we found something better, try to improve further
    if best_rank < n * m * p:
        for target_rank in range(max(min_possible, best_rank - 5), best_rank):
            if time.time() > deadline - 5:
                break
            
            result = _random_search_decomposition(n, m, p, target_rank, num_trials=200, deadline=deadline - 5)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
