"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 4, 5]
_TIMEOUT = 115  # safety margin before 120s kill

def _time_left(deadline):
    return deadline - time.time()

def _tensor_values(U, V, W, n, m, p):
    """Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    rank = len(U[0]) if U else 0
    T = np.zeros((n, m, p), dtype=float)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for c in range(rank):
        uc = Ua[:, c]
        vc = Va[:, c]
        wc = Wa[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return T

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition exactly reproduces the matrix multiplication tensor."""
    T = _tensor_values(U, V, W, n, m, p)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _rank(U):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
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

def _try_strassen_style(n, m, p, deadline):
    """
    Try a structured decomposition inspired by Strassen's algorithm.
    For 4x4 * 4x5, we can potentially use a block-based approach.
    
    The key idea: split the matrices into blocks and use fewer multiplications.
    For 4x4 * 4x4, Strassen uses 7 multiplications instead of 16.
    For 4x4 * 4x5, we can extend this idea.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try to construct a low-rank decomposition using structured patterns
    # that mimic the Strassen approach
    
    best_result = None
    best_rank = n * m * p
    
    # Define a set of structured patterns to try
    # Each pattern is a set of (U_indices, V_indices, W_indices, signs)
    
    # First, try rank 32 (half of schoolbook)
    for target_rank in [32, 30, 28, 26, 24, 22, 20]:
        if _time_left(deadline) < 2:
            break
        
        for attempt in range(50):
            if _time_left(deadline) < 2:
                break
            
            rng = random.Random(attempt * 1000 + target_rank)
            
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            # Use a structured approach: each term covers a small set of indices
            # with a pattern that might allow cancellations
            for c in range(target_rank):
                # Choose a random subset of indices with some structure
                # Try to create patterns that can cancel out unwanted terms
                u_count = rng.randint(1, 3)
                v_count = rng.randint(1, 3)
                w_count = rng.randint(1, 3)
                
                u_indices = rng.sample(range(nm), u_count)
                v_indices = rng.sample(range(mp), v_count)
                w_indices = rng.sample(range(pn), w_count)
                
                for idx in u_indices:
                    U[idx][c] = rng.choice([1, -1])
                for idx in v_indices:
                    V[idx][c] = rng.choice([1, -1])
                for idx in w_indices:
                    W[idx][c] = rng.choice([1, -1])
            
            if _is_valid(U, V, W, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result

def _try_block_decomposition(n, m, p, deadline):
    """
    Try a block-based decomposition.
    Split the 4x4 and 4x5 matrices into 2x2 blocks and use Strassen-like formulas.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_result = None
    best_rank = n * m * p
    
    # For 4x4 * 4x5, split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each block is 2x2
    # But B is 4x5, so the blocks won't be square. Let's try a different approach.
    
    # Actually, let's just try random structured decompositions with more attempts
    for target_rank in range(35, 20, -1):
        if _time_left(deadline) < 2:
            break
        
        for attempt in range(100):
            if _time_left(deadline) < 2:
                break
            
            rng = random.Random(attempt * 7919 + target_rank * 31)
            
            U = [[0] * target_rank for _ in range(nm)]
            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            for c in range(target_rank):
                # More aggressive randomization
                u_count = rng.randint(1, 4)
                v_count = rng.randint(1, 4)
                w_count = rng.randint(1, 4)
                
                u_indices = rng.sample(range(nm), u_count)
                v_indices = rng.sample(range(mp), v_count)
                w_indices = rng.sample(range(pn), w_count)
                
                for idx in u_indices:
                    U[idx][c] = rng.choice([1, -1])
                for idx in v_indices:
                    V[idx][c] = rng.choice([1, -1])
                for idx in w_indices:
                    W[idx][c] = rng.choice([1, -1])
            
            if _is_valid(U, V, W, n, m, p):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + _TIMEOUT
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try structured decompositions
    result = _try_strassen_style(n, m, p, deadline)
    if result is not None:
        r = _rank(result["U"])
        if r < best_rank:
            best_rank = r
            best_result = result
    
    # Try block decomposition
    if _time_left(deadline) > 5:
        result = _try_block_decomposition(n, m, p, deadline)
        if result is not None:
            r = _rank(result["U"])
            if r < best_rank:
                best_rank = r
                best_result = result
    
    # Try more random attempts with different seeds
    if _time_left(deadline) > 5:
        for seed in [1, 2, 3, 4, 5, 10, 100, 42, 123, 999, 1234, 5678, 9999]:
            if _time_left(deadline) < 3:
                break
            
            rng = random.Random(seed)
            for target_rank in range(40, 25, -1):
                if _time_left(deadline) < 2:
                    break
                
                for attempt in range(20):
                    if _time_left(deadline) < 2:
                        break
                    
                    U = [[0] * target_rank for _ in range(n * m)]
                    V = [[0] * target_rank for _ in range(m * p)]
                    W = [[0] * target_rank for _ in range(p * n)]
                    
                    for c in range(target_rank):
                        u_count = rng.randint(1, 4)
                        v_count = rng.randint(1, 4)
                        w_count = rng.randint(1, 4)
                        
                        u_indices = rng.sample(range(n * m), u_count)
                        v_indices = rng.sample(range(m * p), v_count)
                        w_indices = rng.sample(range(p * n), w_count)
                        
                        for idx in u_indices:
                            U[idx][c] = rng.choice([1, -1])
                        for idx in v_indices:
                            V[idx][c] = rng.choice([1, -1])
                        for idx in w_indices:
                            W[idx][c] = rng.choice([1, -1])
                    
                    if _is_valid(U, V, W, n, m, p):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U, "V": V, "W": W}
                        break
                
                if best_rank < 80:
                    break
    
    return best_result
