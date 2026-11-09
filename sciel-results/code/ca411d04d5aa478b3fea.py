"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _compute_error_fast(U, V, W, n, m, p):
    """Vectorized computation of max absolute error using tensor structure."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # For each (i_A, j_A) in U, (j_B, k_B) in V, (k_C, i_C) in W:
    # T[i,j,k] = 1 iff j_A==j_B and i_A==i_C and k_B==k_C
    # We only need to check compatible triples.
    
    max_err = 0.0
    for i_A in range(n):
        for j_A in range(m):
            i = i_A * m + j_A
            for k_B in range(p):
                j = j_A * p + k_B
                k = k_B * n + i_A
                # This is a compatible triple, expected = 1
                actual = np.dot(U[i], V[j] * W[k])
                err = abs(actual - 1.0)
                if err > max_err:
                    max_err = err
                # Also check incompatible triples (sample some)
                # For incompatible: i_A != i_C or j_A != j_B or k_B != k_C
                # We'll check a subset for speed
    return max_err

def _compute_error_full(U, V, W, n, m, p):
    """Full error computation checking all triples."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    max_err = 0.0
    for i_A in range(n):
        for j_A in range(m):
            i = i_A * m + j_A
            for j_B in range(m):
                for k_B in range(p):
                    j = j_B * p + k_B
                    for k_C in range(p):
                        for i_C in range(n):
                            k = k_C * n + i_C
                            expected = 1.0 if (j_A == j_B and i_A == i_C and k_B == k_C) else 0.0
                            actual = np.dot(U[i], V[j] * W[k])
                            err = abs(actual - expected)
                            if err > max_err:
                                max_err = err
    return max_err

def _try_rank(rank, n, m, p, time_limit=10, seed=42):
    """Try to find a decomposition of given rank using iterative refinement with structure-aware init."""
    start_time = time.time()
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(seed)
    
    n_attempts = 8
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Structure-aware initialization:
        # Initialize with small random values but bias towards the tensor structure
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Iterative refinement with alternating least squares
        max_iters = 200
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Update U: for each i, solve least squares
            for i in range(nm):
                i_A = i // m
                j_A = i % m
                
                # Compatible (j, k) pairs: j = j_A*p + k_B, k = k_B*n + i_A
                A_rows = []
                for k_B in range(p):
                    j = j_A * p + k_B
                    k = k_B * n + i_A
                    A_rows.append(V[j] * W[k])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                try:
                    U[i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Update V: for each j, solve least squares
            for j in range(mp):
                j_B = j // p
                k_B = j % p
                
                A_rows = []
                for i_A in range(n):
                    i = i_A * m + j_B
                    k = k_B * n + i_A
                    A_rows.append(U[i] * W[k])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                try:
                    V[j], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Update W: for each k, solve least squares
            for k in range(pn):
                k_C = k // n
                i_C = k % n
                
                A_rows = []
                for j_A in range(m):
                    i = i_C * m + j_A
                    j = j_A * p + k_C
                    A_rows.append(U[i] * V[j])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                try:
                    W[k], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Check error periodically
            if iteration % 10 == 0 or iteration == max_iters - 1:
                err = _compute_error_fast(U, V, W, n, m, p)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err < 1e-10:
                    # Do a polish pass with full error check
                    full_err = _compute_error_full(U, V, W, n, m, p)
                    if full_err < 1e-9:
                        break
                    # Continue refining
                    best_err = full_err
    
    # Final verification
    if best_U is not None and best_err < 1e-9:
        full_err = _compute_error_full(best_U, best_V, best_W, n, m, p)
        if full_err < 1e-9:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140  # 10 second safety margin
    
    # Try ranks from just below schoolbook down to a minimum
    # Start with a few ranks that are more likely to have solutions
    min_rank = max(1, n * m + m * p + p * n - 2)  # Lower bound heuristic
    
    # Try a set of candidate ranks
    candidate_ranks = sorted(set(range(min_rank, n*m*p, 1)), reverse=True)
    
    # Limit the number of ranks to try based on time
    max_ranks_to_try = 15
    
    for rank in candidate_ranks[:max_ranks_to_try]:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        time_limit = min(remaining * 0.8, 12)
        
        result = _try_rank(rank, n, m, p, time_limit=time_limit, seed=42)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
