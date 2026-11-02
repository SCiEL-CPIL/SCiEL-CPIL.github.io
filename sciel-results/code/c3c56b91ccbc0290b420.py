"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _triples(n, m, p):
    """List of (u_index, v_index, w_index) for each (i,j,k)."""
    tr = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tr.append((u, v, w))
    return tr

def _residual_fast(U, V, W, tr):
    """Vectorized max abs residual."""
    r = U.shape[1]
    max_err = 0.0
    for (u, v, w) in tr:
        s = np.dot(U[u] * V[v], W[w])
        e = abs(s - 1.0)
        if e > max_err:
            max_err = e
    return max_err

def _round_to_halfint(A, tol=1e-6):
    """Round to nearest half-integer if close enough, else return None."""
    A2 = A * 2.0
    A_rounded = np.round(A2)
    if np.max(np.abs(A2 - A_rounded)) < tol:
        return A_rounded / 2.0
    return None

def _verify_exact(U, V, W, tr, tol=1e-9):
    """Check if decomposition is exact."""
    return _residual_fast(U, V, W, tr) < tol

def _als_step_fast(U, V, W, tr, n, m, p):
    """One ALS iteration with vectorized operations."""
    r = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Precompute maps
    u_map = {}
    v_map = {}
    w_map = {}
    for (u, v, w) in tr:
        u_map.setdefault(u, []).append((v, w))
        v_map.setdefault(v, []).append((u, w))
        w_map.setdefault(w, []).append((u, v))
    
    # Solve U
    for u, lst in u_map.items():
        A = np.array([V[v] * W[w] for (v, w) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        U_new[u] = sol
    
    # Solve V
    for v, lst in v_map.items():
        A = np.array([U_new[u] * W[w] for (u, w) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        V_new[v] = sol
    
    # Solve W
    for w, lst in w_map.items():
        A = np.array([U_new[u] * V_new[v] for (u, v) in lst])
        b = np.ones(len(lst))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        W_new[w] = sol
    
    return U_new, V_new, W_new

def _search_rank(r, n, m, p, seed, iters, start_time, deadline):
    """Search for a rank-r decomposition."""
    tr = _triples(n, m, p)
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Initialize
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        for _ in range(min(iters, 200)):
            if time.time() > deadline:
                break
            U, V, W = _als_step_fast(U, V, W, tr, n, m, p)
            err = _residual_fast(U, V, W, tr)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        if best_err < 1e-8:
            break
    
    return best_err, best_U, best_V, best_W

def _try_small_integer_search(n, m, p, max_val=3, start_time=None, deadline=None):
    """Try to find a decomposition with small integer entries."""
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 10
    
    tr = _triples(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    # Try ranks from small to large
    for r in range(10, 50):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(r * 1000 + 42)
        
        # Try a few random initializations with small values
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U = rng.randint(-max_val, max_val + 1, (nm, r)) * 0.5
            V = rng.randint(-max_val, max_val + 1, (mp, r)) * 0.5
            W = rng.randint(-max_val, max_val + 1, (pn, r)) * 0.5
            
            # Run ALS
            for _ in range(50):
                if time.time() > deadline:
                    break
                U, V, W = _als_step_fast(U, V, W, tr, n, m, p)
                err = _residual_fast(U, V, W, tr)
                if err < 1e-8:
                    # Check if it's close to half-integers
                    U_h = _round_to_halfint(U, 1e-4)
                    V_h = _round_to_halfint(V, 1e-4)
                    W_h = _round_to_halfint(W, 1e-4)
                    if U_h is not None and V_h is not None and W_h is not None:
                        if _verify_exact(U_h, V_h, W_h, tr):
                            return r, U_h, V_h, W_h
            if time.time() > deadline:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    tr = _triples(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    sU, sV, sW = _schoolbook()
    best_result = (100, sU, sV, sW)  # (rank, U, V, W)
    
    # Strategy 1: Try small-integer search first (fast)
    small_result = _try_small_integer_search(n, m, p, max_val=4, start_time=start_time, deadline=min(deadline, start_time + 30))
    if small_result is not None:
        r, U, V, W = small_result
        if r < best_result[0]:
            best_result = (r, U, V, W)
    
    # Strategy 2: ALS search at various ranks
    candidate_ranks = [20, 25, 30, 35, 40, 45, 50]
    seeds = [42, 1, 7, 123, 999]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_result[0]:
            break
            
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            err, U, V, W = _search_rank(r, n, m, p, seed, 100, start_time, deadline - 5)
            if U is not None and err < 1e-6:
                # Try to round to half-integers
                U_h = _round_to_halfint(U, 1e-3)
                V_h = _round_to_halfint(V, 1e-3)
                W_h = _round_to_halfint(W, 1e-3)
                if U_h is not None and V_h is not None and W_h is not None:
                    if _verify_exact(U_h, V_h, W_h, tr, 1e-6):
                        if r < best_result[0]:
                            best_result = (r, U_h, V_h, W_h)
    
    # Strategy 3: If we have a good approximate solution, try to polish it
    if best_result[0] < 100 and time.time() < deadline - 5:
        r, U, V, W = best_result
        # Try more ALS iterations to tighten
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            U, V, W = _als_step_fast(U, V, W, tr, n, m, p)
            err = _residual_fast(U, V, W, tr)
            if err < 1e-10:
                break
        U_h = _round_to_halfint(U, 1e-4)
        V_h = _round_to_halfint(V, 1e-4)
        W_h = _round_to_halfint(W, 1e-4)
        if U_h is not None and V_h is not None and W_h is not None:
            if _verify_exact(U_h, V_h, W_h, tr, 1e-6):
                best_result = (r, U_h, V_h, W_h)
    
    # Return the best result found
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
