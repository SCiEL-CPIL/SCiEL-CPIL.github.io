"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Baseline: Rank 18 (Strassen + Schoolbook).
2. Search: Rank 17 using ALS initialization + Discrete Local Search (Coordinate Descent on half-integers).
3. Search: Rank 16 (less likely, but attempted with same method).
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T


def _build_target_float():
    return _build_target().astype(np.float64)


def _check_decomposition_float(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros((T.shape[0], T.shape[1], T.shape[2]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices."""
    R = U.shape[1]
    n, m, p = DIMENSIONS
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _snap_to_half_integers(M):
    """Snap matrix values to nearest half-integer."""
    return np.round(M * 2) / 2.0


def _snap_to_integers(M):
    """Snap matrix values to nearest integer."""
    return np.round(M)


def _to_fraction(val):
    """Convert a float to Fraction, preferring small denominators."""
    if abs(val) < 1e-12:
        return Fraction(0)
    # Try half-integer first
    nearest_half = np.round(val * 2) / 2.0
    if abs(val - nearest_half) < 1e-9:
        return Fraction(int(nearest_half * 2), 2)
    # Try integer
    nearest_int = np.round(val)
    if abs(val - nearest_int) < 1e-9:
        return Fraction(int(nearest_int))
    # Try other small denominators
    for denom in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]:
        num = np.round(val * denom)
        if abs(val - num / denom) < 1e-9:
            frac = Fraction(int(num), denom)
            return frac
    # Last resort
    return Fraction(val).limit_denominator(200)


def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U_frac[i, j] = _to_fraction(U[i, j])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_frac[i, j] = _to_fraction(V[i, j])
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_frac[i, j] = _to_fraction(W[i, j])
    
    R = U.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _get_rank18_solution():
    """
    Returns the structural Rank 18 solution:
    Two Strassen blocks (cols 0-1, 2-3) and Schoolbook for col 4.
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        # M1: (a00+a11)(b00+b11)
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M2: (a10+a11)(b00)
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        # M3: (a00)(b11-b01)
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M4: (a11)(b10-b00)
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        # M5: (a00+a01)(b11)
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        # M6: (a10-a00)(b00+b01)
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        # M7: (a01-a11)(b10+b11)
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for col 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W


def _als_update_U(T, V, W, R):
    n, m, p = DIMENSIONS
    U_rows = n * m
    VW = np.einsum('vr,wr->vwr', V, W)
    VW_flat = VW.reshape(-1, R)
    U = np.zeros((U_rows, R), dtype=np.float64)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(VW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_update_V(T, U, W, R):
    n, m, p = DIMENSIONS
    V_rows = m * p
    UW = np.einsum('ur,wr->uwr', U, W)
    UW_flat = UW.reshape(-1, R)
    V = np.zeros((V_rows, R), dtype=np.float64)
    for v in range(V_rows):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UW_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_update_W(T, U, V, R):
    n, m, p = DIMENSIONS
    W_rows = p * n
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((W_rows, R), dtype=np.float64)
    for w in range(W_rows):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _compute_error(U, V, W, T):
    result = _compute_tensor(U, V, W)
    return np.linalg.norm(result - T)

def _als_polish(U, V, W, T, R, max_iters=100, tol=1e-15):
    """High-precision ALS polish."""
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_update_U(T, V, W, R)
        V = _als_update_V(T, U, W, R)
        W = _als_update_W(T, U, V, R)
        err = _compute_error(U, V, W, T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _discrete_local_search(U, V, W, T, max_rounds=50):
    """
    Perform coordinate descent on the discrete lattice of half-integers.
    Assumes U, V, W are already snapped to half-integers.
    """
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    def get_error_sq(U, V, W):
        # Compute tensor and compare to T
        tensor = _compute_tensor(U, V, W)
        return np.sum((tensor - T) ** 2)
    
    current_err = get_error_sq(U, V, W)
    
    improved = True
    while improved and max_rounds > 0:
        improved = False
        max_rounds -= 1
        
        # Try flipping U entries
        for u in range(U.shape[0]):
            for r in range(R):
                old_val = U[u, r]
                # Neighbors: old_val - 0.5, old_val + 0.5
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    # Check if new_val is a valid half-integer (it is, by construction)
                    U[u, r] = new_val
                    new_err = get_error_sq(U, V, W)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                    else:
                        U[u, r] = old_val
        
        # Try flipping V entries
        for v in range(V.shape[0]):
            for r in range(R):
                old_val = V[v, r]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    V[v, r] = new_val
                    new_err = get_error_sq(U, V, W)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                    else:
                        V[v, r] = old_val
        
        # Try flipping W entries
        for w in range(W.shape[0]):
            for r in range(R):
                old_val = W[w, r]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    W[w, r] = new_val
                    new_err = get_error_sq(U, V, W)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                    else:
                        W[w, r] = old_val
                        
    return U, V, W, current_err

def _try_als_rank(R, T, deadline, n_seeds=50, max_iters=50):
    """
    Try to find a rank-R decomposition using ALS with multiple initialization strategies.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    T_float = T.astype(np.float64)
    
    # Use Rank 18 solution as a structured initialization
    U18, V18, W18 = _get_rank18_solution()
    
    best_err = np.inf
    best_solution = None
    
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333, 
             1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1111, 2222, 3333, 4444, 5555]
    
    for seed_idx in range(min(n_seeds, len(seeds))):
        if time.time() > deadline:
            break
        seed = seeds[seed_idx]
        np.random.seed(seed)
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Pure random
        U = np.random.randn(U_rows, R) * 0.5
        V = np.random.randn(V_rows, R) * 0.5
        W = np.random.randn(W_rows, R) * 0.5
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 2: Random with different scales
        U = np.random.randn(U_rows, R)
        V = np.random.randn(V_rows, R)
        W = np.random.randn(W_rows, R)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 3: Sparse random
        U = np.random.randn(U_rows, R) * (np.random.rand(U_rows, R) < 0.3)
        V = np.random.randn(V_rows, R) * (np.random.rand(V_rows, R) < 0.3)
        W = np.random.randn(W_rows, R) * (np.random.rand(W_rows, R) < 0.3)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 4: For R=17, try structured init from rank-18
        if R == 17:
            # Take first 17 columns of rank-18 and add small perturbation
            U_s = U18[:, :17].copy() + np.random.randn(U_rows, 17) * 0.01
            V_s = V18[:, :17].copy() + np.random.randn(V_rows, 17) * 0.01
            W_s = W18[:, :17].copy() + np.random.randn(W_rows, 17) * 0.01
            inits.append((U_s, V_s, W_s))
            
            # Try dropping different columns from rank-18
            for drop_col in range(18):
                mask = [i for i in range(18) if i != drop_col]
                U_s = U18[:, mask].copy() + np.random.randn(U_rows, 17) * 0.01
                V_s = V18[:, mask].copy() + np.random.randn(V_rows, 17) * 0.01
                W_s = W18[:, mask].copy() + np.random.randn(W_rows, 17) * 0.01
                inits.append((U_s, V_s, W_s))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
            
            U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = _als_update_U(T_float, V, W, R)
                V = _als_update_V(T_float, U, W, R)
                W = _als_update_W(T_float, U, V, R)
                
                err = _compute_error(U, V, W, T_float)
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check for exact solution
            if _check_decomposition_float(U, V, W, T_float):
                # Polish with high precision
                U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=50, tol=1e-16)
                
                # Try to rationalize
                for snap_fn in [_snap_to_half_integers, _snap_to_integers]:
                    U_s = snap_fn(U)
                    V_s = snap_fn(V)
                    W_s = snap_fn(W)
                    
                    if _check_decomposition_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                
                # Check if original is exact
                if _check_decomposition_exact(U, V, W, T):
                    return U.tolist(), V.tolist(), W.tolist()
    
    # If we found a very low error solution, try one more polish and discrete search
    if best_solution is not None and best_err < 1e-6:
        U, V, W = best_solution
        U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=200, tol=1e-16)
        
        # Snap to half-integers
        U_s = _snap_to_half_integers(U)
        V_s = _snap_to_half_integers(V)
        W_s = _snap_to_half_integers(W)
        
        # Check if snapped is exact
        if _check_decomposition_exact(U_s, V_s, W_s, T):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
        # If not exact, try discrete local search
        if time.time() < deadline - 5:
            U_ls, V_ls, W_ls, ls_err = _discrete_local_search(U_s, V_s, W_s, T_float, max_rounds=20)
            if _check_decomposition_exact(U_ls, V_ls, W_ls, T):
                return U_ls.tolist(), V_ls.tolist(), W_ls.tolist()
                
        if _check_decomposition_exact(U, V, W, T):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    T = _build_target()
    T_float = T.astype(np.float64)

    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}

    # Try to find Rank 17 with structured initialization
    search_deadline_17 = min(deadline - 20, start_time + 120)
    result = _try_als_rank(17, T, search_deadline_17, n_seeds=80, max_iters=40)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}

    # If time permits, try Rank 16 with ALS
    if time.time() < deadline - 5:
        result = _try_als_rank(16, T, deadline - 2, n_seeds=10, max_iters=30)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}

    return best_result
