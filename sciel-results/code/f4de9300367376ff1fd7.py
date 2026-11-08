"""Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.

The matmul tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p] must be
decomposed as sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1.
"""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]
DEADLINE = 145.0


def _build_constraint_indices(n, m, p):
    """Build active and inactive constraint indices.
    
    Active: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    Inactive: all other (a, b, c) triples
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    active = []
    active_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active.append((a, b, c))
                active_set.add((a, b, c))
    
    inactive = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in active_set:
                    inactive.append((a, b, c))
    
    return active, inactive


def _residuals(params, R, n, m, p, active, inactive):
    """Compute residuals for the tensor decomposition constraints."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = params[:R * nm].reshape(nm, R)
    V = params[R * nm:R * nm + R * mp].reshape(mp, R)
    W = params[R * nm + R * mp:].reshape(pn, R)
    
    residuals = []
    
    # Active constraints: sum_t U[a,t] V[b,t] W[c,t] = 1
    for a, b, c in active:
        val = np.dot(U[a] * V[b], W[c])
        residuals.append(val - 1.0)
    
    # Inactive constraints: sum_t U[a,t] V[b,t] W[c,t] = 0
    for a, b, c in inactive:
        val = np.dot(U[a] * V[b], W[c])
        residuals.append(val)
    
    return np.array(residuals)


def _verify_solution(U, V, W, n, m, p, active, inactive, tol=1e-9):
    """Verify that the decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for a, b, c in active:
        val = np.dot(U[a] * V[b], W[c])
        if abs(val - 1.0) > tol:
            return False
    
    for a, b, c in inactive:
        val = np.dot(U[a] * V[b], W[c])
        if abs(val) > tol:
            return False
    
    return True


def _try_rank(R, n, m, p, active, inactive, start_time, deadline, n_init=10):
    """Try to find a valid rank-R decomposition using multiple random inits."""
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = R * (nm + mp + pn)
    
    for init_seed in range(n_init):
        if time.time() > deadline - 5.0:
            return None
        
        rng = np.random.default_rng(1000 + R * 100 + init_seed)
        x0 = rng.standard_normal(total_params) * 0.5
        
        try:
            result = least_squares(
                _residuals, x0, args=(R, n, m, p, active, inactive),
                method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15
            )
            
            if result.cost < 1e-12:
                params = result.x
                U = params[:R * nm].reshape(nm, R)
                V = params[R * nm:R * nm + R * mp].reshape(mp, R)
                W = params[R * nm + R * mp:].reshape(pn, R)
                
                # Try snapping to half-integers
                for snap_factor in [2, 4, 1]:
                    U_snap = np.round(U * snap_factor) / snap_factor
                    V_snap = np.round(V * snap_factor) / snap_factor
                    W_snap = np.round(W * snap_factor) / snap_factor
                    
                    if _verify_solution(U_snap, V_snap, W_snap, n, m, p, active, inactive, tol=1e-9):
                        return U_snap, V_snap, W_snap
                
                if _verify_solution(U, V, W, n, m, p, active, inactive, tol=1e-10):
                    return U, V, W
                    
        except Exception:
            continue
    
    return None


def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    active, inactive = _build_constraint_indices(n, m, p)
    
    # Theoretical lower bound: n + m + p - 1 = 9
    # Trivial upper bound: n*m*p = 30
    
    lower_bound = n + m + p - 1
    upper_bound = n * m * p
    
    rank_candidates = list(range(12, 26)) + list(range(26, upper_bound + 1))
    
    for R in rank_candidates:
        if time.time() > DEADLINE - 10.0:
            break
        
        result = _try_rank(R, n, m, p, active, inactive, start, DEADLINE, n_init=10)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: trivial rank 30 decomposition
    U, V, W = _construct_trivial(n, m, p)
    return {"U": U, "V": V, "W": W}
