"""
Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
Uses structured construction with systematic search for low-rank solutions.
"""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    nm, mp, pn = n * m, m * p, p * n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_structured_rank(n, m, p, r, num_trials=50):
    """
    Try to find a rank-r decomposition using structured random initialization.
    Key idea: exploit the fact that for fixed (i,k), the equation decouples.
    
    For each (i,k): sum_{j1,j2,c} U[i*m+j1,c]*V[j2*p+k,c]*W[k*n+i,c] * delta_{j1,j2} = 1
    Actually the condition is: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) = sum_j A[i,j]*B[j,k]
    
    This means: sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j2*p+k,c] = delta_{j1,j2} for all j1,j2
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for trial in range(num_trials):
        if _time_remaining() < 2.0:
            break
        
        # Initialize U and V with random ±1 entries, sparse
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        
        for c in range(r):
            # Each column of U: pick ~2-3 random rows
            num_u = min(nm, max(2, r // 3))
            u_rows = np.random.choice(nm, size=num_u, replace=False)
            for row in u_rows:
                U[row, c] = np.random.choice([-1, 1])
            
            num_v = min(mp, max(2, r // 3))
            v_rows = np.random.choice(mp, size=num_v, replace=False)
            for row in v_rows:
                V[row, c] = np.random.choice([-1, 1])
        
        # Now solve for W: for each (i,k), we have m*m equations in r unknowns
        # Equation: for j1,j2: sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
        
        W = np.zeros((pn, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build m*m x r system
                A_sys = np.zeros((m * m, r))
                b_sys = np.zeros(m * m)
                
                eq_idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        b_sys[eq_idx] = 1.0 if j1 == j2 else 0.0
                        for c in range(r):
                            A_sys[eq_idx, c] = U[i * m + j1, c] * V[j2 * p + k, c]
                        eq_idx += 1
                
                # Solve least squares
                try:
                    W_ik, residuals, rank_A, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    W[k * n + i, :] = W_ik
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return U, V, W, r
    
    return None

def _try_rank_range(n, m, p, r_start, r_end, trials_per_rank=20):
    """Try ranks from r_start to r_end-1, return best found."""
    best = None
    for r in range(r_start, r_end):
        if _time_remaining() < 5.0:
            break
        if r >= 60:  # Schoolbook is 60, no point going above
            break
        
        result = _try_structured_rank(n, m, p, r, num_trials=trials_per_rank)
        if result is not None:
            U, V, W, rank = result
            if best is None or rank < best[3]:
                best = (U, V, W, rank)
                # Found a valid one, try to find even lower
                break
    
    return best

def construct():
    n, m, p = 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    # Try to find lower rank
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But tensor rank can be higher. Let's try from 30 upward.
    
    # First try a few ranks with more trials
    for r in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]:
        if _time_remaining() < 10.0:
            break
        if r >= best_rank:
            break
        
        result = _try_structured_rank(n, m, p, r, num_trials=30)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # If we found something, try to improve further
    if best_rank < 60:
        for r in range(30, best_rank):
            if _time_remaining() < 5.0:
                break
            result = _try_structured_rank(n, m, p, r, num_trials=20)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
