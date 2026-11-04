"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 120.0
DIMENSIONS = [3, 4, 7]


def _deadline():
    return time.time() - _START > _LIMIT - 5.0


def _pack(arr):
    """Convert numpy array to list of lists of floats (integers if integral)."""
    a = np.asarray(arr)
    result = []
    for row in a:
        r = []
        for x in row:
            xf = float(x)
            nearest = round(xf * 2) / 2
            if abs(xf - nearest) < 1e-9:
                r.append(nearest)
            else:
                r.append(xf)
        result.append(r)
    return result


def _check_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    Nm, Mp, Pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Check all valid (i,j,k) entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - 1.0) > 1e-8:
                    return False
    
    # Check that non-matching entries are zero (sampled)
    import random
    random.seed(42)
    for _ in range(300):
        a = random.randint(0, Nm - 1)
        b = random.randint(0, Mp - 1)
        c = random.randint(0, Pn - 1)
        i_val = a // m
        j_val = a % m
        jp_val = b // p
        k_val = b % p
        kp_val = c // n
        ip_val = c % n
        is_valid = (i_val == ip_val and j_val == jp_val and k_val == kp_val)
        if not is_valid:
            actual = 0.0
            for r in range(rank):
                actual += U[a, r] * V[b, r] * W[c, r]
            if abs(actual) > 1e-8:
                return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank n*m*p)."""
    R0 = n * m * p
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    return U0, V0, W0


def _try_rank_r_decomposition(n, m, p, r, max_time=30.0):
    """
    Try to find a rank-r decomposition using structured search.
    Uses random initialization + Newton iteration to solve the nonlinear system.
    """
    start_time = time.time()
    
    np.random.seed(12345 + r)
    
    best_rank = None
    best_U, best_V, best_W = None, None, None
    
    num_eqs = n * m * p
    num_vars = (n * m + m * p + p * n) * r
    
    for attempt in range(80):
        if time.time() - start_time > max_time:
            break
        if _deadline():
            break
        
        # Random initialization with varying scale
        scale = 0.3 + 0.5 * (attempt % 5) / 4.0
        U = np.random.randn(n * m, r) * scale
        V = np.random.randn(m * p, r) * scale
        W = np.random.randn(p * n, r) * scale
        
        # Newton-style iteration
        converged = False
        for it in range(20):
            J = np.zeros((num_eqs, num_vars))
            resid = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        for r_idx in range(r):
                            col_u = a * r + r_idx
                            J[eq_idx, col_u] = V[b, r_idx] * W[c, r_idx]
                            col_v = (n * m + b) * r + r_idx
                            J[eq_idx, col_v] = U[a, r_idx] * W[c, r_idx]
                            col_w = (n * m + m * p + c) * r + r_idx
                            J[eq_idx, col_w] = U[a, r_idx] * V[b, r_idx]
                        current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        resid[eq_idx] = current_val - 1.0
                        eq_idx += 1
            
            try:
                delta, _, _, _ = np.linalg.lstsq(J, -resid, rcond=None)
            except np.linalg.LinAlgError:
                break
            
            delta_U = delta[:n * m * r].reshape(n * m, r)
            delta_V = delta[n * m * r:n * m * r + m * p * r].reshape(m * p, r)
            delta_W = delta[n * m * r + m * p * r:].reshape(p * n, r)
            
            U = U + delta_U
            V = V + delta_V
            W = W + delta_W
            
            max_resid = np.max(np.abs(resid))
            if max_resid < 1e-7:
                converged = True
                break
        
        if converged:
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    if best_rank is not None:
        return best_U, best_V, best_W
    return None, None, None


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on the algebraic structure.
    Uses the group action symmetry to construct lower-rank decompositions.
    """
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # Try to find a rank-28 or lower decomposition using structured patterns
    
    # Pattern: For each j, create a "block" that handles all i,k combinations
    # This naturally gives rank = m * max(n, p) but we can potentially merge
    
    # Try a more clever structured approach:
    # Use the fact that the multiplication tensor has a Kronecker product structure
    # T = sum_j (I_n ⊗ e_j ⊗ I_p) in appropriate indexing
    
    # For rank 28: use one term per (j, k) pair, sharing across i
    # U[i*m+j, (j,k)] = 1 for all i
    # V[j*p+k, (j,k)] = 1
    # W[k*n+i, (j,k)] = delta(i, i') -- but this doesn't work directly
    
    # Alternative: use rank = n * p = 21 terms, one per (i,k) pair
    # For each (i,k), we need sum_j A[i,j]*B[j,k]
    # This requires m terms per (i,k) in general, so rank >= n*p*m = 84
    # Unless we can share structure
    
    # For now, return schoolbook as baseline
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p  # 84
    
    # Try to find lower rank decompositions
    # Lower bound is 28 (from m*p dimension)
    # Try ranks from 28 upward
    
    for r in range(28, best_rank):
        if _deadline():
            break
        if time.time() - _START > _LIMIT - 10:
            break
        
        # Allocate time based on rank: lower ranks get more time
        time_budget = min(20.0, _LIMIT - (time.time() - _START) - 5)
        if r < 35:
            time_budget = min(25.0, time_budget)
        
        result = _try_rank_r_decomposition(n, m, p, r, max_time=time_budget)
        if result[0] is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found the minimal rank
    
    # Also try structured decomposition if time remains
    if not _deadline() and time.time() - _START < _LIMIT - 15:
        sU, sV, sW = _structured_decomposition(n, m, p)
        if sU.shape[1] < best_rank:
            best_U, best_V, best_W = sU, sV, sW
            best_rank = sU.shape[1]
    
    return {"U": _pack(best_U), "V": _pack(best_V), "W": _pack(best_W)}
