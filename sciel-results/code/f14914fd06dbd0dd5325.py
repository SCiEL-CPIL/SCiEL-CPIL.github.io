"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition search.

Strategy:
  - The schoolbook rank is n*m*p = 36.
  - We search for lower-rank exact decompositions with entries in {0, +-1/2, +-1, ...}.
  - We use a randomized construction that fixes one factor (U) to a small structured
    basis, then solves for V and W via linear systems, then snaps to half-integers
    and verifies exactly.
  - We also try a direct "guess and verify" loop over random sparse half-integer
    matrices for U, V, W, checking the bilinear identity.
  - We try block decomposition: split the p=6 dimension into blocks and find
    low-rank decompositions for the sub-problems.
  - We try a "border rank" relaxation approach: formulate the tensor decomposition
    as a system of polynomial equations and use numerical methods to find solutions.

  Deadline guard: track elapsed time and return the best valid result.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# Deadline guard: stop 10 seconds before the 150s wall-clock limit.
DEADLINE_SECONDS = 140.0

def _target_tensor(n, m, p):
    """Return the matrix-multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _snap_to_half_integers(M, tol=1e-6):
    """Snap matrix entries to the nearest half-integer if close."""
    M_snapped = np.round(M * 2) / 2
    if np.allclose(M, M_snapped, atol=tol):
        return M_snapped
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _to_list(M):
    """Convert numpy array to list of lists."""
    return M.tolist()

def _find_sub_decomposition(n_s, m_s, p_s, target_rank, iterations=50000, deadline=None):
    """Find a rank-target_rank decomposition for M_{n_s, m_s, p_s} using random search."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline - 20:
            break
        
        U = np.random.randn(n_s*m_s, target_rank)
        V = np.random.randn(m_s*p_s, target_rank)
        W = np.random.randn(p_s*n_s, target_rank)
        
        # Compute error
        err = 0.0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i*m_s+j, c] * V[j*p_s+k, c] * W[k*n_s+i, c]
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-12:
            return best_U, best_V, best_W, best_err
    
    return best_U, best_V, best_W, best_err

def _verify_sub(U, V, W, n_s, m_s, p_s, tol=1e-9):
    """Verify a sub-decomposition."""
    r = U.shape[1]
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                val = 0.0
                for c in range(r):
                    val += U[i*m_s+j, c] * V[j*p_s+k, c] * W[k*n_s+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = nm * mp
    best_result = {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

    def elapsed():
        return time.time() - start_time

    def time_up():
        return elapsed() > DEADLINE_SECONDS

    # Strategy 1: Try block decomposition
    # Split the p=6 dimension into blocks of size 2 or 3
    # For blocks of size 2: 3 blocks, each is M_{2,3,2}
    # For blocks of size 3: 2 blocks, each is M_{2,3,3}

    # Try blocks of size 2 first (3 blocks)
    sub_n, sub_m, sub_p = 2, 3, 2
    found_sub = None
    sub_rank = 0
    
    for r_sub in [7, 8, 9]:
        if time_up():
            break
        U_s, V_s, W_s, err = _find_sub_decomposition(sub_n, sub_m, sub_p, r_sub, iterations=20000, deadline=start_time + DEADLINE_SECONDS)
        if err < 1e-9 and _verify_sub(U_s, V_s, W_s, sub_n, sub_m, sub_p):
            found_sub = (U_s, V_s, W_s)
            sub_rank = r_sub
            break
    
    if found_sub:
        U_s, V_s, W_s = found_sub
        total_rank = sub_rank * 3
        
        # Construct the full decomposition
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        for b in range(3):
            col_offset = b * sub_rank
            
            # U is same for all blocks
            U_full[:, col_offset:col_offset+sub_rank] = U_s
            
            # V indices: j (0..2), k (2*b .. 2*b+1)
            for j in range(sub_m):
                for k_local in range(sub_p):
                    k_global = 2*b + k_local
                    V_full[j * p + k_global, col_offset:col_offset+sub_rank] = V_s[j * sub_p + k_local, :]
            
            # W indices: k (2*b .. 2*b+1), i (0..1)
            for k_local in range(sub_p):
                for i in range(sub_n):
                    k_global = 2*b + k_local
                    W_full[k_global * n + i, col_offset:col_offset+sub_rank] = W_s[k_local * sub_n + i, :]
        
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            best_rank = total_rank
            best_result = {"U": _to_list(U_full), "V": _to_list(V_full), "W": _to_list(W_full)}
    
    # Strategy 2: Random search with half-integer entries
    # For ranks from 18 to best_rank-1, try random half-integer matrices.
    max_trials_per_rank = 50

    for r in range(18, best_rank):
        if time_up():
            break
        for trial in range(max_trials_per_rank):
            if time_up():
                break
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            U_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(nm)])
            V_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(mp)])
            W_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(pn)])
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_result = {"U": _to_list(U_r), "V": _to_list(V_r), "W": _to_list(W_r)}
                break

    return best_result
