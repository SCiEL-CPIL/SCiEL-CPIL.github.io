"""Rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4


def _verify_einsum(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def _verify_strict(U, V, W, n=N, m=M, p=P, tol=1e-12):
    """Strict verification."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def schoolbook(na, mb, pb):
    """Schoolbook decomposition for na x mb * mb x pb, rank na*mb*pb."""
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2. Rank 7.
    U: (4,7) indexed U[i*2+j], V: (4,7) indexed V[j*2+k], W: (4,7) indexed W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2: (A10+A11)*B00 -> C10(+1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3: A00*(B01-B11) -> C01(+1), C11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4: A11*(B10-B00) -> C00(+1), C10(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5: (A00+A10)*B11 -> C00(-1), C01(+1)
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6: (A01-A10)*(B00+B10) -> C11(+1)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1

    # M7: (A01-A11)*(B01+B11) -> C00(+1)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _add_sub(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start, n, m, p):
    """Add a sub-decomposition into the full matrices at the given offsets."""
    r = len(U_sub[0])
    for c in range(r):
        col = col_start + c
        for i_prime in range(na):
            for j_prime in range(mb):
                u_val = U_sub[i_prime*mb + j_prime][c]
                if u_val != 0:
                    U[(i_off + i_prime)*m + (j_off + j_prime)][col] += u_val
        for j_prime in range(mb):
            for k_prime in range(pb):
                v_val = V_sub[j_prime*pb + k_prime][c]
                if v_val != 0:
                    V[(j_off + j_prime)*p + (k_off + k_prime)][col] += v_val
        for k_prime in range(pb):
            for i_prime in range(na):
                w_val = W_sub[k_prime*na + i_prime][c]
                if w_val != 0:
                    W[(k_off + k_prime)*n + (i_off + i_prime)][col] += w_val


def _build_2plus1_split():
    """
    Split A (3x4) into rows 0-1 (2x4) and row 2 (1x4).
    Split B (4x4) into four 2x2 blocks.
    C (3x4) similarly split.
    
    2x2*2x2 blocks use Strassen (rank 7) -> 4 blocks * 7 = 28
    1x2*2x2 strips use schoolbook (rank 4) -> 4 strips * 4 = 16
    Total: 44
    """
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4  # 28 + 16 = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 44

    col = 0

    # Top 2 rows of C (i=0,1), left 2 cols (k=0,1):
    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 0, 0, 2, 2, 2, col, n, m, p)
    col += 7

    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 2, 0, 2, 2, 2, col, n, m, p)
    col += 7

    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 0, 2, 2, 2, 2, col, n, m, p)
    col += 7

    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 2, 2, 2, 2, 2, col, n, m, p)
    col += 7

    # Bottom row of C (i=2), left 2 cols (k=0,1):
    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 0, 0, 1, 2, 2, col, n, m, p)
    col += 4

    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 2, 0, 1, 2, 2, col, n, m, p)
    col += 4

    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 0, 2, 1, 2, 2, col, n, m, p)
    col += 4

    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 2, 2, 1, 2, 2, col, n, m, p)
    col += 4

    return U, V, W


def _build_candidate_pool(n, m, p):
    """
    Build a large pool of candidate rank-1 tensors for sparse recovery.
    Includes structured candidates from various block decompositions.
    Returns (U_pool, V_pool, W_pool) with shapes (n*m, K), (m*p, K), (p*n, K)
    """
    candidates = []  # List of (u_vec, v_vec, w_vec) tuples

    # Add schoolbook basis vectors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((u, v, w))

    # Add Strassen-like structured candidates for 2x2 subblocks
    for i_off in range(0, n-1, 1):
        for j_off in range(0, m-1, 1):
            for k_off in range(0, p-1, 1):
                Us, Vs, Ws = strassen_2x2()
                for c in range(7):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    for ii in range(2):
                        for jj in range(2):
                            u[(i_off+ii)*m + (j_off+jj)] = Us[ii*2+jj][c]
                    for jj in range(2):
                        for kk in range(2):
                            v[(j_off+jj)*p + (k_off+kk)] = Vs[jj*2+kk][c]
                    for kk in range(2):
                        for ii in range(2):
                            w[(k_off+kk)*n + (i_off+ii)] = Ws[kk*2+ii][c]
                    candidates.append((u, v, w))

    # Add random candidates
    rng = np.random.default_rng(12345)
    for _ in range(50):
        u = rng.standard_normal(n*m)
        v = rng.standard_normal(m*p)
        w = rng.standard_normal(p*n)
        # Normalize
        u = u / (np.linalg.norm(u) + 1e-10)
        v = v / (np.linalg.norm(v) + 1e-10)
        w = w / (np.linalg.norm(w) + 1e-10)
        candidates.append((u, v, w))

    # Convert to matrices
    K = len(candidates)
    U_pool = np.zeros((n*m, K))
    V_pool = np.zeros((m*p, K))
    W_pool = np.zeros((p*n, K))
    for idx, (u, v, w) in enumerate(candidates):
        U_pool[:, idx] = u
        V_pool[:, idx] = v
        W_pool[:, idx] = w

    return U_pool, V_pool, W_pool


def _sparse_recovery(U_pool, V_pool, W_pool, n, m, p, target_rank, max_iters=100):
    """
    Use iterative hard thresholding to find a sparse combination of pool vectors
    that reconstructs the ones tensor.
    """
    K = U_pool.shape[1]
    target = np.ones(n*m*p)
    
    # Precompute the tensor for each candidate
    # T_k = u_k * v_k * w_k (as flattened tensor)
    T = np.zeros((K, n*m*p))
    for k in range(K):
        u3 = U_pool[:, k].reshape(n, m)
        v3 = V_pool[:, k].reshape(m, p)
        w3 = W_pool[:, k].reshape(p, n)
        T[k] = np.einsum('ij,jk,ki->ijk', u3, v3, w3).flatten()
    
    # Iterative hard thresholding
    coeffs = np.zeros(K)
    for it in range(max_iters):
        # Compute residual
        recon = T.T @ coeffs
        residual = target - recon
        
        # Gradient step
        coeffs += T @ residual
        
        # Hard threshold: keep only top target_rank coefficients
        if target_rank < K:
            idx = np.argsort(np.abs(coeffs))[::-1][:target_rank]
            coeffs_new = np.zeros(K)
            coeffs_new[idx] = coeffs[idx]
            coeffs = coeffs_new
    
    # Check if we got a good solution
    recon = T.T @ coeffs
    if not np.allclose(recon, target, atol=1e-6):
        return None
    
    # Extract non-zero coefficients
    nz = np.nonzero(np.abs(coeffs) > 1e-8)[0]
    if len(nz) == 0:
        return None
    
    U_sol = U_pool[:, nz]
    V_sol = V_pool[:, nz]
    W_sol = W_pool[:, nz]
    
    # Scale columns by coefficients
    for i, idx in enumerate(nz):
        U_sol[:, i] *= coeffs[idx]
    
    return U_sol, V_sol, W_sol


def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=30):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies
    among the rank-1 terms and eliminating them.
    """
    start = time.time()
    deadline = start + max_time
    r = U.shape[1]
    
    if r <= target_rank:
        return U, V, W
    
    # Convert to numpy
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Build the tensor matrix: each column k gives a rank-1 tensor
    # We want to find a subset of columns that still spans the same tensor
    # and has size <= target_rank
    
    # Flatten each rank-1 tensor
    T = np.zeros((r, n*m*p))
    for k in range(r):
        u3 = U[:, k].reshape(n, m)
        v3 = V[:, k].reshape(m, p)
        w3 = W[:, k].reshape(p, n)
        T[k] = np.einsum('ij,jk,ki->ijk', u3, v3, w3).flatten()
    
    target = np.ones(n*m*p)
    
    # Check if the current decomposition is valid
    recon = T.T @ np.ones(r)
    if not np.allclose(recon, target, atol=1e-9):
        return None, None, None
    
    # Try to find a smaller subset
    # Use greedy: start with all, try to remove one at a time
    active = list(range(r))
    
    while len(active) > target_rank and time.time() < deadline:
        removed = False
        for idx_to_try in list(active):
            trial = [i for i in active if i != idx_to_try]
            if len(trial) < target_rank:
                continue
            
            # Check if the remaining terms can still reconstruct
            # Solve: find coefficients c such that sum c_k * T[k] = target
            T_sub = T[trial, :].T  # (n*m*p, len(trial))
            
            # Least squares
            try:
                c, _, _, _ = np.linalg.lstsq(T_sub, target, rcond=None)
            except:
                continue
            
            recon_sub = T_sub @ c
            if np.allclose(recon_sub, target, atol=1e-9):
                # Check if coefficients are "nice" (close to integers or simple rationals)
                # For now, just accept if it works
                active = trial
                removed = True
                break
        
        if not removed:
            break
    
    if len(active) <= target_rank:
        U_out = U[:, active]
        V_out = V[:, active]
        W_out = W[:, active]
        
        # Need to recompute coefficients
        T_sub = T[active, :].T
        c, _, _, _ = np.linalg.lstsq(T_sub, target, rcond=None)
        
        # Scale U columns by coefficients
        U_out = U_out * c[None, :]
        
        return U_out, V_out, W_out
    
    return None, None, None


def _build_3plus1_split():
    """
    Alternative split: treat all 3 rows together.
    Use 3x2 * 2x2 blocks for each column pair.
    3x2 * 2x2 has rank 12 (schoolbook) or potentially less.
    4 column pairs * 12 = 48 (worse than 44)
    
    But we can try: 3x4 * 4x4 as a whole with structured decomposition.
    """
    n, m, p = N, M, P
    # This is just schoolbook for now
    return schoolbook(n, m, p)


def _build_hybrid_decomposition():
    """
    Build a hybrid decomposition that mixes different strategies.
    Try to use Strassen where beneficial and schoolbook elsewhere.
    """
    n, m, p = N, M, P
    
    # Strategy 1: 2+1 row split (current best: rank 44)
    U1, V1, W1 = _build_2plus1_split()
    
    # Strategy 2: Try 1+2 row split
    rank_total = 4 * 7 + 4 * 4  # Same structure, different offsets
    U2 = [[0] * rank_total for _ in range(n * m)]
    V2 = [[0] * rank_total for _ in range(m * p)]
    W2 = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    # Row 0 alone
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U2, V2, W2, Us, Vs, Ws, 0, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
    
    # Rows 1-2 together
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U2, V2, W2, Us, Vs, Ws, 1, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
    
    # Verify both
    valid1 = _verify_einsum(U1, V1, W1, n, m, p)
    valid2 = _verify_einsum(U2, V2, W2, n, m, p)
    
    if valid1 and not valid2:
        return U1, V1, W1, 44
    elif valid2 and not valid1:
        return U2, V2, W2, 44
    elif valid1 and valid2:
        # Both valid, return the one with lower rank (same here)
        return U1, V1, W1, 44
    else:
        # Fallback to schoolbook
        Us, Vs, Ws = schoolbook(n, m, p)
        return Us, Vs, Ws, 48


def _try_greedy_sparse(U_pool, V_pool, W_pool, n, m, p, target_rank, max_time=20):
    """
    Greedy algorithm: iteratively add the best candidate that reduces residual.
    """
    start = time.time()
    deadline = start + max_time
    K = U_pool.shape[1]
    target = np.ones(n*m*p)
    
    # Precompute tensor for each candidate
    T = np.zeros((K, n*m*p))
    for k in range(K):
        u3 = U_pool[:, k].reshape(n, m)
        v3 = V_pool[:, k].reshape(m, p)
        w3 = W_pool[:, k].reshape(p, n)
        T[k] = np.einsum('ij,jk,ki->ijk', u3, v3, w3).flatten()
    
    residual = target.copy()
    selected = []
    
    for step in range(target_rank):
        if time.time() > deadline:
            break
        
        # Find candidate that best reduces residual
        # <T[k], residual> for each k
        scores = T @ residual
        best_k = np.argmax(np.abs(scores))
        
        if abs(scores[best_k]) < 1e-10:
            break
        
        # Add this candidate
        selected.append(best_k)
        
        # Update residual: project out the component along T[best_k]
        # In the span of selected candidates
        T_sel = T[selected, :].T  # (n*m*p, len(selected))
        
        # Solve least squares for coefficients
        try:
            coeffs, _, _, _ = np.linalg.lstsq(T_sel, target, rcond=None)
        except:
            coeffs = np.ones(len(selected))
        
        recon = T_sel @ coeffs
        residual = target - recon
    
    if len(selected) < target_rank:
        return None
    
    # Check final reconstruction
    T_sel = T[selected, :].T
    coeffs, _, _, _ = np.linalg.lstsq(T_sel, target, rcond=None)
    recon = T_sel @ coeffs
    
    if not np.allclose(recon, target, atol=1e-6):
        return None
    
    U_sol = U_pool[:, selected]
    V_sol = V_pool[:, selected]
    W_sol = W_pool[:, selected]
    U_sol = U_sol * coeffs[None, :]
    
    return U_sol, V_sol, W_sol


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Returns dict with 'U', 'V', 'W'.
    """
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (evaluator kills at 150s)
    
    # Start with the known rank-44 decomposition
    U, V, W, best_rank = _build_hybrid_decomposition()
    
    if not _verify_einsum(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48
        Us, Vs, Ws = schoolbook(n, m, p)
        return {"U": Us, "V": Vs, "W": Ws}
    
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank using various strategies
    
    # Strategy 1: Rank reduction by finding linear dependencies
    for target_rank in range(32, best_rank):
        if time.time() > deadline - 20:
            break
        
        U_r, V_r, W_r = _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=15)
        if U_r is not None and _verify_einsum(U_r, V_r, W_r, n, m, p):
            best_rank = target_rank
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    # Strategy 2: Build candidate pool and use sparse recovery
    if time.time() < deadline - 30:
        U_pool, V_pool, W_pool = _build_candidate_pool(n, m, p)
        
        for target_rank in range(32, best_rank + 1):
            if time.time() > deadline - 15:
                break
            
            # Try greedy sparse
            U_g, V_g, W_g = _try_greedy_sparse(U_pool, V_pool, W_pool, n, m, p, target_rank, max_time=10)
            if U_g is not None and _verify_einsum(U_g, V_g, W_g, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_g, V_g, W_g
                break
            
            # Try iterative hard thresholding
            if time.time() < deadline - 10:
                U_iht, V_iht, W_iht = _sparse_recovery(U_pool, V_pool, W_pool, n, m, p, target_rank, max_iters=50)
                if U_iht is not None and _verify_einsum(U_iht, V_iht, W_iht, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_iht, V_iht, W_iht
                    break
    
    # Final verification with strict tolerance
    if _verify_strict(best_U, best_V, best_W, n, m, p, tol=1e-12):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If strict verification fails, try to clean up the solution
    # Round to nearest rational with small denominator
    U_arr = np.asarray(best_U, dtype=np.float64)
    V_arr = np.asarray(best_V, dtype=np.float64)
    W_arr = np.asarray(best_W, dtype=np.float64)
    
    def _round_to_rational(arr, max_denom=12):
        out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val) < 1e-10:
                    out[i, j] = 0
                    continue
                best_num, best_den = 0, 1
                best_err = abs(val)
                for den in range(1, max_denom + 1):
                    num = round(val * den)
                    err = abs(val - num / den)
                    if err < best_err:
                        best_err = err
                        best_num = num
                        best_den = den
                out[i, j] = best_num / best_den
        return out
    
    U_clean = _round_to_rational(U_arr)
    V_clean = _round_to_rational(V_arr)
    W_clean = _round_to_rational(W_arr)
    
    if _verify_einsum(U_clean, V_clean, W_clean, n, m, p, tol=1e-9):
        return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
    
    # Fallback: return original if it passes basic verification
    if _verify_einsum(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Final fallback: schoolbook
    Us, Vs, Ws = schoolbook(n, m, p)
    return {"U": Us, "V": Vs, "W": Ws}
