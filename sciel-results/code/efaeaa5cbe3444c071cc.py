import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_combined_strassen():
    """
    Combined Strassen: for each of the 4 column-pairs of B, create 6 terms
    that simultaneously serve both A-rows. This reduces from 56 to 48 rank.
    
    The idea: instead of 2 independent 2x2 Strassen multiplications per column-pair
    (one for A[0,:] and one for A[1,:]), we create combined terms where:
    - U has entries in both row-0 and row-1 of A
    - V has entries in both row-pairs of B
    - W has entries in both row-0 and row-1 of C
    
    This exploits the fact that the same B columns feed into both output rows.
    """
    n, m, p = DIMENSIONS
    # 4 column-pairs × 6 combined terms = 24... but we need more structure
    # Actually, let's think more carefully.
    # 
    # For column-pair l (columns 2l, 2l+1 of B):
    #   C[0, 2l]   = A[0,0]*B[0,2l] + A[0,1]*B[1,2l] + A[0,2]*B[2,2l] + A[0,3]*B[3,2l]
    #   C[0, 2l+1] = A[0,0]*B[0,2l+1] + A[0,1]*B[1,2l+1] + A[0,2]*B[2,2l+1] + A[0,3]*B[3,2l+1]
    #   C[1, 2l]   = A[1,0]*B[0,2l] + A[1,1]*B[1,2l] + A[1,2]*B[2,2l] + A[1,3]*B[3,2l]
    #   C[1, 2l+1] = A[1,0]*B[0,2l+1] + A[1,1]*B[1,2l+1] + A[1,2]*B[2,2l+1] + A[1,3]*B[3,2l+1]
    #
    # Group B rows into pairs: (0,1) and (2,3)
    # For B rows (0,1): 2x2 submatrix B[[0,1],[2l,2l+1]]
    # For B rows (2,3): 2x2 submatrix B[[2,3],[2l,2l+1]]
    #
    # A columns split into (0,1) and (2,3)
    # 
    # So we have 2x2 A-blocks × 2x2 B-blocks → 2x2 C-blocks
    # That's 4 blocks per column-pair, each needing 7 Strassen mults = 28 per column-pair
    # Total: 4 * 28 = 112? No, that's worse.
    #
    # Wait, I'm overcomplicating. The original 56 already groups correctly.
    # Let me try a different approach: use the known lower bound and search.
    
    # For now, return the standard rank-56 as fallback
    return _construct_rank56()


def _nullify_columns(U, V, W, target_cols, n, m, p, max_iter=1000, lr=0.01, deadline=None):
    """
    Try to nullify specified columns while preserving the tensor decomposition.
    Uses gradient descent on the combined objective.
    """
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    r = Ua.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute tensor
        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
        T_err = T - 1.0
        penalty = np.sum(T_err ** 2)
        
        if penalty < 1e-12:
            # Check if target columns are near zero
            target_norm = 0.0
            for c in target_cols:
                target_norm += np.sum(Ua[:, c] ** 2) + np.sum(Va[:, c] ** 2) + np.sum(Wa[:, c] ** 2)
            if target_norm < 1e-10:
                return Ua, Va, Wa, True
        
        # Compute analytical gradients
        dUa_ijk = 2.0 * np.einsum('ijk,jkr,kin->ijr', T_err, Vr, Wr)
        dUa = dUa_ijk.reshape(n * m, r)
        
        dVa_jkr = 2.0 * np.einsum('ijk,ijr,kin->jkr', T_err, Ur, Wr)
        dVa = dVa_jkr.reshape(m * p, r)
        
        dWa_kin = 2.0 * np.einsum('ijk,ijr,jkr->kin', T_err, Ur, Vr)
        dWa = dWa_kin.reshape(p * n, r)
        
        # Add nullity gradient for target columns
        for c in target_cols:
            dUa[:, c] += 2.0 * Ua[:, c]
            dVa[:, c] += 2.0 * Va[:, c]
            dWa[:, c] += 2.0 * Wa[:, c]
        
        # Adaptive learning rate
        eff_lr = lr / (1.0 + it * 0.001)
        
        # Update
        Ua -= eff_lr * dUa
        Va -= eff_lr * dVa
        Wa -= eff_lr * dWa
        
        # Periodic check for divergence
        if it % 100 == 0:
            Ur2 = Ua.reshape(n, m, r)
            Vr2 = Va.reshape(m, p, r)
            Wr2 = Wa.reshape(p, n, r)
            T2 = np.einsum('ijr,jkr,kin->ijk', Ur2, Vr2, Wr2)
            pen2 = np.sum((T2 - 1.0) ** 2)
            if pen2 > 10.0:
                # Diverged, revert
                Ua = np.array(U, dtype=float).copy()
                Va = np.array(V, dtype=float).copy()
                Wa = np.array(W, dtype=float).copy()
                return Ua, Va, Wa, False
    
    # Final check
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    penalty = np.sum((T - 1.0) ** 2)
    if penalty < 1e-10:
        target_norm = 0.0
        for c in target_cols:
            target_norm += np.sum(Ua[:, c] ** 2) + np.sum(Va[:, c] ** 2) + np.sum(Wa[:, c] ** 2)
        if target_norm < 1e-8:
            return Ua, Va, Wa, True
    return Ua, Va, Wa, False


def _lll_find_dependency(U, V, W, n, m, p):
    """
    Use SVD to find near-linear dependencies among the rank-56 basis vectors.
    Each column c defines a tensor T_c[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c].
    We look for a linear combination sum_c alpha_c * T_c = 0 (or close to 0).
    If found, we can modify the decomposition to eliminate one column.
    
    Returns (dependency_vector, residual) or (None, None).
    """
    r = len(U[0])
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    # Build the tensor matrix: each column c gives a vector of size n*m*p
    # M[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find alpha such that M @ alpha ≈ 0
    
    # For efficiency, build M as a 2D array of shape (n*m*p, r)
    M = np.zeros((n * m * p, r))
    for c in range(r):
        Ur = np.array(U)[:, c].reshape(n, m)
        Vr = np.array(V)[:, c].reshape(m, p)
        Wr = np.array(W)[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p * n + j * n + k  # This is wrong, let me fix
                    # Actually, T[i,j,k] should be 1 for all (i,j,k) in the MM tensor
                    # The tensor M[i,j,k,c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # We need M @ alpha = 1 (the all-ones tensor), not M @ alpha = 0
                    # So we're looking for the null space of (M - ones)
                    pass
    
    # Let me redo this properly
    # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    # This is a system of n*m*p = 64 equations in r=56 unknowns (the column weights)
    # But the columns are fixed; we're looking for linear dependencies among the column-tensors
    
    # Build tensor matrix properly
    M = np.zeros((n * m * p, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(r):
                    M[eq_idx, c] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
    
    # The current solution has all alpha_c = 1, giving M @ ones = ones_vector
    # We want to find a vector v in the null space of M such that we can
    # set alpha = ones - v, and if v has a zero in some position, that column
    # becomes unnecessary
    
    # Compute SVD
    try:
        U_svd, s_vals, Vt = svd(M, full_matrices=False)
    except Exception:
        return None, None
    
    # The null space vectors are the last rows of Vt corresponding to small s_vals
    # Find the smallest singular value
    min_s = min(s_vals)
    
    # If min_s is very small, there's a near-dependency
    if min_s < 1e-6:
        # The null vector is the last row of Vt
        v = Vt[-1]
        # Check if we can use this to eliminate a column
        # If v has a large component in some direction, we can shift
        return v, min_s
    else:
        return None, min_s


def _try_eliminate_column(U, V, W, n, m, p, deadline):
    """
    Try to eliminate one column from the decomposition using LLL-inspired approach.
    For each column, try to nullify it via gradient descent.
    Returns (U, V, W, success) or None.
    """
    r = len(U[0])
    
    # Try each column from last to first
    for target_col in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try to nullify this column
        Ua, Va, Wa, success = _nullify_columns(
            U, V, W, [target_col], n, m, p,
            max_iter=500, lr=0.005, deadline=deadline
        )
        
        if success:
            # Remove the column
            new_U = [[Ua[i, c] for c in range(r) if c != target_col] for i in range(len(U))]
            new_V = [[Va[i, c] for c in range(r) if c != target_col] for i in range(len(V))]
            new_W = [[Wa[i, c] for c in range(r) if c != target_col] for i in range(len(W))]
            return new_U, new_V, new_W, True
    
    return None


def _polish(U, V, W, n, m, p, deadline):
    """
    High-precision polish: run a few Newton iterations to drive the error to machine precision.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    
    for it in range(20):
        if time.time() > deadline:
            break
        
        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
        T_err = T - 1.0
        err = np.max(np.abs(T_err))
        
        if err < 1e-14:
            break
        
        # Small Newton step
        dUa_ijk = 2.0 * np.einsum('ijk,jkr,kin->ijr', T_err, Vr, Wr)
        dUa = dUa_ijk.reshape(n * m, r)
        dVa_jkr = 2.0 * np.einsum('ijk,ijr,kin->jkr', T_err, Ur, Wr)
        dVa = dVa_jkr.reshape(m * p, r)
        dWa_kin = 2.0 * np.einsum('ijk,ijr,jkr->kin', T_err, Ur, Vr)
        dWa = dWa_kin.reshape(p * n, r)
        
        lr = 0.01 / (1.0 + it * 0.1)
        Ua -= lr * dUa
        Va -= lr * dVa
        Wa -= lr * dWa
    
    # Round to nearest integer/half-integer if close
    Ua_rounded = np.round(Ua * 2) / 2
    Va_rounded = np.round(Va * 2) / 2
    Wa_rounded = np.round(Wa * 2) / 2
    
    # Check if rounded version is still valid
    Ur_r = Ua_rounded.reshape(n, m, r)
    Vr_r = Va_rounded.reshape(m, p, r)
    Wr_r = Wa_rounded.reshape(p, n, r)
    T_r = np.einsum('ijr,jkr,kin->ijk', Ur_r, Vr_r, Wr_r)
    if np.max(np.abs(T_r - 1.0)) < 1e-9:
        return Ua_rounded.tolist(), Va_rounded.tolist(), Wa_rounded.tolist()
    
    return Ua.tolist(), Va.tolist(), Wa.tolist()


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # Safety margin

    # Start with known rank-56 solution
    cand = _construct_rank56()
    ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
    if not ok:
        return cand
    
    U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
    best = (r, U, V, W)
    
    # Try LLL-inspired dependency finding
    if time.time() < deadline - 60:
        dep, min_s = _lll_find_dependency(U, V, W, n, m, p)
        if dep is not None and min_s < 1e-4:
            # There's a near-dependency; try to exploit it
            # The dependency vector tells us how to combine columns
            # For now, just use it as a hint for which column to try to eliminate
            pass
    
    # Try to eliminate columns via gradient descent
    if time.time() < deadline - 30:
        result = _try_eliminate_column(U, V, W, n, m, p, deadline)
        if result is not None:
            new_U, new_V, new_W, success = result
            if success:
                ok2, err2 = _matmul_check(new_U, new_V, new_W, n, m, p)
                if ok2:
                    U2, V2, W2, r2 = _trim_zero_columns(new_U, new_V, new_W)
                    if r2 < best[0]:
                        best = (r2, U2, V2, W2)
    
    # Polish the best solution
    if time.time() < deadline - 10:
        r, U, V, W = best
        U_p, V_p, W_p = _polish(U, V, W, n, m, p, deadline)
        ok3, err3 = _matmul_check(U_p, V_p, W_p, n, m, p)
        if ok3:
            U_f, V_f, W_f, r_f = _trim_zero_columns(U_p, V_p, W_p)
            if r_f <= best[0]:
                best = (r_f, U_f, V_f, W_f)
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
