"""Rank-17 Strassen-based decomposition for 2x3 * 3x3 matmul,
with a probabilistic algebraic search for a rank-16 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    """Full matmul tensor T[i,j,k]=1 iff (i//M)==(k%N) and (i%M)==(j//P)."""
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            ja2, kb = divmod(j, P)
            if ja != ja2:
                continue
            for k in range(W_DIM):
                kb2, ic = divmod(k, N)
                if ia == ic and kb == kb2:
                    T[i, j, k] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    """Valid rank-17 decomposition: Strassen(2x2x2) for (j in {0,1}, k in {0,1})
    plus schoolbook for the remaining 10 terms."""
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    # Strassen columns 0..6
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    # Schoolbook columns 7..16
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=None):
    """Alternating least squares. Returns (U, V, W, converged)."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        # Solve for U
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        # Solve for V
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        # Solve for W
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
    return U, V, W, True


def _solve_sparse_support(support_U, support_V, support_W, T, deadline=None):
    """
    Given sparse support patterns (sets of non-zero indices) for U, V, W,
    solve the resulting linear system for the non-zero coefficients.
    
    support_U: list of (row, col) pairs for non-zero entries in U
    support_V: list of (row, col) pairs for non-zero entries in V
    support_W: list of (row, col) pairs for non-zero entries in W
    
    Returns (U, V, W) or None if the system is inconsistent.
    """
    n_u = len(support_U)
    n_v = len(support_V)
    n_w = len(support_W)
    n_vars = n_u + n_v + n_w
    
    if n_vars == 0:
        return None
    
    # Build the linear system: for each tensor entry T[i,j,k],
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    # This is nonlinear, but if we fix supports, we can use a different approach.
    # Actually, let's use the fact that we can iterate: fix two, solve for third.
    
    # Better approach: use ALS with fixed supports
    U = np.zeros((U_DIM, 16))
    V = np.zeros((V_DIM, 16))
    W = np.zeros((W_DIM, 16))
    
    # Set the supports
    for (i, r) in support_U:
        U[i, r] = 1.0  # placeholder, will be solved
    for (j, r) in support_V:
        V[j, r] = 1.0
    for (k, r) in support_W:
        W[k, r] = 1.0
    
    # Now use ALS to refine the values while respecting the support structure
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        # Solve for U entries
        for (i, r) in support_U:
            # U[i,r] = sum_{j,k} T[i,j,k]*V[j,r]*W[k,r] / sum_{j,k} V[j,r]^2*W[k,r]^2
            num = np.sum(T[i, :, :] * V[:, r, None] * W[None, :, r])
            den = np.sum((V[:, r] ** 2) * (W[:, r] ** 2))
            if abs(den) > 1e-12:
                U[i, r] = num / den
            else:
                U[i, r] = 0.0
        
        # Solve for V entries
        for (j, r) in support_V:
            num = np.sum(T[:, j, :] * U[:, r, None] * W[None, :, r])
            den = np.sum((U[:, r] ** 2) * (W[:, r] ** 2))
            if abs(den) > 1e-12:
                V[j, r] = num / den
            else:
                V[j, r] = 0.0
        
        # Solve for W entries
        for (k, r) in support_W:
            num = np.sum(T[:, :, k] * U[:, r, None] * V[None, :, r])
            den = np.sum((U[:, r] ** 2) * (V[:, r] ** 2))
            if abs(den) > 1e-12:
                W[k, r] = num / den
            else:
                W[k, r] = 0.0
    
    return U, V, W


def _probabilistic_search(T, deadline, n_trials=50):
    """
    Probabilistic algebraic search: randomly sample sparse support patterns
    and solve for coefficients.
    """
    U17, V17, W17 = _rank17()
    best_result = None
    best_err = float('inf')
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + trial * 31)
        
        # Start from the rank-17 structure and try to merge or drop columns
        # Strategy: take 16 of the 17 columns, then randomly perturb supports
        
        # Choose which column to drop from rank-17
        drop_col = rng.randint(17)
        
        # Build initial support from remaining 16 columns
        support_U = []
        support_V = []
        support_W = []
        
        for c in range(17):
            if c == drop_col:
                continue
            new_col = len(support_U) // max(1, len([s for s in support_U if s[1] == -1]))  # track col index
            
            # Find actual column index in the new 16-col scheme
            col_idx = c if c < drop_col else c - 1
            
            for i in range(U_DIM):
                if abs(U17[i, c]) > 1e-10:
                    support_U.append((i, col_idx))
            for j in range(V_DIM):
                if abs(V17[j, c]) > 1e-10:
                    support_V.append((j, col_idx))
            for k in range(W_DIM):
                if abs(W17[k, c]) > 1e-10:
                    support_W.append((k, col_idx))
        
        # Randomly add a few extra support entries to allow flexibility
        n_extra = rng.randint(0, 5)
        for _ in range(n_extra):
            col_idx = rng.randint(16)
            i = rng.randint(U_DIM)
            if (i, col_idx) not in support_U:
                support_U.append((i, col_idx))
            j = rng.randint(V_DIM)
            if (j, col_idx) not in support_V:
                support_V.append((j, col_idx))
            k = rng.randint(W_DIM)
            if (k, col_idx) not in support_W:
                support_W.append((k, col_idx))
        
        # Solve
        U, V, W = _solve_sparse_support(support_U, support_V, support_W, T, deadline)
        
        if U is not None:
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(recon - T)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                break
    
    return best_result


def _try_rank16(deadline):
    """Bounded search for a rank-16 decomposition.
    Returns (U,V,W) or None."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

    # Strategy 1: Start from rank-17, drop one column, refine with ALS
    U17, V17, W17 = _rank17()
    for drop_col in range(17):
        if time.time() > deadline:
            break
        if drop_col < 16:
            U = np.hstack([U17[:, :drop_col], U17[:, drop_col+1:]]).copy()
            V = np.hstack([V17[:, :drop_col], V17[:, drop_col+1:]]).copy()
            W = np.hstack([W17[:, :drop_col], W17[:, drop_col+1:]]).copy()
        else:
            U = U17[:, :16].copy()
            V = V17[:, :16].copy()
            W = W17[:, :16].copy()
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=80, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 2: Probabilistic algebraic search
    result = _probabilistic_search(T, deadline, n_trials=40)
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    return None


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

    # Baseline: valid rank-17 (verified).
    U17, V17, W17 = _rank17()
    T = _build_tensor()
    base = (U17, V17, W17)
    if not _validate(U17, V17, W17, T):
        # Fallback to guaranteed-correct schoolbook rank-18.
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        base = (U, V, W)

    # Attempt rank-16 (improvement over 17) within the time budget.
    try:
        r16 = _try_rank16(deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
