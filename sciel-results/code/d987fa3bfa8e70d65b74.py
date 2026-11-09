import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 150.0  # evaluator kills at 150s; use 140s safety margin


def _make_schoolbook(n, m, p):
    """Return (U, V, W) as numpy float arrays for schoolbook M_{n,m,p}.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _decompose(U, V, W):
    """Build dense 3-tensor T from factor matrices via einsum."""
    n_m, r = U.shape
    m_p = V.shape[0]
    p_n = W.shape[0]
    # T[i_m, j_p, k_n] = sum_c U[i_m,c] * V[j_p,c] * W[k_n,c]
    return np.einsum('ir,jr,kr->ijk', U, V, W)


def _prune(U, V, W, tol=1e-7):
    """Remove zero columns and return new (U,V,W) and new rank."""
    # A column c is zero iff U[:,c] == 0 OR V[:,c] == 0 OR W[:,c] == 0
    nz = (np.abs(U) > tol).any(axis=0) & (np.abs(V) > tol).any(axis=0) & (np.abs(W) > tol).any(axis=0)
    if nz.all():
        return U, V, W, U.shape[1]
    return U[:, nz], V[:, nz], W[:, nz], int(nz.sum())


def _try_zero(U, V, W, r, deadline, max_trials=200):
    """Try to reduce rank by 1 by zeroing an entry of a factor.
    Returns (new_U, new_V, new_W, new_rank) on success, or None.
    """
    n_m, r_cur = U.shape
    m_p, _ = V.shape
    p_n, _ = W.shape

    # Pick random entries to try zeroing
    rng = np.random.default_rng()
    for trial in range(max_trials):
        if time.time() > deadline:
            return None

        # Randomly pick which factor and which entry
        factor_idx = rng.integers(0, 3)
        if factor_idx == 0:
            row = rng.integers(0, n_m)
            col = rng.integers(0, r_cur)
            if abs(U[row, col]) < 1e-12:
                continue
            # Try to zero U[row, col]
            # delta_c = -U[row,col] * V[:,col] * W[:,col]
            # We need: U' = U - u*delta, V' = V - v*delta, W' = W - w*delta
            # where u[row]=1, v[row]=1, w[row]=1 and delta[c] = -U[row,col] * V[:,col] * W[:,col] / (V[row,col]*W[row,col])
            # Simplify: we want U'[row,col] = 0
            # U'[row,col] = U[row,col] - u[row]*delta[col] = 0 => u[row]*delta[col] = U[row,col]
            # Set u[row]=1, then delta[col] = U[row,col]
            # For other c != col: U'[row,c] = U[row,c] - delta[c] (u[row]=1)
            # But we only care that the TENSOR is preserved, not that U' has specific form.
            # Actually the constraint is: U' * V' * W' == U * V * W (as tensors)
            # This is a nonlinear constraint. Let's use the standard approach:
            # We want to find u, v, w, delta such that:
            #   (U - u delta^T)(V - v delta^T)(W - w delta^T) = U V^T W^T (tensor)
            # This is hard. Instead, use the simpler "one entry zeroing" method:
            # Pick a nonzero entry, say U[i,j] != 0.
            # We want to modify the j-th column to zero out U[i,j].
            # Let U' = U, V' = V, W' = W except we adjust column j.
            # The j-th term contributes U[:,j] * V[:,j] * W[:,j] to T.
            # If we change V[:,j] -> V[:,j] + a, then the change in T is U[:,j] * a * W[:,j].
            # We want to cancel U[i,j] in some way... this doesn't directly zero an entry.

            # Let me use the proper tensor rank reduction algorithm.
            # The key insight: if we can find u, v, w, delta (rank-1 updates) such that
            # (U + u delta^T)(V + v delta^T)(W + w delta^T) = U V W (as tensor)
            # and one of the new columns becomes zero, we can prune.
            #
            # Actually, the standard approach for "zeroing one entry":
            # Given T = sum_r U_r V_r W_r (rank r), pick a nonzero entry, say the (i,j) entry of factor A (U).
            # We want to find a rank-(r) decomposition where that entry is zero.
            # This is possible if there exists a vector combination...
            #
            # Let me try a different, more direct approach:
            # Randomly perturb and check if rank decreases after SVD-like pruning.

            # Actually, let me implement the Anisole algorithm properly.
            # For zeroing U[i,j]:
            # We need vectors u, v, w and scalars such that the new decomposition
            # has U'[i,j] = 0 and T is preserved.
            # The condition is that the (i,j) "slice" can be absorbed.

            pass  # skip for now, use random perturbation approach

        elif factor_idx == 1:
            row = rng.integers(0, m_p)
            col = rng.integers(0, r_cur)
            if abs(V[row, col]) < 1e-12:
                continue
            pass
        else:
            row = rng.integers(0, p_n)
            col = rng.integers(0, r_cur)
            if abs(W[row, col]) < 1e-12:
                continue
            pass

    return None


def _random_perturb_prune(U, V, W, deadline, num_rounds=50):
    """Try random perturbations of the decomposition followed by pruning."""
    best_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())
    rng = np.random.default_rng()

    for rnd in range(num_rounds):
        if time.time() > deadline:
            break

        # Randomly modify a few entries in one factor
        factor_idx = rng.integers(0, 3)
        U_p, V_p, W_p = U.copy(), V.copy(), W.copy()
        if factor_idx == 0:
            for _ in range(rng.integers(1, 4)):
                r = rng.integers(0, n_m)
                c = rng.integers(0, best_rank)
                U_p[r, c] += rng.normal(0, 0.1)
        elif factor_idx == 1:
            for _ in range(rng.integers(1, 4)):
                r = rng.integers(0, m_p)
                c = rng.integers(0, best_rank)
                V_p[r, c] += rng.normal(0, 0.1)
        else:
            for _ in range(rng.integers(1, 4)):
                r = rng.integers(0, p_n)
                c = rng.integers(0, best_rank)
                W_p[r, c] += rng.normal(0, 0.1)

        # Verify tensor is approximately preserved
        T_orig = _decompose(U, V, W)
        T_new = _decompose(U_p, V_p, W_p)
        if np.max(np.abs(T_new - T_orig)) > 1e-6:
            continue

        # Try to prune
        U_pruned, V_pruned, W_pruned, new_rank = _prune(U_p, V_p, W_p, tol=1e-6)
        if new_rank < best_rank:
            best_rank = new_rank
            best = (U_pruned, V_pruned, W_pruned)

    return best[0], best[1], best[2], best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Verify schoolbook is correct
    T = _decompose(U, V, W)
    # Check: T should have T[i*m+j, j'*p+k, k'*n+i] = 1 when i==i', j==j', k==k'
    # Actually let's just verify the matrix multiplication property
    # For random A (n x m) and B (m x p), C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T should satisfy: sum_{j_p, i_m, k_n} T[...] ...
    # Let's trust the schoolbook and focus on rank reduction.

    # Strategy 1: Direct rank-1 zeroing attempts
    # For each nonzero entry in each factor, try to zero it via a rank-1 correction
    # that preserves the tensor.

    # The proper method: Given T = sum_r U_r V_r W_r, to zero out U[i,j]:
    # We need to find u, v, w (vectors of lengths n*m, m*p, p*n) and a scalar delta
    # such that the new decomposition has U'[i,j] = 0 and T is unchanged.
    #
    # One approach: if there exist indices i', j' such that U[i',j'] * V[i',j] * W[i',j] != 0
    # and the "cross terms" can be arranged... this is complex.
    #
    # Let me use a simpler heuristic: random linear combinations of columns.
    # If we replace column c with a linear combination of other columns,
    # the tensor might change, but if the combination is in the span, it's preserved.
    # Actually, any invertible linear transformation of the factor matrices preserves T:
    # T = U V W = (U S)(S^{-1} V) W for any invertible S.
    # So we can apply random invertible S and then prune!

    rng = np.random.default_rng(42)
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = best_rank

    for iteration in range(200):
        if time.time() > deadline:
            break

        # Apply a random invertible transformation to the factor columns
        # U' = U S, V' = S^{-1} V, W' = W  (for some random invertible S)
        # Or: U' = U S, V' = V S^{-1}, W' = W
        S = np.eye(current_rank)
        # Add small random perturbations to make S non-identity
        for _ in range(rng.integers(1, 5)):
            r1 = rng.integers(0, current_rank)
            r2 = rng.integers(0, current_rank)
            val = rng.normal(0, 0.5)
            S[r1, r2] += val

        # Check S is invertible
        det = np.linalg.det(S)
        if abs(det) < 1e-6:
            continue

        S_inv = np.linalg.inv(S)

        new_U = current_U @ S
        new_V = current_V @ S_inv
        new_W = current_W  # unchanged

        # Verify tensor preservation
        T_check = _decompose(new_U, new_V, new_W)
        T_ref = _decompose(current_U, current_V, current_W)
        if np.max(np.abs(T_check - T_ref)) > 1e-5:
            continue

        # Try to prune
        pruned_U, pruned_V, pruned_W, new_rank = _prune(new_U, new_V, new_W, tol=1e-6)
        if new_rank < current_rank:
            current_rank = new_rank
            current_U, current_V, current_W = pruned_U, pruned_V, pruned_W
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

        # Also try the other direction: transform W
        S2 = np.eye(current_rank)
        for _ in range(rng.integers(1, 5)):
            r1 = rng.integers(0, current_rank)
            r2 = rng.integers(0, current_rank)
            S2[r1, r2] += rng.normal(0, 0.5)
        det2 = np.linalg.det(S2)
        if abs(det2) < 1e-6:
            continue
        S2_inv = np.linalg.inv(S2)
        new_U2 = current_U @ S2
        new_V2 = current_V
        new_W2 = current_W @ S2_inv
        T_check2 = _decompose(new_U2, new_V2, new_W2)
        if np.max(np.abs(T_check2 - T_ref)) > 1e-5:
            continue
        pruned_U2, pruned_V2, pruned_W2, new_rank2 = _prune(new_U2, new_V2, new_W2, tol=1e-6)
        if new_rank2 < current_rank:
            current_rank = new_rank2
            current_U, current_V, current_W = pruned_U2, pruned_V2, pruned_W2
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Try random perturbations + pruning as well
    U3, V3, W3, r3 = _random_perturb_prune(best_U, best_V, best_W, deadline, num_rounds=20)
    if r3 < best_rank:
        best_rank = r3
        best_U, best_V, best_W = U3, V3, W3

    # Final verification
    T_final = _decompose(best_U, best_V, best_W)
    T_school = _decompose(U, V, W)
    if np.max(np.abs(T_final - T_school)) > 1e-4:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
