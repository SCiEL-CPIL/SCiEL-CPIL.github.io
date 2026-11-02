import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 120.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    # T indexed by (i,k) output, (i,j) first, (j,k) second
    # We'll store as a 3D array T[i,k, i,j, j,k] -> but that's 5D.
    # Instead store as dict or as the "canonical" tensor of shape (n*p, n*m, m*p)
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i       # W convention: W[k*n+i, r]
                u_idx = i * m + j
                v_idx = j * p + k
                T[out_idx, u_idx, v_idx] = 1.0
    return T

def _flattening_lower_bound(T):
    # min over all matricizations of the matrix rank
    I, J, K = T.shape
    # matricize as I x (J*K)
    M1 = T.reshape(I, J * K)
    M2 = T.transpose(0, 2, 1).reshape(J, I * K)
    M3 = T.transpose(1, 0, 2).reshape(K, I * J)
    r1 = np.linalg.matrix_rank(M1)
    r2 = np.linalg.matrix_rank(M2)
    r3 = np.linalg.matrix_rank(M3)
    return max(r1, r2, r3)

def _als_decompose(T, r, n_iter=200, seed=0, tol=1e-10):
    """Alternating least squares for tensor decomposition T ~= sum_t U_t V_t W_t.
    Returns (U, V, W) each of shape (dim, r) or None if fails."""
    I, J, K = T.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((I, r))
    V = rng.standard_normal((J, r))
    W = rng.standard_normal((K, r))
    # normalize
    for t in range(r):
        nrm = max(1e-12, np.sqrt(np.sum(U[:, t]**2) + np.sum(V[:, t]**2) + np.sum(W[:, t]**2)))
        U[:, t] /= nrm
        V[:, t] /= nrm
        W[:, t] /= nrm
    best_err = np.inf
    for it in range(n_iter):
        # update U: for each i, U[i,:] solves sum_t U[i,t] V[:,t] W[:,t] = T[i,:,:]
        # i.e. (V ⊙ W) U[i,:]^T = T[i,:,:].reshape(-1)  where V⊙W is (J*K, r)
        VW = np.einsum('jt,kt->jkt', V, W).reshape(J * K, r)
        for i in range(I):
            rhs = T[i, :, :].reshape(-1)
            sol, *_ = np.linalg.lstsq(VW, rhs, rcond=None)
            U[i, :] = sol
        UW = np.einsum('it,kt->ikt', U, W).reshape(I * K, r)
        for j in range(J):
            rhs = T[:, j, :].reshape(-1)
            sol, *_ = np.linalg.lstsq(UW, rhs, rcond=None)
            V[j, :] = sol
        UV = np.einsum('it,jt->ijt', U, V).reshape(I * J, r)
        for k in range(K):
            rhs = T[:, :, k].reshape(-1)
            sol, *_ = np.linalg.lstsq(UV, rhs, rcond=None)
            W[k, :] = sol
        # compute error
        recon = np.einsum('it,jt,kt->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
        if err < tol:
            break
    if best_err < 1e-6:
        return U, V, W, best_err
    return None

def _try_rank(T, r, max_seeds=8, n_iter=150):
    for seed in range(max_seeds):
        res = _als_decompose(T, r, n_iter=n_iter, seed=seed)
        if res is not None:
            U, V, W, err = res
            return U, V, W
    return None

def _to_int_or_half(M):
    # Convert to integer/half-integer list of lists
    out = []
    for row in M:
        newrow = []
        for x in row:
            # round to nearest half
            v = round(2 * x) / 2.0
            # snap to int if close
            if abs(v - round(v)) < 1e-6:
                v = float(round(v))
            newrow.append(v)
        out.append(newrow)
    return out

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    lb = _flattening_lower_bound(T)
    # lower bound is at least max flattening rank; also at least max(I,J,K)
    lb = max(lb, max(T.shape))
    best = None
    # try from lb upward, but cap total time
    for r in range(int(lb), n * m * p + 1):
        if time.time() - start > DEADLINE - 10:
            break
        # Reduce iterations for lower ranks to try more ranks if needed?
        # Or just keep it simple.
        res = _try_rank(T, r, max_seeds=4, n_iter=80)
        if res is not None:
            U, V, W = res
            # verify exactly
            recon = np.einsum('it,jt,kt->ijk', U, V, W)
            if np.allclose(recon, T, atol=1e-5):
                best = (r, U, V, W)
                break
    if best is None:
        # fall back to schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best = (r, U, V, W)
    r, U, V, W = best
    return {"U": _to_int_or_half(U), "V": _to_int_or_half(V), "W": _to_int_or_half(W)}
