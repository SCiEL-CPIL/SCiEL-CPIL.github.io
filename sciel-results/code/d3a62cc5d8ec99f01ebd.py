"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition.

Strategy:
  1. Build a valid schoolbook decomposition (rank 96).
  2. Refine with a time-budgeted, rank-preserving local search (gradient
     descent on the Frobenius residual) plus a provably-valid column
     elimination pass that can only reduce rank while keeping validity.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


# EVOLVE-BLOCK-START

def _build_schoolbook():
    """Return (U,V,W) for 3x4x8 matrix multiplication, rank 96.
    U: (12,96), V: (32,96), W: (24,96). Verified valid."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _residual(U, V, W):
    """Frobenius norm of the tensor residual."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for c in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    # target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] -= 1.0
    return np.linalg.norm(T)


def _try_eliminate_columns(U, V, W, tol=1e-9):
    """Remove columns that do not change the residual (provably valid).
    Returns (U, V, W, removed_count)."""
    base_res = _residual(U, V, W)
    removed = 0
    keep = np.ones(U.shape[1], dtype=bool)
    for c in range(U.shape[1]):
        if not keep[c]:
            continue
        # try removing column c
        mask = np.ones(U.shape[1], dtype=bool)
        mask[c] = False
        U2 = U[:, mask]
        V2 = V[:, mask]
        W2 = W[:, mask]
        r2 = _residual(U2, V2, W2)
        if abs(r2 - base_res) < tol:
            keep[c] = False
            removed += 1
            U = U2
            V = V2
            W = W2
            base_res = r2
    return U, V, W, removed


def _local_search(U, V, W, deadline, max_iter=200, lr=0.05):
    """Rank-preserving local search: gradient descent on the residual.
    Only accepts updates that do not increase the residual."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    best_res = _residual(U, V, W)
    best = (U.copy(), V.copy(), W.copy())
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # compute gradient of 0.5*||T||^2 w.r.t. U, V, W
        # T = sum_c outer(U_c, V_c, W_c)
        # dT/du_a = sum_c v_c w_c (broadcast)
        # grad_U[a] = sum_c (T contracted with v_c w_c)
        # Use einsum for efficiency
        T = np.zeros((n * m, m * p, p * n))
        for c in range(R):
            T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        # subtract target
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] -= 1.0
        # grad_U[a] = sum_c V_c[b] W_c[c] T[a,b,c] = einsum('abc,bc->a', T, VW)
        # but we need per-c: grad_U[a] = sum_c T[a, :, :] * V_c * W_c
        # = einsum('abc,bc->a', T, VW) where VW is not separable per c
        # Actually: grad_U[a] = sum_c sum_bc T[a,b,c] V_c[b] W_c[c]
        # Let's compute per c for correctness (R is small)
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for c in range(R):
            # contribution of column c to gradient
            # grad_U[:,c] = T * V_c * W_c contracted over (b,c_idx)
            grad_U[:, c] = np.einsum('abc,b,c->a', T, V[:, c], W[:, c])
            grad_V[:, c] = np.einsum('abc,a,c->b', T, U[:, c], W[:, c])
            grad_W[:, c] = np.einsum('abc,a,b->c', T, U[:, c], V[:, c])
        # update
        U_new = U - lr * grad_U
        V_new = V - lr * grad_V
        W_new = W - lr * grad_W
        r_new = _residual(U_new, V_new, W_new)
        if r_new < best_res - 1e-12:
            U, V, W = U_new, V_new, W_new
            best_res = r_new
            best = (U.copy(), V.copy(), W.copy())
            # try to eliminate columns after improvement
            U, V, W, rem = _try_eliminate_columns(U, V, W)
            if rem > 0:
                best = (U.copy(), V.copy(), W.copy())
                best_res = _residual(U, V, W)
        else:
            # reduce learning rate
            lr *= 0.8
            if lr < 1e-6:
                break
    return best[0], best[1], best[2]


def _to_halfint(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2


def construct():
    deadline = time.time() + 140  # 140s budget, 10s safety margin
    n, m, p = DIMENSIONS

    # Step 1: build schoolbook decomposition
    U, V, W = _build_schoolbook()
    # round to half-integers (should already be integers)
    U = _to_halfint(U)
    V = _to_halfint(V)
    W = _to_halfint(W)

    # Verify
    res = _residual(U, V, W)
    if res > 1e-6:
        # This should not happen, but if it does, return the raw schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Step 2: try to eliminate columns
    U, V, W, removed = _try_eliminate_columns(U, V, W)
    res = _residual(U, V, W)

    # Step 3: local search if time permits
    if time.time() < deadline - 20:
        U, V, W = _local_search(U, V, W, deadline)
        # round to half-integers after local search
        U = _to_halfint(U)
        V = _to_halfint(V)
        W = _to_halfint(W)
        # verify and try elimination again
        res = _residual(U, V, W)
        if res < 1e-6:
            U, V, W, removed2 = _try_eliminate_columns(U, V, W)

    # Final safety check
    res = _residual(U, V, W)
    if res > 1e-6:
        # fallback to the original schoolbook
        U, V, W = _build_schoolbook()
        U = _to_halfint(U)
        V = _to_halfint(V)
        W = _to_halfint(W)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
