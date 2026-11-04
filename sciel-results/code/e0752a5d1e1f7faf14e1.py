"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _rank17():
    """Return the known, verified rank-17 decomposition (Strassen + direct)."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))

    terms = [
        # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _schoolbook():
    """Final fallback: exact schoolbook decomposition (rank n*m*p)."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return U_fb, V_fb, W_fb

def _try_rank16(deadline):
    """
    Search for a rank-16 decomposition.

    Key idea: a rank-16 tensor lives in the set of tensors whose
    16 component vectors (U[:,r], V[:,r], W[:,r]) satisfy the tensor
    equation. We warm-start from the known rank-17 by dropping one
    column, then refine using a column-wise least-squares / alternating
    projection scheme with multiple restarts so we can escape local
    minima. We accept the result only if it matches T exactly.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    Tflat = T.reshape(n_a * n_b * n_c)

    def error_of(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((res.reshape(-1) - Tflat) ** 2)

    def refine(U, V, W, max_sweeps, deadline):
        # Alternating least-squares over the three factor matrices.
        # For fixed V,W, the best U column r is the projection of the
        # residual onto the span of (V[:,r] * W[:,r]) outer-structured.
        # We solve each column r by least squares on the flattened tensor.
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            # --- update U ---
            for r in range(rank):
                v = V[:, r]; w = W[:, r]
                # residual tensor R = T - sum_{r' != r} U[:,r'] v_r' w_r'
                res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res -= np.einsum('a,b,c->abc', U[:, r], v, w)
                # We want U[:,r] minimizing ||T - (res + U[:,r] v w)||^2
                # => minimize ||res + U[:,r] (v⊗w)||^2
                # This is a linear least squares in U[:,r].
                rhs = -res.reshape(-1)
                # feature vector for U[:,r]: for each (a,b,c), coeff = v[b]*w[c]
                feat = (v[:, None] * w[None, :]).reshape(-1)  # length n_b*n_c
                # objective over a: sum_{a} || rhs_a + U_a * feat ||^2
                # where rhs_a = res.reshape(n_a, -1)[a]
                resA = res.reshape(n_a, -1)
                # for each a: min_Ua || resA[a] + Ua*feat ||^2
                # Ua = - (resA[a]·feat) / (feat·feat)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                U[:, r] = -(resA @ feat) / f2
            # --- update V ---
            for r in range(rank):
                u = U[:, r]; w = W[:, r]
                res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res -= np.einsum('a,b,c->abc', u, V[:, r], w)
                rhs = -res.reshape(-1)
                feat = (u[:, None] * w[None, :]).reshape(-1)
                resB = res.reshape(n_b, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                V[:, r] = -(resB @ feat) / f2
            # --- update W ---
            for r in range(rank):
                u = U[:, r]; v = V[:, r]
                res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                res -= np.einsum('a,b,c->abc', u, v, W[:, r])
                rhs = -res.reshape(-1)
                feat = (u[:, None] * v[None, :]).reshape(-1)
                resC = res.reshape(n_c, -1)
                f2 = float(feat @ feat)
                if f2 < 1e-14:
                    continue
                W[:, r] = -(resC @ feat) / f2
        return U, V, W

    best = None
    best_err = np.inf
    base_U, base_V, base_W = _rank17()

    # Restart strategy: drop a different column from rank-17 each time.
    for restart in range(40):
        if time.time() > deadline - 5:
            break
        drop = restart % 17
        U = np.delete(base_U, drop, axis=1).copy()
        V = np.delete(base_V, drop, axis=1).copy()
        W = np.delete(base_W, drop, axis=1).copy()

        # small random perturbation to escape the exact rank-17 manifold
        if restart >= 17:
            U = U + 0.01 * np.random.randn(n_a, rank)
            V = V + 0.01 * np.random.randn(n_b, rank)
            W = W + 0.01 * np.random.randn(n_c, rank)

        sweeps = 60 if restart < 20 else 120
        U, V, W = refine(U, V, W, sweeps, deadline)
        err = error_of(U, V, W)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-6:
            return U, V, W

    if best is not None:
        U, V, W = best
        if error_of(U, V, W) < 1e-6:
            return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under the 150s kill

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # 1) Try to find a rank-16 decomposition.
    r16 = _try_rank16(deadline)
    if r16 is not None:
        U, V, W = r16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2) Verified rank-17 (Strassen + direct).
    U, V, W = _rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3) Final fallback: exact schoolbook.
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
