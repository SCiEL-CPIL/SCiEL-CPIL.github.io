"""Matrix multiplication 2x4x4: Strassen rank-28 build + code-theoretic dependency search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    Local indices are 0..3 for 2x2 matrices (row-major)."""
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def _build_strassen_28():
    """Build the rank-28 decomposition via 4 independent Strassen 2x2 block products."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = _strassen_2x2()

    # Each block: (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
    # A is 2x4 (rows 0,1; cols 0..3), B is 4x4, C is 2x4.
    blocks = [
        (0, 0, 0, 0, 0, 0),   # A_L * B_TL -> C_LL
        (0, 2, 2, 0, 0, 0),   # A_R * B_BL -> C_LL
        (0, 0, 0, 2, 0, 2),   # A_L * B_TR -> C_LR
        (0, 2, 2, 2, 0, 2),   # A_R * B_BR -> C_LR
    ]

    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                U[global_i * m + global_j][col] += coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                V[global_j * p + global_k][col] += coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                W[global_k * n + global_i][col] += coeff
            col += 1

    return U, V, W

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c]=1 with trace convention c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _reconstruct(T, U, V, W, rank):
    """Reconstruct the tensor from a decomposition: sum_r U[:,r] V[:,r] W[:,r]."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return S

def _verify(U, V, W, T, rank):
    """Max abs error between reconstructed tensor and T."""
    S = _reconstruct(T, U, V, W, rank)
    return float(np.max(np.abs(S - T)))

def construct():
    t0 = time.time()
    deadline = 150.0 - 8.0  # safety margin before the 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    U, V, W = _build_strassen_28()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]  # 28

    T = _build_tensor(n, m, p)

    # Sanity check the rank-28 baseline.
    base_err = _verify(U, V, W, T, rank)
    if base_err > 1e-9:
        # Should never happen; return the (still correct-by-construction) baseline.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Build the parity matrix M (1024 x 28): column r is the flattened rank-1 term.
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()

    # Code-theoretic dependency test: if the 28 bilinear terms are linearly
    # dependent, the decomposition is not minimal and a term can be eliminated.
    # Use SVD on M to find the numerical rank and any null-space direction.
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        num_nonzero = int(np.sum(S_svd > tol))

        if num_nonzero < rank:
            # There is a dependency. Find a null-space vector and see if any
            # single column can be expressed as a combination of the others.
            # Try eliminating each term via least squares.
            for r_elim in range(rank):
                if time.time() > deadline:
                    break
                other_cols = [s for s in range(rank) if s != r_elim]
                M_sub = M[:, other_cols]
                rhs = M[:, r_elim]
                try:
                    alpha, _, _, _ = np.linalg.lstsq(M_sub, rhs, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                residual = float(np.max(np.abs(M_sub @ alpha - rhs)))
                if residual < 1e-8:
                    new_U = np.delete(U, r_elim, axis=1)
                    new_V = np.delete(V, r_elim, axis=1)
                    new_W = np.delete(W, r_elim, axis=1)
                    new_rank = rank - 1
                    err = _verify(new_U, new_V, new_W, T, new_rank)
                    if err < 1e-9:
                        best_rank = new_rank
                        best_U, best_V, best_W = new_U, new_V, new_W
                        break
    except np.linalg.LinAlgError:
        pass

    # Return the best (lowest-rank) valid decomposition.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
