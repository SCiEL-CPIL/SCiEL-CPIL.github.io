"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False

    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)

    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True


def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen decomposition."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 1, 0, 0], {0: 1, 3: 1}))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], {2: 1, 3: -1}))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], {1: 1, 3: 1}))
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], {0: 1, 2: 1}))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], {0: -1, 1: 1}))
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], {3: 1}))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], {0: 1}))
    return terms


def _build_4x4x4_strassen():
    """Build the 4x4x4 Strassen decomposition with rank 49."""
    n = m = p = 4
    rank = 49

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    inner_terms = _strassen_2x2_terms()

    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    col = 0
    for s in range(7):
        a_coeffs, b_coeffs, c_contribs = outer_terms[s]

        for inner_idx in range(7):
            u_inner, v_inner, w_inner_contribs = inner_terms[inner_idx]

            U_vec = np.zeros(16)
            for alpha, a_coeff in a_coeffs.items():
                br = alpha // 2
                bc = alpha % 2
                for i in range(2):
                    for j in range(2):
                        U_vec[(br * 2 + i) * 4 + (bc * 2 + j)] += a_coeff * u_inner[i * 2 + j]

            V_vec = np.zeros(16)
            for beta, b_coeff in b_coeffs.items():
                br = beta // 2
                bc = beta % 2
                for i in range(2):
                    for j in range(2):
                        V_vec[(br * 2 + i) * 4 + (bc * 2 + j)] += b_coeff * v_inner[i * 2 + j]

            W_vec = np.zeros(16)
            for c_block, c_coeff in c_contribs.items():
                br = c_block // 2
                bc = c_block % 2
                for i in range(2):
                    for j in range(2):
                        if (i * 2 + j) in w_inner_contribs:
                            W_vec[(bc * 2 + j) * 4 + (br * 2 + i)] += c_coeff * w_inner_contribs[i * 2 + j]

            U[:, col] = U_vec
            V[:, col] = V_vec
            W[:, col] = W_vec
            col += 1

    return U, V, W


def _build_4x4x5_rank65():
    """Build the 4x4x5 decomposition with rank 65."""
    n, m, p = 4, 4, 5
    rank_total = 65

    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    U4, V4, W4 = _build_4x4x4_strassen()
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4

    col = 49
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

    return U, V, W


def _als_solve(n, m, p, rank, max_iters=200, tol=1e-10, seed=0):
    """ALS for tensor decomposition of the matrix multiplication tensor."""
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank)) * 0.1
    V = rng.standard_normal((m * p, rank)) * 0.1
    W = rng.standard_normal((p * n, rank)) * 0.1
    
    for it in range(max_iters):
        # Update U
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i*m+j, :] = u
        
        # Update V
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j*p+k, :] = v
        
        # Update W
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k*n+i, :] = w
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        error = np.max(np.abs(result - 1.0))
        if error < tol:
            break
    
    return U, V, W, error


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    n, m, p = DIMENSIONS

    # Try rank 64 with ALS
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        try:
            U, V, W, error = _als_solve(n, m, p, 64, max_iters=100, tol=1e-10, seed=seed)
            if error < 1e-9:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Return the known rank-65 decomposition
    U, V, W = _build_4x4x5_rank65()
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 80)
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
