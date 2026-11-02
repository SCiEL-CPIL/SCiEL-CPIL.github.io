"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen.
    Each term is (u_dict, v_dict, w_dict) with local indices:
    u: (i, j) for A[i,j]; v: (j, k) for B[j,k]; w: (i, k) for C[i,k].
    """
    # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # M2: (A10+A11)B00 -> C10(+1), C11(-1)
    # M3: A00(B01-B11) -> C01(+1), C11(+1)
    # M4: A11(B10-B00) -> C00(+1), C10(+1)
    # M5: (A00+A01)B11 -> C00(-1), C01(+1)
    # M6: (A10-A00)(B00+B01) -> C11(+1)
    # M7: (A01-A11)(B10+B11) -> C00(+1)
    terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    return terms


def _schoolbook_2x2x3_terms():
    """Return the 6 rank-1 terms for 2x2x3 schoolbook.
    u: (i, j) for A[i,j] (2x2); v: (j, k) for B[j,k] (2x3); w: (i, k) for C[i,k] (2x3).
    """
    terms = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                terms.append((
                    {(i, j): 1},
                    {(j, k): 1},
                    {(i, k): 1},
                ))
    return terms


def _build_kronecker_decomposition(n, m, p):
    """Build the 4x4x6 decomposition as the Kronecker product of
    2x2x2 (rank 7) and 2x2x3 (rank 6) tensors. Total rank = 42.
    """
    s_terms = _strassen_2x2_terms()
    b_terms = _schoolbook_2x2x3_terms()

    # Build full U (16 x 42), V (24 x 42), W (24 x 42)
    U = np.zeros((n * m, 42), dtype=float)
    V = np.zeros((m * p, 42), dtype=float)
    W = np.zeros((p * n, 42), dtype=float)

    r = 0
    for su, sv, sw in s_terms:
        for bu, bv, bw in b_terms:
            # For each local index, combine (i1,i2), (j1,j2), (k1,k2)
            # U index: (i1*2+i2, j1*2+j2) -> (i1*2+i2)*m + (j1*2+j2)
            # V index: (j1*2+j2, k1*3+k2) -> (j1*2+j2)*p + (k1*3+k2)
            # W index: (k1*3+k2, i1*2+i2) -> (k1*3+k2)*n + (i1*2+i2)
            # Since su keys are (i, j) local to 2x2, and bu keys are (i, j) local to 2x2,
            # the combined U index is (i_s*2 + i_b, j_s*2 + j_b)
            # Similarly for V: su key (i,j) gives j_s, bu key (j,k) gives j_b, k_b
            # And for W: su key (i,k) gives i_s, k_s; bu key (i,k) gives i_b, k_b

            # We need to be careful: su is (i, j), sv is (j, k), sw is (i, k) for the 2x2x2
            # bu is (i, j), bv is (j, k), bw is (i, k) for the 2x2x3

            # For each combination of (i_s, j_s) in su, (j_s, k_s) in sv, (i_s, k_s) in sw
            # and (i_b, j_b) in bu, (j_b, k_b) in bv, (i_b, k_b) in bw
            # The global indices are:
            # i = i_s*2 + i_b, j = j_s*2 + j_b, k = k_s*3 + k_b
            # U[i*m + j] += su[(i_s,j_s)] * bu[(i_b,j_b)]
            # V[j*p + k] += sv[(j_s,k_s)] * bv[(j_b,k_b)]
            # W[k*n + i] += sw[(i_s,k_s)] * bw[(i_b,k_b)]

            for (i_s, j_s), c_s_u in su.items():
                for (j_s2, k_s), c_s_v in sv.items():
                    assert j_s == j_s2, "Strassen U and V j-indices must match"
                    for (i_s2, k_s2), c_s_w in sw.items():
                        assert i_s == i_s2 and k_s == k_s2, "Strassen W indices must match U and V"
                        for (i_b, j_b), c_b_u in bu.items():
                            for (j_b2, k_b), c_b_v in bv.items():
                                assert j_b == j_b2, "Schoolbook U and V j-indices must match"
                                for (i_b2, k_b2), c_b_w in bw.items():
                                    assert i_b == i_b2 and k_b == k_b2, "Schoolbook W indices must match U and V"

                                    gi = i_s * 2 + i_b
                                    gj = j_s * 2 + j_b
                                    gk = k_s * 3 + k_b

                                    U[gi * m + gj, r] += c_s_u * c_b_u
                                    V[gj * p + gk, r] += c_s_v * c_b_v
                                    W[gk * n + gi, r] += c_s_w * c_b_w
            r += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify the decomposition is correct."""
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += np.einsum('ij,jk,ki->ik', u_vec, v_vec, w_vec)

    return np.allclose(C_computed, C_expected, atol=1e-10)


def _remove_redundant_terms(U, V, W, n, m, p, tol=1e-9):
    """Remove linearly dependent rank-1 terms from the decomposition."""
    rank = U.shape[1]
    keep = list(range(rank))

    for i in range(len(keep)):
        if i >= len(keep):
            break
        ri = keep[i]
        for j in range(i + 1, len(keep)):
            rj = keep[j]
            u_i = U[:, ri]
            u_j = U[:, rj]
            v_i = V[:, ri]
            v_j = V[:, rj]
            w_i = W[:, ri]
            w_j = W[:, rj]

            mask_u = np.abs(u_i) > tol
            if not np.any(mask_u):
                if np.allclose(u_j, 0, atol=tol):
                    mask_v = np.abs(v_i) > tol
                    if not np.any(mask_v):
                        if np.allclose(v_j, 0, atol=tol) and np.allclose(w_j, 0, atol=tol):
                            keep.remove(rj)
                            j -= 1
                        continue
                    continue
                else:
                    continue

            c_u = u_j[mask_u] / u_i[mask_u]
            if not np.allclose(c_u, c_u[0], rtol=1e-7, atol=tol):
                continue
            c = c_u[0]

            if not np.allclose(v_j, c * v_i, rtol=1e-7, atol=tol):
                continue
            if not np.allclose(w_j, c * w_i, rtol=1e-7, atol=tol):
                continue

            keep.remove(rj)
            j -= 1

    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, True
    return U, V, W, False


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build the Kronecker product decomposition
    U, V, W = _build_kronecker_decomposition(n, m, p)

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to remove redundant terms
    U, V, W, reduced = _remove_redundant_terms(U, V, W, n, m, p)

    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If reduction broke it, rebuild
        U, V, W = _build_kronecker_decomposition(n, m, p)

    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
