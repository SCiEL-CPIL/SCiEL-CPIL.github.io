"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition:
    - Part 1: A2 * B2 (j=2, k=0..4)  -> 10 terms (outer product of 2x1 and 1x5)
    - Part 2: A1 * B13 (j=0,1, k=4)  -> 4 terms (2x2 * 2x1)
    - Part 3: A1 * B11 (j=0,1, k=0,1) -> 7 terms (Strassen 2x2 * 2x2)
    - Part 4: A1 * B12 (j=0,1, k=2,3) -> 7 terms (Strassen 2x2 * 2x2)
    Total: 10 + 4 + 7 + 7 = 28
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0

    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1

    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1 = (A00+A11)*(B00+B11)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1

    # M2 = (A10+A11)*B00
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1

    # M3 = A00*(B11-B01)
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1

    # M4 = A11*(B00-B10)
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1

    # M5 = (A00+A01)*B11
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1

    # M6 = (A10-A00)*(B00+B01)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1

    # M7 = (A01-A11)*(B10+B11)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1

    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1 = (A00+A11)*(B02+B13)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1

    # M2 = (A10+A11)*B02
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1

    # M3 = A00*(B13-B03)
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1

    # M4 = A11*(B02-B12)
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1

    # M5 = (A00+A01)*B13
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1

    # M6 = (A10-A00)*(B02+B03)
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1

    # M7 = (A01-A11)*(B12+B13)
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1

    assert r == rank, f"Expected {rank} columns, got {r}"
    return {"U": U, "V": V, "W": W}


def _is_rank1_tensor(res, tol=1e-8):
    """Check if a 3-tensor res[a,b,c] is rank-1 (res[a,b,c] = u[a]*v[b]*w[c]).
    Returns (True, u, v, w) if rank-1, else (False, None, None, None)."""
    if np.max(np.abs(res)) < 1e-12:
        return True, None, None, None  # zero tensor is trivially rank-1

    # Find a nonzero slice in the c dimension
    c_ref = None
    for c in range(res.shape[2]):
        if np.max(np.abs(res[:, :, c])) > 1e-12:
            c_ref = c
            break
    if c_ref is None:
        return True, None, None, None

    ref_slice = res[:, :, c_ref]

    # Check proportionality of all slices to ref_slice
    w_vec = np.zeros(res.shape[2])
    w_vec[c_ref] = 1.0
    for c in range(res.shape[2]):
        slice_c = res[:, :, c]
        if np.max(np.abs(slice_c)) < 1e-12:
            w_vec[c] = 0.0
            continue
        # Find a nonzero entry in ref_slice
        idx = np.nonzero(ref_slice)
        if len(idx[0]) == 0:
            return False, None, None, None
        a0, b0 = idx[0][0], idx[0][1]
        if abs(ref_slice[a0, b0]) < 1e-12:
            return False, None, None, None
        ratio = slice_c[a0, b0] / ref_slice[a0, b0]
        w_vec[c] = ratio
        # Check proportionality
        if np.max(np.abs(slice_c - ratio * ref_slice)) > tol * max(1.0, np.max(np.abs(slice_c))):
            return False, None, None, None

    # Now check if ref_slice is rank-1 as a matrix
    from numpy.linalg import svd
    s_vals = svd(ref_slice, compute_uv=False)
    if len(s_vals) > 1 and s_vals[1] > tol * max(s_vals[0], 1e-12):
        return False, None, None, None

    U_mat, s, Vt = svd(ref_slice, full_matrices=False)
    if s[0] < 1e-12:
        return False, None, None, None

    u_vec = U_mat[:, 0] * np.sqrt(s[0])
    v_vec = Vt[0, :] * np.sqrt(s[0])

    # Verify the full factorization
    recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
    if np.max(np.abs(recon - res)) > tol:
        return False, None, None, None

    return True, u_vec, v_vec, w_vec


def _try_rank_reduction(T, U, V, W, deadline):
    """Try to reduce rank by dropping one term and checking if the residual is rank-1.
    Only returns a verified lower-rank decomposition."""
    rank = U.shape[1]
    best_result = None

    for r0 in range(rank):
        if time.time() > deadline:
            break

        # Remove term r0
        U_rem = np.delete(U, r0, axis=1)
        V_rem = np.delete(V, r0, axis=1)
        W_rem = np.delete(W, r0, axis=1)

        # Compute the residual tensor
        T_recon = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem, optimize=True)
        residual = T - T_recon

        # Check if residual is rank-1
        is_r1, u, v, w = _is_rank1_tensor(residual)
        if is_r1:
            if u is None:
                # Residual is zero: we have a valid rank-(rank-1) decomposition
                new_result = {
                    'U': U_rem.tolist(),
                    'V': V_rem.tolist(),
                    'W': W_rem.tolist()
                }
                valid, diff = _verify_decomposition(new_result['U'], new_result['V'], new_result['W'], T)
                if valid:
                    best_result = new_result
                    break  # Can't do better than rank-1 reduction from this start
            else:
                # Residual is rank-1: we can replace the dropped term with this rank-1 term
                # This keeps the same rank, so no improvement. Skip.
                pass

    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 5
    T = _build_target_tensor(n, m, p)

    # Try rank-28 decomposition first
    result = _rank28_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)

    if valid:
        # Attempt to reduce rank by searching for linear dependencies
        if time.time() < deadline - 10:
            U_arr = np.array(result['U'], dtype=np.float64)
            V_arr = np.array(result['V'], dtype=np.float64)
            W_arr = np.array(result['W'], dtype=np.float64)
            reduced = _try_rank_reduction(T, U_arr, V_arr, W_arr, deadline - 5)
            if reduced is not None:
                return reduced
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
