"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_target_mask(n, m, p):
    """Precompute the target mask and index arrays for verification."""
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    idx_a, idx_b, idx_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = True
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    return target, np.array(idx_a), np.array(idx_b), np.array(idx_c)

def _verify(U, V, W, tol=1e-9, target_mask=None, idx_a=None, idx_b=None, idx_c=None):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build tensor via sum of outer products
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

    if target_mask is None:
        target_mask, idx_a, idx_b, idx_c = _build_target_mask(n, m, p)

    target_vals = T[idx_a, idx_b, idx_c]
    max_err = np.max(np.abs(target_vals - 1.0))
    non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0

    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5, target_mask=None, idx_a=None, idx_b=None, idx_c=None):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol,
               target_mask=target_mask, idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()

    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol,
               target_mask=target_mask, idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()

    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Split B (3x8) into 4 blocks of 3x2.
    Each block: Strassen rank-7 for 2x2*2x2 part + 4 rank-1 terms for extra row.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))

    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen rank-7 for A[0:2,0:2] * B[0:2,k0:k1+1]
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1

        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1

        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1

        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1

        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1

        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1

        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1

        # 4 rank-1 terms: A[:,2] * B[2,k] -> C[:,k]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1

    assert col_offset == 44, f"Expected 44, got {col_offset}"
    return U.tolist(), V.tolist(), W.tolist()

def _build_rank42_decomposition():
    """
    Try to build rank-42 decomposition.
    Split B (3x8) into 4 blocks of 3x2.
    For 3 blocks use Strassen(7)+4=11, for 1 block try a different structure.
    Alternatively: use a 2x3 x 3x4 block with rank 15 instead of 2*11=22.
    """
    n, m, p = 2, 3, 8
    # Try: 2 blocks of 3x4, each with rank 21 = 42 total
    # For 2x3 x 3x4: schoolbook is 24. Can we do 21?
    # This is equivalent to computing C = A @ B where A is 2x3, B is 3x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We need rank-21 for this subproblem.
    # This is non-trivial. Let's try a hybrid approach instead.
    
    # Alternative: 3 blocks of 3x2 (rank 11 each = 33) + 1 block of 3x2 (rank 9)
    # Total = 42
    # For the last block, can we do 2x3 x 3x2 with rank 9?
    # Schoolbook is 12. Strassen gives 7+4=11. Can we do 9?
    # 2x3 x 3x2: C[i,k] = a[i,0]*b[0,k] + a[i,1]*b[1,k] + a[i,2]*b[2,k]
    # This is a bilinear form. The tensor is 2*3*2 = 12 entries.
    # Lower bound: max(2*3, 3*2, 2*2) = 6. So 9 might be possible.
    
    # Let me try to construct it explicitly.
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Try rank-9:
    # m1 = (a00+a11)(b00+b11) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # m2 = (a00-a11)(b00-b11) -> same
    # m3 = (a01+a10)(b10+b01) -> same
    # m4 = (a01-a10)(b10-b01) -> same
    # m5 = a02*b20 -> C[0,0]
    # m6 = a02*b21 -> C[0,1]
    # m7 = a12*b20 -> C[1,0]
    # m8 = a12*b21 -> C[1,1]
    # m9 = ? We need to get the cross terms right.
    
    # Actually this is getting complex. Let's just return None and rely on rank-44.
    return None

def _als_decompose(rank, max_iters=80, deadline=None, seed=None):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    Uses batched least-squares for speed.
    """
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(seed)

    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))

    # Precompute index structures
    # For updating U[a]: we need V[j*p+k] and W[k*n+i] for all k, where a = i*m+j
    # For updating V[b]: b = j*p+k, need U[i*m+j] and W[k*n+i] for all i
    # For updating W[c]: c = k*n+i, need U[i*m+j] and V[j*p+k] for all j

    # Precompute target mask and indices once
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    idx_a, idx_b, idx_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_mask[i*m+j, j*p+k, k*n+i] = True
                idx_a.append(i*m+j)
                idx_b.append(j*p+k)
                idx_c.append(k*n+i)
    idx_a = np.array(idx_a); idx_b = np.array(idx_b); idx_c = np.array(idx_c)
    non_target_mask = ~target_mask

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # --- Update U ---
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass

        # --- Update V ---
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass

        # --- Update W ---
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass

        # Check convergence (every few iterations to save time)
        if it % 3 == 2 or it == max_iters - 1:
            T = np.einsum('ar,br,cr->abc', U, V, W)
            target_err = np.max(np.abs(T[idx_a, idx_b, idx_c] - 1.0))
            non_target_err = np.max(np.abs(T[non_target_mask])) if non_target_mask.any() else 0.0
            if target_err < 1e-8 and non_target_err < 1e-8:
                return U.tolist(), V.tolist(), W.tolist()

    return None

def _polish_decomposition(U, V, W, deadline=None, tol=1e-10):
    """
    High-precision polish pass: refine U, V, W with tighter tolerance.
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(U[0])

    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    idx_a, idx_b, idx_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_mask[i*m+j, j*p+k, k*n+i] = True
                idx_a.append(i*m+j)
                idx_b.append(j*p+k)
                idx_c.append(k*n+i)
    idx_a = np.array(idx_a); idx_b = np.array(idx_b); idx_c = np.array(idx_c)
    non_target_mask = ~target_mask

    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break

        # --- Update U ---
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=1e-14)[0]
                except:
                    pass

        # --- Update V ---
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=1e-14)[0]
                except:
                    pass

        # --- Update W ---
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(m), rcond=1e-14)[0]
                except:
                    pass

        if it % 5 == 4 or it == 49:
            T = np.einsum('ar,br,cr->abc', U, V, W)
            target_err = np.max(np.abs(T[idx_a, idx_b, idx_c] - 1.0))
            non_target_err = np.max(np.abs(T[non_target_mask])) if non_target_mask.any() else 0.0
            if target_err < tol and non_target_err < tol:
                return U.tolist(), V.tolist(), W.tolist()

    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48

    # Precompute target mask
    target_mask, idx_a, idx_b, idx_c = _build_target_mask(n, m, p)

    # Schoolbook fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1

    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank

    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Step 1: Try the explicit rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6, target_mask=target_mask,
                   idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
            rounded = _round_and_verify(U44, V44, W44, tol=1e-5,
                                        target_mask=target_mask,
                                        idx_a=idx_a, idx_b=idx_b, idx_c=idx_c)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass

    # Step 2: Try to find better rank via ALS (limited time budget)
    if best_rank <= 44:
        # Try ranks from 42 up to best_rank-1
        for rank in range(42, best_rank):
            if time.time() > deadline:
                break

            found = False
            max_trials = 5
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                seed = rank * 100 + trial
                result = _als_decompose(rank, max_iters=40, deadline=deadline, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6, target_mask=target_mask,
                               idx_a=idx_a, idx_b=idx_b, idx_c=idx_c):
                        # Polish for higher precision
                        polished = _polish_decomposition(U, V, W, deadline=deadline, tol=1e-10)
                        if polished:
                            U, V, W = polished
                        rounded = _round_and_verify(U, V, W, tol=1e-5,
                                                    target_mask=target_mask,
                                                    idx_a=idx_a, idx_b=idx_b, idx_c=idx_c)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        found = True
                        break

            if found:
                continue

    return {"U": best_U, "V": best_V, "W": best_W}
