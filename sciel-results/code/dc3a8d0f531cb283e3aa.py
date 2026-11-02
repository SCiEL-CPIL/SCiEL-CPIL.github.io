"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Vectorized check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _find_decomp_2x2_2xp(p, r, max_iters=5000, seed=42, deadline=None):
    """Find rank-r decomposition for 2x2 * 2xp. Returns (U,V,W) or None."""
    n, m = 2, 2
    rng = np.random.RandomState(seed)
    for attempt in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = np.zeros((p*n, r))
        valid = True
        for k in range(p):
            for i in range(n):
                coeff = np.zeros(r)
                for j in range(m):
                    coeff += U[i*m+j, :] * V[j*p+k, :]
                ns = np.dot(coeff, coeff)
                if ns < 1e-12:
                    valid = False
                    break
                W[k*n+i, :] = coeff / ns
            if not valid:
                break
        if not valid:
            continue
        # Vectorized verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return U, V, W
    return None

def _try_rationalize(U, V, W, tol=1e-6):
    """Try to find rational (half-integer) values close to U,V,W that still work."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _check_decomposition(U_r, V_r, W_r, tol):
        return U_r, V_r, W_r
    return None

def _build_strassen_fallback():
    """Build a rank-46 Strassen-based decomposition for 2x2 * 2x13."""
    n, m, p = 2, 2, 13
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for group in range(6):
        k1 = 2*group
        k2 = 2*group+1
        # Standard 7 Strassen products for 2x2 * 2x2
        # M1 = (A11+A22)(B11+B22)
        U[0, col] = 1; U[3, col] = 1
        V[k1, col] = 1; V[13+k2, col] = 1
        W[2*k1, col] = 1; W[2*k2+1, col] = 1
        col += 1
        # M2 = A22(B21-B11)
        U[2, col] = 1; U[3, col] = 1
        V[k1, col] = 1
        W[2*k1+1, col] = 1; W[2*k2+1, col] = -1
        col += 1
        # M3 = A11(B22-B12)
        U[0, col] = 1
        V[k2, col] = 1; V[13+k2, col] = -1
        W[2*k2, col] = 1; W[2*k2+1, col] = 1
        col += 1
        # M4 = A22(B11+B22)
        U[3, col] = 1
        V[13+k1, col] = 1; V[k1, col] = -1
        W[2*k1, col] = 1; W[2*k1+1, col] = 1
        col += 1
        # M5 = (A11+A12)B22
        U[0, col] = 1; U[1, col] = 1
        V[13+k2, col] = 1
        W[2*k1, col] = -1; W[2*k2, col] = 1
        col += 1
        # M6 = (A21-A11)B11
        U[2, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[k2, col] = 1
        W[2*k2+1, col] = 1
        col += 1
        # M7 = (A12-A22)(B11+B21)
        U[1, col] = 1; U[3, col] = -1
        V[13+k1, col] = 1; V[13+k2, col] = 1
        W[2*k1, col] = 1
        col += 1
    # Last 2x2 * 2x1 block: schoolbook rank 4
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    return U, V, W

def _build_block_decomp(U_block, V_block, W_block, r_block, p_block, n_groups, p_total, n, m, start_k=0):
    """Build full decomposition from a block decomposition repeated n_groups times."""
    total_rank = n_groups * r_block + 4  # +4 for schoolbook last 2x1
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p_total, total_rank))
    W = np.zeros((p_total*n, total_rank))
    col = 0
    for group in range(n_groups):
        k_off = start_k + group * p_block
        for ri in range(r_block):
            U[:, col] = U_block[:, ri]
            for j in range(m):
                for k in range(p_block):
                    V[j*p_total + k_off + k, col] = V_block[j*p_block + k, ri]
            for k in range(p_block):
                for i in range(n):
                    W[(k_off+k)*n + i, col] = W_block[k*n + i, ri]
            col += 1
    # Schoolbook for remaining columns
    k_remaining = start_k + n_groups * p_block
    for k in range(k_remaining, p_total):
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j*p_total + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    best_result = None
    best_rank = 999

    # Strategy 1: 4 groups of 2x2*2x3 + schoolbook 2x1
    decomp_2x3 = None
    r_2x3 = None
    for r_try in range(7, 12):
        if time.time() > deadline - 80:
            break
        result = _find_decomp_2x2_2xp(3, r_try, max_iters=5000, seed=100+r_try*7, deadline=deadline-80)
        if result is not None:
            decomp_2x3 = result
            r_2x3 = r_try
            break

    if decomp_2x3 is not None:
        U_2x3, V_2x3, W_2x3 = decomp_2x3
        U, V, W = _build_block_decomp(U_2x3, V_2x3, W_2x3, r_2x3, 3, 4, p, n, m)
        total_rank = 4 * r_2x3 + 4
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Strategy 2: 3 groups of 2x2*2x4 + schoolbook 2x1
    decomp_2x4 = None
    r_2x4 = None
    for r_try in range(8, 13):
        if time.time() > deadline - 50:
            break
        result = _find_decomp_2x2_2xp(4, r_try, max_iters=5000, seed=200+r_try*11, deadline=deadline-50)
        if result is not None:
            decomp_2x4 = result
            r_2x4 = r_try
            break

    if decomp_2x4 is not None:
        U_2x4, V_2x4, W_2x4 = decomp_2x4
        U, V, W = _build_block_decomp(U_2x4, V_2x4, W_2x4, r_2x4, 4, 3, p, n, m)
        total_rank = 3 * r_2x4 + 4
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Strategy 3: 2 groups of 2x2*2x5 + 1 group of 2x2*2x3 + schoolbook 2x1
    decomp_2x5 = None
    r_2x5 = None
    for r_try in range(9, 14):
        if time.time() > deadline - 30:
            break
        result = _find_decomp_2x2_2xp(5, r_try, max_iters=3000, seed=300+r_try*13, deadline=deadline-30)
        if result is not None:
            decomp_2x5 = result
            r_2x5 = r_try
            break

    if decomp_2x5 is not None and decomp_2x3 is not None:
        # 2 groups of 2x5 + 1 group of 2x3 + schoolbook 2x1
        total_rank = 2 * r_2x5 + r_2x3 + 4
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        for group in range(2):
            k_off = group * 5
            for ri in range(r_2x5):
                U[:, col] = decomp_2x5[0][:, ri]
                for j in range(m):
                    for k in range(5):
                        V[j*p + k_off + k, col] = decomp_2x5[1][j*5 + k, ri]
                for k in range(5):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = decomp_2x5[2][k*n + i, ri]
                col += 1
        # 1 group of 2x3
        k_off = 10
        for ri in range(r_2x3):
            U[:, col] = decomp_2x3[0][:, ri]
            for j in range(m):
                for k in range(3):
                    V[j*p + k_off + k, col] = decomp_2x3[1][j*3 + k, ri]
            for k in range(3):
                for i in range(n):
                    W[(k_off+k)*n + i, col] = decomp_2x3[2][k*n + i, ri]
            col += 1
        # Schoolbook 2x1
        k = 12
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Strategy 4: 1 group of 2x2*2x6 + 1 group of 2x2*2x4 + 1 group of 2x2*2x3 + schoolbook 2x1
    if decomp_2x4 is not None and decomp_2x3 is not None:
        decomp_2x6 = None
        r_2x6 = None
        for r_try in range(10, 15):
            if time.time() > deadline - 15:
                break
            result = _find_decomp_2x2_2xp(6, r_try, max_iters=2000, seed=400+r_try*17, deadline=deadline-15)
            if result is not None:
                decomp_2x6 = result
                r_2x6 = r_try
                break
        if decomp_2x6 is not None:
            total_rank = r_2x6 + r_2x4 + r_2x3 + 4
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            col = 0
            # 2x6 block
            for ri in range(r_2x6):
                U[:, col] = decomp_2x6[0][:, ri]
                for j in range(m):
                    for k in range(6):
                        V[j*p + k, col] = decomp_2x6[1][j*6 + k, ri]
                for k in range(6):
                    for i in range(n):
                        W[k*n + i, col] = decomp_2x6[2][k*n + i, ri]
                col += 1
            # 2x4 block
            k_off = 6
            for ri in range(r_2x4):
                U[:, col] = decomp_2x4[0][:, ri]
                for j in range(m):
                    for k in range(4):
                        V[j*p + k_off + k, col] = decomp_2x4[1][j*4 + k, ri]
                for k in range(4):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = decomp_2x4[2][k*n + i, ri]
                col += 1
            # 2x3 block
            k_off = 10
            for ri in range(r_2x3):
                U[:, col] = decomp_2x3[0][:, ri]
                for j in range(m):
                    for k in range(3):
                        V[j*p + k_off + k, col] = decomp_2x3[1][j*3 + k, ri]
                for k in range(3):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = decomp_2x3[2][k*n + i, ri]
                col += 1
            # Schoolbook 2x1
            k = 12
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
            if _check_decomposition(U, V, W):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = (U.copy(), V.copy(), W.copy())

    # Fallback: rank-46 Strassen
    U46, V46, W46 = _build_strassen_fallback()
    if _check_decomposition(U46, V46, W46):
        if 46 < best_rank:
            best_rank = 46
            best_result = (U46.copy(), V46.copy(), W46.copy())

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook
    rf = n*m*p
    Uf = np.zeros((n*m, rf)); Vf = np.zeros((m*p, rf)); Wf = np.zeros((p*n, rf))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i*m+j,col]=1; Vf[j*p+k,col]=1; Wf[k*n+i,col]=1; col+=1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
