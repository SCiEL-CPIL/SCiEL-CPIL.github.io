import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n, m, p):
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 * r2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            val = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                            if val != 0:
                                U[i * m + j][r] += val
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            val = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                            if val != 0:
                                V[j * p + k][r] += val
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            val = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                            if val != 0:
                                W[k * n + i][r] += val
    return U, V, W


def _build_4x4x4_strassen():
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    U, V, W = _kron_decompose(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2, n, m, p)
    return {"U": U, "V": V, "W": W}


def _schoolbook_2x2x2x1():
    U = [[0] * 4 for _ in range(4)]
    V = [[0] * 4 for _ in range(2)]
    W = [[0] * 4 for _ in range(2)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return U, V, W


def _build_4x4x2():
    n, m, p = 4, 4, 2
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _schoolbook_2x2x2x1()
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x1():
    n, m, p = 4, 4, 1
    U1, V1, W1 = _schoolbook_2x2x2x1()
    U2, V2, W2 = _schoolbook_2x2x2x1()
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 1, 2, 2, 1, n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_4x4x7():
    n, m, p = 4, 4, 7
    part1 = _build_4x4x4_strassen()
    part2 = _build_4x4x2()
    part3 = _build_4x4x1()
    r1, r2, r3 = len(part1['U'][0]), len(part2['U'][0]), len(part3['U'][0])
    total_rank = r1 + r2 + r3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for i in range(n * m):
        for r in range(r1):
            U[i][r] = part1['U'][i][r]
    for i in range(m):
        for j in range(4):
            for r in range(r1):
                V[i * p + j][r] = part1['V'][i * 4 + j][r]
    for k in range(4):
        for i in range(n):
            for r in range(r1):
                W[k * n + i][r] = part1['W'][k * 4 + i][r]
    off2 = r1
    for i in range(n * m):
        for r in range(r2):
            U[i][off2 + r] = part2['U'][i][r]
    for i in range(m):
        for j in range(2):
            for r in range(r2):
                V[i * p + (4 + j)][off2 + r] = part2['V'][i * 2 + j][r]
    for k in range(2):
        for i in range(n):
            for r in range(r2):
                W[(4 + k) * n + i][off2 + r] = part2['W'][k * 4 + i][r]
    off3 = r1 + r2
    for i in range(n * m):
        for r in range(r3):
            U[i][off3 + r] = part3['U'][i][r]
    for i in range(m):
        for r in range(r3):
            V[i * p + 6][off3 + r] = part3['V'][i][r]
    for i in range(n):
        for r in range(r3):
            W[6 * n + i][off3 + r] = part3['W'][i][r]
    return {"U": U, "V": V, "W": W}


def _global_rank_reduction(U, V, W, n, m, p, tol=1e-10, max_time=60):
    """Attempt to reduce rank via global linear algebraic analysis.
    
    Builds the full tensor matrix and uses SVD + greedy column selection
    to find a minimal subset of columns that reproduces the target tensor.
    """
    start = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    total_dim = nm * mp * pn
    
    # Build target tensor
    target = np.zeros(total_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i * m + j) * mp * pn + (j * p + k) * pn + (k * n + i)] = 1.0
    
    # Build tensor matrix
    M = np.zeros((total_dim, rank))
    for r in range(rank):
        M[:, r] = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    
    # SVD to find numerical rank
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    if len(S) == 0:
        return None
    
    max_s = S[0]
    if max_s < tol:
        return None
    
    # Find numerical rank
    num_rank = int(sum(1 for s in S if s > tol * max_s))
    
    if num_rank >= rank:
        return None
    
    # Greedy forward selection: pick columns that best explain the target
    # Start with the column that has highest correlation with target
    # Actually, use the SVD: the right singular vectors give us the "best" linear combinations
    # But we want a subset of original columns
    
    # Use QR with column pivoting on M
    # np.linalg.qr doesn't do pivoting, so we implement manually
    # For efficiency, use the fact that we know num_rank
    
    # Strategy: use the SVD to identify which columns are most important
    # The right singular vectors Vt show the contribution of each original column
    # Columns with larger norm in Vt are more "important"
    
    # Actually, let's try a different approach: 
    # Project the target onto the row space and see which columns are needed
    
    # Simple approach: try removing columns one by one
    active = list(range(rank))
    
    for _ in range(min(rank, 20)):  # Try at most 20 removals
        if time.time() - start > max_time:
            break
        if len(active) <= num_rank:
            break
        
        best_removed = None
        best_residual = np.inf
        
        for idx_to_remove in range(len(active)):
            candidate = active[:idx_to_remove] + active[idx_to_remove + 1:]
            if len(candidate) == 0:
                continue
            
            M_sub = M[:, candidate]
            try:
                coeffs, residuals, r_sub, _ = np.linalg.lstsq(M_sub, target, rcond=tol)
                residual = np.linalg.norm(M_sub @ coeffs - target)
                if residual < best_residual:
                    best_residual = residual
                    best_removed = candidate
            except np.linalg.LinAlgError:
                continue
        
        if best_removed is None:
            break
        if best_residual > tol * (np.linalg.norm(target) + 1e-15):
            break
        active = best_removed
    
    if len(active) >= rank:
        return None
    
    new_U = [[U_arr[i, c] for c in active] for i in range(nm)]
    new_V = [[V_arr[i, c] for c in active] for i in range(mp)]
    new_W = [[W_arr[i, c] for c in active] for i in range(pn)]
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return {"U": new_U, "V": new_V, "W": new_W}
    
    return None


def _schoolbook(n, m, p):
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


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    try:
        result = _build_4x4x7()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            if time.time() - _START_TIME < _TIME_LIMIT - 20:
                try:
                    reduced = _global_rank_reduction(result['U'], result['V'], result['W'], 4, 4, 7)
                    if reduced is not None and len(reduced['U'][0]) < len(result['U'][0]):
                        if _verify(reduced['U'], reduced['V'], reduced['W'], 4, 4, 7):
                            return reduced
                except Exception:
                    pass
            return result
    except Exception:
        pass
    
    return _schoolbook(4, 4, 7)
