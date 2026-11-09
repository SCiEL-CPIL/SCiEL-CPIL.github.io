import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 5.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, j] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _verify_tensor(U, V, W, n, m, p):
    """Verify the tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,k}"""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    # Compute result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                expected[i, :, k] = 1.0
    return np.allclose(result, expected, atol=1e-9, rtol=0.0)


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _strassen_2x2_2x2():
    """
    Return a rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U shape: (4, 7) -> U[i*2+j, c]
    V shape: (4, 7) -> V[j*2+k, c]
    W shape: (4, 7) -> W[k*2+i, c]
    Identity: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,k}
    """
    n = m = p = 2
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    
    # Strassen's 7 multiplications for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[i,j,c] = coefficient of a[i,j] in M_c
    # V[j,k,c] = coefficient of b[j,k] in M_c
    # W[k,i,c] = coefficient of M_c in C[i,k] (note: C[i,k] = sum_c W[k,i,c]*M_c)
    
    # M1: (a00+a11)(b00+b11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1  # C[0,0] and C[1,1]
    
    # M2: (a10+a11)b00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1  # C[1,0]
    
    # M3: a00(b01-b11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+1, 2] = 1  # C[0,1] and C[1,1]
    
    # M4: a11(b10-b00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1  # C[0,0] and C[1,0]
    
    # M5: (a00+a01)b11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1  # C[0,0] -= M5, C[0,1] += M5
    
    # M6: (a10-a00)(b00+b01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1  # C[1,1]
    
    # M7: (a01-a11)(b10+b11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1  # C[0,0]
    
    return U, V, W


def _embed_2x2_to_3x3(U2, V2, W2, rank2, n=3, m=3, p=3):
    """
    Embed a rank-rank2 decomposition for 2x2*2x2 into a rank-rank2 decomposition for 3x3*3x3.
    The 3x3 identity is block-diagonal with a 2x2 identity block and a 1x1 identity.
    We need the extra 1x1 to be handled, so we add one more term for the (2,2) entry.
    Total rank = rank2 + 1.
    """
    R = rank2 + 1
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    
    # Embed the 2x2 part: indices 0,1 for rows/cols
    for i in range(2):
        for j in range(2):
            for c in range(rank2):
                U[i*m+j, c] = U2[i*2+j, c]
            for k in range(2):
                for c in range(rank2):
                    V[j*p+k, c] = V2[j*2+k, c]
            for k in range(2):
                for c in range(rank2):
                    W[k*n+i, c] = W2[k*2+i, c]
    
    # Handle the (2,2) entry of the identity: C[2,2] = a[2,2]*b[2,2]
    U[2*m+2, rank2] = 1
    V[2*p+2, rank2] = 1
    W[2*n+2, rank2] = 1
    
    return U, V, W


def _build_3x6_from_3x3(U3, V3, W3, rank3):
    """
    Build a 3x3 * 3x6 decomposition from two 3x3 * 3x3 decompositions.
    The 6-dimensional output is split into two 3-dimensional blocks.
    Total rank = 2 * rank3.
    """
    n, m, p = 3, 3, 6
    R = 2 * rank3
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for q in range(2):
        base = q * rank3
        for c in range(rank3):
            col = base + c
            # U is the same for both blocks
            for i in range(n):
                for k in range(m):
                    U[i*m+k, col] = U3[i*m+k, c]
            # V: block q uses columns q*3 to q*3+2 of the 6-column matrix
            for k in range(m):
                for j in range(3):
                    V[k*p + (q*3+j), col] = V3[k*3+j, c]
            # W: block q uses rows q*3 to q*3+2 of the 18-row W
            for j in range(3):
                for i in range(n):
                    W[(q*3+j)*n + i, col] = W3[j*n+i, c]
    
    return U, V, W


def _try_strassen_based():
    """Try to build a low-rank decomposition using Strassen for 2x2 blocks."""
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Verify the 2x2 Strassen decomposition
    if not _verify_tensor(U2, V2, W2, 2, 2, 2):
        return None
    
    # Embed to 3x3
    U3, V3, W3 = _embed_2x2_to_3x3(U2, V2, W2, 7)
    
    # Verify the 3x3 decomposition
    if not _verify_tensor(U3, V3, W3, 3, 3, 3):
        return None
    
    # Build 3x3 * 3x6
    U, V, W = _build_3x6_from_3x3(U3, V3, W3, 8)
    
    # Verify the full 3x6 decomposition
    if not _verify_tensor(U, V, W, 3, 3, 6):
        return None
    
    # Also verify with random matrices
    if not _verify_3x6(U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _find_3x3x2_decomp(rank, seed, trials=500):
    """Find rank decomposition for 3x3 * 3x2 matrix multiplication."""
    n, m, p = 3, 3, 2
    rng = np.random.default_rng(seed)
    
    for trial in range(trials):
        if not _ok():
            return None
        
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = rng.choice(vals, size=(rank, n, m))
        V = rng.choice(vals, size=(rank, m, p))
        
        W = np.zeros((rank, n, p))
        valid = True
        
        for i in range(n):
            for j in range(p):
                A = np.zeros((m * m, rank))
                b = np.zeros(m * m)
                for k in range(m):
                    for kp in range(m):
                        A[k * m + kp, :] = U[:, i, k] * V[:, kp, j]
                        b[k * m + kp] = 1.0 if k == kp else 0.0
                
                try:
                    W_col, *_ = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                W_col_rounded = np.round(W_col * 2) / 2
                W[:, i, j] = W_col_rounded
                
                if not np.allclose(A @ W_col_rounded, b, atol=1e-9, rtol=0.0):
                    valid = False
                    break
        
        if not valid:
            continue
        
        check = True
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    for kp in range(m):
                        s = sum(U[t, i, k] * V[t, kp, j] * W[t, i, j] for t in range(rank))
                        expected = 1.0 if k == kp else 0.0
                        if abs(s - expected) > 1e-9:
                            check = False
                            break
                    if not check:
                        break
                if not check:
                    break
            if not check:
                break
        
        if check:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None


def _build_3x6_from_3x3x2(r, d0, d1, d2):
    n, m, p = 3, 3, 6
    R = 3 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1, d2)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(2):
                    V[k * 6 + (q * 2 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(2):
                    W[(q * 2 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def construct():
    # Strategy 1: Strassen-based decomposition (rank 16 for 3x6)
    result = _try_strassen_based()
    if result is not None:
        return result
    
    # Strategy 2: Try to find 3x3x2 decompositions and combine
    for r in range(12, 7, -1):
        if not _ok():
            break
        d0 = _find_3x3x2_decomp(r, seed=1000 + r)
        if d0 is None:
            continue
        d1 = _find_3x3x2_decomp(r, seed=3000 + r)
        if d1 is None:
            continue
        d2 = _find_3x3x2_decomp(r, seed=5000 + r)
        if d2 is None:
            continue
        U, V, W = _build_3x6_from_3x3x2(r, d0, d1, d2)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook()
