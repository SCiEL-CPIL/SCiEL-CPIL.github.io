import numpy as np
import time

def _strassen_2x2():
    """Returns U, V, W for 2x2 Strassen (Rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4 Strassen (Rank 49).
    Uses recursive Strassen: 4x4 = 2x2 blocks, each 2x2 multiplication uses rank 7.
    Total rank = 7 * 7 = 49.
    U, V, W shapes: (16, 49)
    Indexing: row = i*2 + a where i is block row (0,1), a is within-block row (0,1)
    """
    U2, V2, W2 = _strassen_2x2()
    R = 7
    U = np.zeros((16, R * R))
    V = np.zeros((16, R * R))
    W = np.zeros((16, R * R))
    
    for r1 in range(R):
        for r2 in range(R):
            r = r1 * R + r2
            # U: block (i, a) where i in {0,1}, a in {0,1}
            for i in range(2):
                for a in range(2):
                    U[i*2 + a, r] = U2[i, r1] * U2[a, r2]
            for j in range(2):
                for b in range(2):
                    V[j*2 + b, r] = V2[j, r1] * V2[b, r2]
            for k in range(2):
                for c in range(2):
                    W[k*2 + c, r] = W2[k, r1] * W2[c, r2]
    return U, V, W

def _build_4plus2_decomposition():
    """
    Build a decomposition for 6x6 using 4+2 block structure.
    6x6 matrix = [[A4x4, B4x2], [C2x4, D2x2]]
    
    This is complex; instead use the Kronecker product of Strassen(4x4) and Strassen(2x2)
    which gives rank 49 * 7 = 343. That's worse than 189.
    
    Better: Use the 3x3 grid of 2x2 blocks with Strassen on each, but with a smarter
    combination. The current approach (3x3 schoolbook ⊗ 2x2 Strassen) gives 189.
    
    Alternative structural approach: Use a direct 6x6 Strassen-like algorithm.
    For now, return the known 189-rank decomposition.
    """
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j, r] = 1
                V3[j*3 + k, r] = 1
                W3[k*3 + i, r] = 1
                r += 1
    
    U2, V2, W2 = _strassen_2x2()
    R3, R2 = 27, 7
    R_total = R3 * R2
    
    U = np.zeros((36, R_total))
    V = np.zeros((36, R_total))
    W = np.zeros((36, R_total))
    
    for r3 in range(R3):
        for r2 in range(R2):
            r = r3 * R2 + r2
            for i in range(3):
                for a in range(2):
                    U[i*2 + a, r] = U3[i, r3] * U2[a, r2]
            for j in range(3):
                for b in range(2):
                    V[j*2 + b, r] = V3[j, r3] * V2[b, r2]
            for k in range(3):
                for c in range(2):
                    W[k*2 + c, r] = W3[k, r3] * W2[c, r2]
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify the decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(42)
    for _ in range(150):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    U, V, W = _build_4plus2_decomposition()
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook 216
    U_sb = np.zeros((36, 216))
    V_sb = np.zeros((36, 216))
    W_sb = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*6+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
