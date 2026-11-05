"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    Ur = U.reshape(n, m, r) if isinstance(U, np.ndarray) else np.array(U).reshape(n, m, r)
    Vr = V.reshape(m, p, r) if isinstance(V, np.ndarray) else np.array(V).reshape(m, p, r)
    Wr = W.reshape(p, n, r) if isinstance(W, np.ndarray) else np.array(W).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """
    Verified rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0] and C[1,1]
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1] and C[1,1]
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0] and C[0,1]
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _schoolbook_3x3():
    """
    Schoolbook rank-27 decomposition for 3x3x3 matrix multiplication.
    U: (9,27), V: (9,27), W: (9,27)
    """
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[k * 3 + i, col] = 1.0
                col += 1
    return U, V, W


def _liu_3x3():
    """
    Attempt to construct a rank-19 decomposition for 3x3 matrix multiplication.
    If a valid rank-19 decomposition is found and verified, return it.
    Otherwise, return None.
    """
    # The rank of 3x3 matrix multiplication is known to be 19 over the reals.
    # Constructing the explicit tensor factors for the rank-19 algorithm is complex.
    # We will attempt to generate a candidate and verify it. If it fails, we return None.
    
    # A known rank-19 algorithm exists, but writing it out explicitly is error-prone.
    # Instead, we can try to find a decomposition via optimization or use a known one.
    # For now, we return None to indicate we don't have a verified rank-19 decomposition ready.
    # In a future iteration, we could implement the specific rank-19 algorithm.
    
    return None


def _kronecker_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compute the Kronecker product of two tensor decompositions.
    
    If (U1, V1, W1) decomposes n1 x m1 x p1 matrix multiplication with rank r1,
    and (U2, V2, W2) decomposes n2 x m2 x p2 matrix multiplication with rank r2,
    then the Kronecker product decomposes (n1*n2) x (m1*m2) x (p1*p2)
    matrix multiplication with rank r1 * r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    U = np.zeros((n1 * n2 * m1 * m2, rank), dtype=float)
    V = np.zeros((m1 * m2 * p1 * p2, rank), dtype=float)
    W = np.zeros((p1 * p2 * n1 * n2, rank), dtype=float)
    
    # U: index (i1*n2+i2)*(m1*m2) + (j1*m2+j2), column c1*r2+c2
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1_val = U1[i1 * m1 + j1, c1]
                if u1_val == 0:
                    continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u1_val != 0 and u2_val != 0:
                                row = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                                col = c1 * r2 + c2
                                U[row, col] = u1_val * u2_val
    
    # V: index (j1*m2+j2)*(p1*p2) + (k1*p2+k2), column c1*r2+c2
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1_val = V1[j1 * p1 + k1, c1]
                if v1_val == 0:
                    continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v1_val != 0 and v2_val != 0:
                                row = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                col = c1 * r2 + c2
                                V[row, col] = v1_val * v2_val
    
    # W: index (k1*p2+k2)*(n1*n2) + (i1*n2+i2), column c1*r2+c2
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1_val = W1[k1 * n1 + i1, c1]
                if w1_val == 0:
                    continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w1_val != 0 and w2_val != 0:
                                row = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                col = c1 * r2 + c2
                                W[row, col] = w1_val * w2_val
    
    return U, V, W


def _block_strassen_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    
                    col += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27).
    Total rank = 7 * 27 = 189.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: Kronecker product of Strassen 2x2 and schoolbook 3x3
    sU, sV, sW = _strassen_2x2()
    kU, kV, kW = _schoolbook_3x3()
    U_k, V_k, W_k = _kronecker_product(sU, sV, sW, kU, kV, kW, 2, 2, 2, 3, 3, 3)
    
    # Verify quickly
    if _verify(U_k, V_k, W_k, 6, 6, 6):
        return {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    
    # Fallback: block Strassen
    U_b, V_b, W_b = _block_strassen_6x6x6()
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
