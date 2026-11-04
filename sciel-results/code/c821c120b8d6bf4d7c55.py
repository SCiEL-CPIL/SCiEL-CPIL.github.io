"""
Matrix multiplication 4x4x4 tensor decomposition.
Primary: rank-28 via Strassen block-level (7 terms) x schoolbook scalar-level (4 terms each).
Fallback: rank-49 via Kronecker product of Strassen, then schoolbook rank-64.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, r] = coeff of A[i,j] in term r
    V[j*2+k, r] = coeff of B[j,k] in term r
    W[k*2+i, r] = coeff of C[i,k] in term r (trace convention: W[k*n+i, r])
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C01, C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition: Strassen at block level (7 terms) x schoolbook at scalar level (4 terms).
    
    View 4x4 matrices as 2x2 blocks of 2x2 submatrices.
    At block level, use Strassen: 7 block multiplications.
    At scalar level within each block product, use schoolbook: 4 scalar multiplications.
    Total rank = 7 * 4 = 28.
    
    Indexing:
    - A[i,j] where i = 2*i1+i2, j = 2*j1+j2
    - B[j,k] where j = 2*j1+j2, k = 2*k1+k2
    - C[i,k] where i = 2*i1+i2, k = 2*k1+k2
    
    For Strassen term r1 and schoolbook sub-term s=(i2,j2,k2):
    - U index: i*m+j = (2*i1+i2)*4 + (2*j1+j2)
    - V index: j*p+k = (2*j1+j2)*4 + (2*k1+k2)
    - W index: k*n+i = (2*k1+k2)*4 + (2*i1+i2)
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for i2 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    s = i2 * 4 + j2 * 2 + k2
                    r = r1 * 4 + s
                    
                    for i1 in range(2):
                        for j1 in range(2):
                            for k1 in range(2):
                                # Full indices
                                i = 2 * i1 + i2
                                j = 2 * j1 + j2
                                k = 2 * k1 + k2
                                
                                # Block-level indices for Strassen
                                bi = i1  # block row of A
                                bj = j1  # block col of A = block row of B
                                bk = k1  # block col of B
                                
                                # Strassen factors (block level)
                                u_val = U2[bi * 2 + bj][r1]
                                v_val = V2[bj * 2 + bk][r1]
                                w_val = W2[bk * 2 + bi][r1]
                                
                                if u_val != 0 and v_val != 0 and w_val != 0:
                                    U[i * m + j][r] = u_val
                                    V[j * p + k][r] = v_val
                                    W[k * n + i][r] = w_val
                                else:
                                    # At least one factor is zero, so this term doesn't contribute
                                    pass
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _kronecker_decomposition(n, m, p):
    """
    Rank-49 decomposition via Kronecker product of Strassen 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    
                                    u_val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                    v_val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    w_val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                    
                                    if u_val != 0:
                                        U[i * m + j][r] = u_val
                                    if v_val != 0:
                                        V[j * p + k][r] = v_val
                                    if w_val != 0:
                                        W[k * n + i][r] = w_val
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _schoolbook_decomposition(n, m, p):
    """Rank-64 schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Try rank-28 first (best score)
    try:
        result28 = _rank28_decomposition(n, m, p)
        valid28, diff28 = _verify_decomp(n, m, p, result28["U"], result28["V"], result28["W"])
        if valid28:
            return {"U": result28["U"], "V": result28["V"], "W": result28["W"]}
    except Exception:
        pass
    
    if time.time() > deadline:
        result64 = _schoolbook_decomposition(n, m, p)
        return {"U": result64["U"], "V": result64["V"], "W": result64["W"]}
    
    # Try rank-49 (Kronecker product of Strassen)
    try:
        result49 = _kronecker_decomposition(n, m, p)
        valid49, diff49 = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"])
        if valid49:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
    except Exception:
        pass
    
    if time.time() > deadline:
        result64 = _schoolbook_decomposition(n, m, p)
        return {"U": result64["U"], "V": result64["V"], "W": result64["W"]}
    
    # Final fallback: schoolbook
    result64 = _schoolbook_decomposition(n, m, p)
    return {"U": result64["U"], "V": result64["V"], "W": result64["W"]}
# EVOLVE-BLOCK-END
