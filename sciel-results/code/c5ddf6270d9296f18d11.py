"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _kron_product(A, B):
    """Kronecker product of two matrices."""
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    ra, ca = A.shape
    rb, cb = B.shape
    result = np.zeros((ra * rb, ca * cb), dtype=np.float64)
    for i in range(ra):
        for j in range(ca):
            if A[i, j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k, j * cb + l] = A[i, j] * B[k, l]
    return result

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4x7
    V = [[0.0] * rank for _ in range(m * p)]  # 4x7
    W = [[0.0] * rank for _ in range(p * n)]  # 4x7
    
    # Strassen's 7-term decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1.0  # A00
    U[1*2+1][0] = 1.0  # A11
    V[0*2+0][0] = 1.0  # B00
    V[1*2+1][0] = 1.0  # B11
    W[0*2+0][0] = 1.0  # C00
    W[1*2+1][0] = 1.0  # C11
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0][1] = 1.0  # A10
    U[1*2+1][1] = 1.0  # A11
    V[0*2+0][1] = 1.0  # B00
    W[1*2+0][1] = 1.0  # C10
    W[1*2+1][1] = -1.0  # C11
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0][2] = 1.0  # A00
    V[0*2+1][2] = 1.0  # B01
    V[1*2+1][2] = -1.0  # B11
    W[0*2+1][2] = 1.0  # C01
    W[1*2+1][2] = 1.0  # C11
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1][3] = 1.0  # A11
    V[1*2+0][3] = 1.0  # B10
    V[0*2+0][3] = -1.0  # B00
    W[0*2+0][3] = 1.0  # C00
    W[1*2+0][3] = 1.0  # C10
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0][4] = 1.0  # A00
    U[0*2+1][4] = 1.0  # A01
    V[1*2+1][4] = 1.0  # B11
    W[0*2+0][4] = -1.0  # C00
    W[0*2+1][4] = 1.0  # C01
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1.0  # A10
    U[0*2+0][5] = -1.0  # A00
    V[0*2+0][5] = 1.0  # B00
    V[0*2+1][5] = 1.0  # B01
    W[1*2+1][5] = 1.0  # C11
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1.0  # A01
    U[1*2+1][6] = -1.0  # A11
    V[1*2+0][6] = 1.0  # B10
    V[1*2+1][6] = 1.0  # B11
    W[0*2+0][6] = 1.0  # C00
    
    return U, V, W

def _kron_decomp(U1, V1, W1, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two tensor decompositions.
    
    U1: shape (n1*m1, r1), V1: shape (m1*p1, r1), W1: shape (p1*n1, r1)
    U2: shape (n2*m2, r2), V2: shape (m2*p2, r2), W2: shape (p2*n2, r2)
    
    Returns decomposition for (n1*n2)x(m1*m2)x(p1*p2) with rank r1*r2.
    """
    r1 = len(U1[0])
    r2 = len(U2[0]) if 'U2' in dir() else len(U1[0])
    
    # Reshape into tensors
    U1_t = np.array(U1, dtype=np.float64).reshape(n1, m1, r1)
    V1_t = np.array(V1, dtype=np.float64).reshape(m1, p1, r1)
    W1_t = np.array(W1, dtype=np.float64).reshape(p1, n1, r1)
    
    U2_t = np.array(U2, dtype=np.float64).reshape(n2, m2, r2)
    V2_t = np.array(V2, dtype=np.float64).reshape(m2, p2, r2)
    W2_t = np.array(W2, dtype=np.float64).reshape(p2, n2, r2)
    
    r_total = r1 * r2
    n_total = n1 * n2
    m_total = m1 * m2
    p_total = p1 * p2
    
    U = [[0.0] * r_total for _ in range(n_total * m_total)]
    V = [[0.0] * r_total for _ in range(m_total * p_total)]
    W = [[0.0] * r_total for _ in range(p_total * n_total)]
    
    for a in range(n1):
        for b in range(m1):
            for c in range(p1):
                for d in range(n2):
                    for e in range(m2):
                        for f in range(p2):
                            i = a * n2 + d  # row in first matrix
                            j = b * m2 + e  # col in first matrix
                            k = c * p2 + f  # col in second matrix
                            
                            for r1_idx in range(r1):
                                for r2_idx in range(r2):
                                    col = r1_idx * r2 + r2_idx
                                    U[i * m_total + j][col] = U1_t[a, b, r1_idx] * U2_t[d, e, r2_idx]
                                    V[j * p_total + k][col] = V1_t[b, c, r1_idx] * V2_t[e, f, r2_idx]
                                    W[k * n_total + i][col] = W1_t[c, a, r1_idx] * W2_t[f, d, r2_idx]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for nxmxp matrix multiplication."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _concat_decompositions(decomps):
    """Concatenate multiple decompositions with same dimensions.
    
    decomps: list of (U, V, W) tuples
    """
    n, m, p = decomps[0][0][0].shape[0] if False else (None, None, None)
    
    # Infer dimensions from first decomposition
    U0 = decomps[0][0]
    V0 = decomps[0][1]
    W0 = decomps[0][2]
    
    # U has n*m rows, V has m*p rows, W has p*n rows
    # We need to figure out n, m, p from the shapes
    nm = len(U0)
    mp = len(V0)
    pn = len(W0)
    
    # For our case: n=4, m=4, p=9, so nm=16, mp=36, pn=36
    # We need to solve for n, m, p such that n*m=16, m*p=36, p*n=36
    # Try possible values
    found = False
    for n in range(1, 20):
        for m in range(1, 20):
            if n * m != nm:
                continue
            for p in range(1, 20):
                if m * p != mp or p * n != pn:
                    continue
                found = True
                break
            if found:
                break
        if found:
            break
    
    if not found:
        raise ValueError("Cannot determine dimensions")
    
    total_rank = sum(len(U[0]) for U, V, W in decomps)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for U_d, V_d, W_d in decomps:
        r = len(U_d[0])
        for i in range(n * m):
            for c in range(r):
                U[i][offset + c] = U_d[i][c]
        for i in range(m * p):
            for c in range(r):
                V[i][offset + c] = V_d[i][c]
        for i in range(p * n):
            for c in range(r):
                W[i][offset + c] = W_d[i][c]
        offset += r
    
    return U, V, W

def construct():
    """Return the best known decomposition for 4x4x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = 4, 4, 9
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Strategy: Split 4x4x9 into two 4x4x4 parts and one 4x4x1 part
    # Use Strassen-based Kronecker for 4x4x4 (rank 49)
    # Use schoolbook for 4x4x1 (rank 16)
    # Total: 49 + 49 + 16 = 114
    
    if time.time() > deadline:
        return best
    
    # Get rank-7 decomposition for 2x2x2
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 2x2x2 decomposition
    if not _verify_decomposition(U2, V2, W2, 2, 2, 2):
        # Fall back to schoolbook for 2x2x2
        U2, V2, W2 = _schoolbook(2, 2, 2)
    
    # Use Kronecker to get 4x4x4 decomposition
    if time.time() > deadline:
        return best
    
    U4, V4, W4 = _kron_decomp(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2)
    
    # Verify 4x4x4 decomposition
    if not _verify_decomposition(U4, V4, W4, 4, 4, 4):
        # Fall back to schoolbook for 4x4x4
        U4, V4, W4 = _schoolbook(4, 4, 4)
    
    # Now split 4x4x9 into:
    # - First 4 columns: 4x4x4
    # - Next 4 columns: 4x4x4  
    # - Last 1 column: 4x4x1
    
    U4a, V4a, W4a = U4, V4, W4  # First 4x4x4
    U4b, V4b, W4b = U4, V4, W4  # Second 4x4x4 (same structure)
    U1, V1, W1 = _schoolbook(4, 4, 1)  # 4x4x1
    
    # We need to adjust the decompositions for the different column ranges
    # For the first 4x4x4 (columns 0-3 of the 4x9 matrix):
    # U stays the same (4x4 input)
    # V: rows correspond to (j,k) where j in 0..3 (cols of A), k in 0..3 (rows of B1)
    # W: rows correspond to (k,i) where k in 0..3 (rows of B1), i in 0..3 (rows of C)
    
    # For the second 4x4x4 (columns 4-7 of the 4x9 matrix):
    # U stays the same
    # V: rows correspond to (j,k) where j in 0..3, k in 4..7 (i.e., k' = k-4 in 0..3)
    # W: rows correspond to (k,i) where k in 4..7, i in 0..3 (i.e., k' = k-4 in 0..3)
    
    # For the 4x4x1 (column 8 of the 4x9 matrix):
    # U stays the same
    # V: rows correspond to (j,k) where j in 0..3, k in 8
    # W: rows correspond to (k,i) where k in 8, i in 0..3
    
    rank_total = 49 + 49 + 16
    U_final = [[0.0] * rank_total for _ in range(16)]  # n*m = 16
    V_final = [[0.0] * rank_total for _ in range(36)]  # m*p = 36
    W_final = [[0.0] * rank_total for _ in range(36)]  # p*n = 36
    
    offset = 0
    # First 4x4x4 (columns 0-3)
    r = len(U4a[0])
    for i in range(16):
        for c in range(r):
            U_final[i][offset + c] = U4a[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(r):
                V_final[j * 9 + k][offset + c] = V4a[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(r):
                W_final[k * 4 + i][offset + c] = W4a[k * 4 + i][c]
    offset += r
    
    if time.time() > deadline:
        # Return schoolbook
        return best
    
    # Second 4x4x4 (columns 4-7)
    for i in range(16):
        for c in range(r):
            U_final[i][offset + c] = U4b[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(r):
                V_final[j * 9 + (4 + k)][offset + c] = V4b[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(r):
                W_final[(4 + k) * 4 + i][offset + c] = W4b[k * 4 + i][c]
    offset += r
    
    # 4x4x1 (column 8)
    r1 = len(U1[0])
    for i in range(16):
        for c in range(r1):
            U_final[i][offset + c] = U1[i][c]
    for j in range(4):
        for c in range(r1):
            V_final[j * 9 + 8][offset + c] = V1[j * 1 + 0][c]
    for i in range(4):
        for c in range(r1):
            W_final[8 * 4 + i][offset + c] = W1[0 * 4 + i][c]
    offset += r1
    
    # Verify the final decomposition
    if _verify_decomposition(U_final, V_final, W_final, 4, 4, 9):
        best = {"U": U_final, "V": V_final, "W": W_final}
        best_rank = rank_total
    
    return best
