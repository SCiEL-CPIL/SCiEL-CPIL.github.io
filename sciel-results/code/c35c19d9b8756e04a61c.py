"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # C00 (k=0, i=0)
    W[3][0] = 1  # C11 (k=1, i=1)
    
    # M2: (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[1][1] = 1  # C10 (k=0, i=1)
    W[3][1] = -1 # C11 (k=1, i=1)
    
    # M3: a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[2][2] = 1  # C01 (k=1, i=0)
    W[3][2] = 1  # C11 (k=1, i=1)
    
    # M4: a11(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00 (k=0, i=0)
    W[1][3] = 1  # C10 (k=0, i=1)
    
    # M5: (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # C00 (k=0, i=0)
    W[2][4] = 1  # C01 (k=1, i=0)
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # C11 (k=1, i=1)
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # C00 (k=0, i=0)
    W[2][6] = 1  # C01 (k=1, i=0)
    
    return U, V, W

def _schoolbook_2x2x4():
    """
    Return schoolbook decomposition for 2x2 x 2x4.
    n=2, m=2, p=4. Rank = 16.
    U: (4, 16), V: (8, 16), W: (8, 16)
    """
    n, m, p = 2, 2, 4
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
    return U, V, W

def _kronecker_decompose(n, m, p):
    """
    Construct a decomposition for n x m * m x p using Kronecker product of
    a 2x2x2 Strassen decomposition and a 2x2x4 schoolbook decomposition.
    
    We view the matrices as:
    A (n x m) = A1 (2 x 2) kron A2 (2 x 2) ? No, dimensions don't match directly for simple kron.
    
    Let's map indices:
    i = 2*i1 + i2, where i1 in {0,1}, i2 in {0,1}
    j = 2*j1 + j2, where j1 in {0,1}, j2 in {0,1}
    k = 2*k1 + k2, where k1 in {0,1}, k2 in {0,1,2,3}
    
    Wait, p=8. So k = 2*k1 + k2 where k1 in {0,1}, k2 in {0,1,2,3}.
    This corresponds to a block structure where the 'inner' dimension is size 2 for i and j, and size 4 for k?
    
    Let's use the standard Kronecker tensor product definition.
    If T1 is a decomposition of (n1, m1, p1) with rank r1 and T2 is a decomposition of (n2, m2, p2) with rank r2,
    then T1 x T2 is a decomposition of (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    Here we want (4, 4, 8).
    We can write 4 = 2*2, 4 = 2*2, 8 = 2*4.
    So we combine T_Strassen (2,2,2) rank 7 and T_Schoolbook (2,2,4) rank 16.
    Resulting dims: (2*2, 2*2, 2*4) = (4, 4, 8).
    Total rank: 7 * 16 = 112.
    """
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 4
    
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _schoolbook_2x2x4()
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n1*n2 * m1*m2)]
    V = [[0] * rank for _ in range(m1*m2 * p1*p2)]
    W = [[0] * rank for _ in range(p1*p2 * n1*n2)]
    
    # U[i1*n2+i2, j1*m2+j2, r1*r2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # Note: Index in U is i*m + j. Here i = i1*n2+i2, j = j1*m2+j2.
    # So row index = (i1*n2+i2)*m + (j1*m2+j2) where m = m1*m2.
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_U = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                    for r1_idx in range(r1):
                        val1 = U1[i1*m1 + j1][r1_idx]
                        if val1 == 0:
                            continue
                        for r2_idx in range(r2):
                            val2 = U2[i2*m2 + j2][r2_idx]
                            if val2 == 0:
                                continue
                            col = r1_idx * r2 + r2_idx
                            U[row_U][col] = val1 * val2
    
    # V[j1*m2+j2, k1*p2+k2, r1*r2+r2] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
    # Row index in V is j*p + k. Here j = j1*m2+j2, k = k1*p2+k2.
    # p = p1*p2.
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_V = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                    for r1_idx in range(r1):
                        val1 = V1[j1*p1 + k1][r1_idx]
                        if val1 == 0:
                            continue
                        for r2_idx in range(r2):
                            val2 = V2[j2*p2 + k2][r2_idx]
                            if val2 == 0:
                                continue
                            col = r1_idx * r2 + r2_idx
                            V[row_V][col] = val1 * val2
    
    # W[k1*p2+k2, i1*n2+i2, r1*r2+r2] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    # Row index in W is k*n + i. Here k = k1*p2+k2, i = i1*n2+i2.
    # n = n1*n2.
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_W = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                    for r1_idx in range(r1):
                        val1 = W1[k1*n1 + i1][r1_idx]
                        if val1 == 0:
                            continue
                        for r2_idx in range(r2):
                            val2 = W2[k2*n2 + i2][r2_idx]
                            if val2 == 0:
                                continue
                            col = r1_idx * r2 + r2_idx
                            W[row_W][col] = val1 * val2
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try Kronecker decomposition: (2x2x2) x (2x2x4), rank = 7*16 = 112
    kron_result = _kronecker_decompose(n, m, p)
    if kron_result is not None:
        kron_rank = len(kron_result["U"][0])
        if kron_rank < best_rank:
            # Verify the decomposition
            if _verify(kron_result["U"], kron_result["V"], kron_result["W"], n, m, p):
                best = kron_result
                best_rank = kron_rank
    
    return best
# EVOLVE-BLOCK-END
