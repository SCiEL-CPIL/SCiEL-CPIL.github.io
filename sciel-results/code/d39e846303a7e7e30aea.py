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
    W[2][1] = 1  # C10 (k=0, i=1)
    W[3][1] = -1 # C11 (k=1, i=1)
    
    # M3: a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[1][2] = 1  # C01 (k=1, i=0)
    W[3][2] = 1  # C11 (k=1, i=1)
    
    # M4: a11(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00 (k=0, i=0)
    W[2][3] = 1  # C10 (k=0, i=1)
    
    # M5: (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # C00 (k=0, i=0)
    W[1][4] = 1  # C01 (k=1, i=0)
    
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
    W[1][6] = 1  # C01 (k=1, i=0)
    
    return U, V, W

def _decomp_2x2x4():
    """
    Return a rank-7 decomposition for 2x2 x 2x4.
    This is constructed by replicating the Strassen 2x2x2 decomposition
    for each pair of columns in the 4-column matrix.
    U: (4, 7), V: (8, 7), W: (8, 7)
    """
    U2, V2, W2 = _strassen_2x2()
    
    n, m, p = 2, 2, 4
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U is same as Strassen U
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                U[i*m+j][r] = U2[i*m+j][r]
                
    # V[j, k, r] = V2[j, k%2, r]
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                V[j*p+k][r] = V2[j*2 + (k % 2)][r]
                
    # W[k, i, r] = W2[k%2, i, r]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k*n+i][r] = W2[(k % 2)*n + i][r]
                
    return U, V, W

def _kronecker_4x4x8():
    """
    Construct decomposition for 4x4 x 4x8 using Kronecker product
    of 2x2x4 (rank 7) and 2x2x2 (rank 7).
    Total rank 49.
    """
    n1, m1, p1 = 2, 2, 4
    n2, m2, p2 = 2, 2, 2
    
    U1, V1, W1 = _decomp_2x2x4()
    U2, V2, W2 = _strassen_2x2()
    
    r1 = 7
    r2 = 7
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i1*n2+i2, j1*m2+j2, r1*r2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_U = (i1*n2 + i2) * m + (j1*m2 + j2)
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
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_V = (j1*m2 + j2) * p + (k1*p2 + k2)
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
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_W = (k1*p2 + k2) * n + (i1*n2 + i2)
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
    deadline = start_time + 140.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try Kronecker decomposition: (2x2x4) x (2x2x2), rank = 7*7 = 49
    kron_result = _kronecker_4x4x8()
    if kron_result is not None:
        kron_rank = len(kron_result["U"][0])
        if kron_rank < best_rank:
            if _verify(kron_result["U"], kron_result["V"], kron_result["W"], n, m, p):
                best = kron_result
                best_rank = kron_rank
    
    return best
# EVOLVE-BLOCK-END
