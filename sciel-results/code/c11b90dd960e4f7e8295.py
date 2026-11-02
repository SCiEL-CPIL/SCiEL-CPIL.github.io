"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor using NumPy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2():
    """Return Strassen's 7-term decomposition for 2x2 x 2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _schoolbook_1x1x2():
    """Schoolbook 1x1x2: rank 2."""
    n, m, p = 1, 1, 2
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _kronecker(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two matrix multiplication decompositions.
    (n1,m1,p1) x (n2,m2,p2) -> (n1*n2, m1*m2, p1*p2)
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U[i*m+j, r1*r2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # where i = i1*n2 + i2, j = j1*m2 + j2
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_U = (i1*n2 + i2) * m + (j1*m2 + j2)
                    for r1i in range(r1):
                        v1 = U1[i1*m1 + j1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = U2[i2*m2 + j2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            U[row_U][col] = v1 * v2
    
    # V[j*p+k, r] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
    # where j = j1*m2 + j2, k = k1*p2 + k2
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_V = (j1*m2 + j2) * p + (k1*p2 + k2)
                    for r1i in range(r1):
                        v1 = V1[j1*p1 + k1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = V2[j2*p2 + k2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            V[row_V][col] = v1 * v2
    
    # W[k*n+i, r] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    # where k = k1*p2 + k2, i = i1*n2 + i2
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_W = (k1*p2 + k2) * n + (i1*n2 + i2)
                    for r1i in range(r1):
                        v1 = W1[k1*n1 + i1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = W2[k2*n2 + i2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            W[row_W][col] = v1 * v2
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_448_direct():
    """
    Direct construction of 4x4x8 rank-98 decomposition using two 4x4x4 Strassen blocks.
    This is equivalent to Kronecker(2x2x2, 2x2x2) for each half of the 8 output columns.
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _strassen_2x2()
    d444 = _kronecker(U_s, V_s, W_s, U_s, V_s, W_s, 2, 2, 2, 2, 2, 2)
    
    # First half: k = 0..3, rank terms 0..48
    for r in range(49):
        for i in range(16):
            U[i][r] = d444["U"][i][r]
        for j in range(4):
            for k in range(4):
                V[j*8 + k][r] = d444["V"][j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*4 + i][r] = d444["W"][k*4 + i][r]
    
    # Second half: k = 4..7, rank terms 49..97
    for r in range(49):
        for i in range(16):
            U[i][49 + r] = d444["U"][i][r]
        for j in range(4):
            for k in range(4):
                V[j*8 + (4 + k)][49 + r] = d444["V"][j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[(4 + k)*4 + i][49 + r] = d444["W"][k*4 + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Start with schoolbook as fallback (rank 128)
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try direct 4x4x8 rank-98 construction
    if time.time() < deadline - 20:
        d_448 = _construct_448_direct()
        if _verify(d_448["U"], d_448["V"], d_448["W"], n, m, p):
            best = d_448
            best_rank = 98
    
    return best
# EVOLVE-BLOCK-END
