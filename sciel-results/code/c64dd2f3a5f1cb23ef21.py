"""Matrix multiplication 4x4 x 4x6 via Kronecker product of Strassen and schoolbook."""
import time

DIMENSIONS = [4, 4, 6]

def _strassen_222():
    """Return (U, V, W) for 2x2 x 2x2 matrix mult, rank 7.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each rank term c, U[i*m+j, c] gives coefficient for A[i,j]
    # V[j*p+k, c] gives coefficient for B[j,k]
    # W[k*n+i, c] gives coefficient for C[i,k]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1  # C00, C11
    
    # Term 1: M2 = (A10+A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1  # C10
    
    # Term 2: M3 = A00(B10-B11)
    U[0*2+0][2] = 1
    V[1*2+0][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1  # C01
    
    # Term 3: M4 = A11(B01-B10)
    U[1*2+1][3] = 1
    V[0*2+1][3] = 1; V[1*2+0][3] = -1
    W[0*2+0][3] = 1; W[1*2+1][3] = 1  # C00, C11
    
    # Term 4: M5 = (A00+A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[1*2+1][4] = -1  # C00, C11 (with - sign)
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[1*2+0][5] = 1
    W[1*2+1][5] = 1  # C11
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[0*2+1][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = -1  # C00 (with - sign)
    
    return U, V, W

def _schoolbook_223():
    """Return (U, V, W) for 2x2 x 2x3 matrix mult, rank 12.
    U shape: (2*2, 12) = (4, 12)
    V shape: (2*3, 12) = (6, 12)
    W shape: (3*2, 12) = (6, 12)
    """
    n = 2; m = 2; p = 3
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

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions.
    U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
    Result: rank = r1*r2
    U: (n1*n2 * m1*m2, r1*r2)
    V: (m1*m2 * p1*p2, r1*r2)
    W: (p1*p2 * n1*n2, r1*r2)
    """
    # Determine dimensions from shapes
    n1m1, r1 = len(U1), len(U1[0])
    m1p1, _ = len(V1), len(V1[0])
    p1n1, _ = len(W1), len(W1[0])
    
    n2m2, r2 = len(U2), len(U2[0])
    m2p2, _ = len(V2), len(V2[0])
    p2n2, _ = len(W2), len(W2[0])
    
    # Infer n1, m1, p1
    # n1*m1 = n1m1, m1*p1 = m1p1, p1*n1 = p1n1
    # For our case: n1=2, m1=2, p1=2
    n1 = int(n1m1 ** 0.5)
    m1 = n1m1 // n1
    p1 = m1p1 // m1
    
    n2 = int(n2m2 ** 0.5)
    m2 = n2m2 // n2
    p2 = m2p2 // m2
    
    rank = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1*m1+j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2*m2+j2][c2]
                            if u2 == 0:
                                continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j][c] += u1 * u2
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1*p1+k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2*p2+k2][c2]
                            if v2 == 0:
                                continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k][c] += v1 * v2
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W1[k1*n1+i1][c1]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2*n2+i2][c2]
                            if w2 == 0:
                                continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i][c] += w1 * w2
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _schoolbook_223()
    
    U, V, W = _kron_decomp(U1, V1, W1, U2, V2, W2)
    
    # Convert to clean integer values
    U = [[round(x) for x in row] for row in U]
    V = [[round(x) for x in row] for row in V]
    W = [[round(x) for x in row] for row in W]
    
    # Remove zero columns if any (shouldn't happen with this construction)
    # But let's verify rank
    rank = len(U[0])
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # This shouldn't happen for this fast computation
        pass
    
    return {"U": U, "V": V, "W": W}
