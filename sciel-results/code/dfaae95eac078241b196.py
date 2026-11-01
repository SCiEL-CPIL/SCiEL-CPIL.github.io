"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Returns (U, V, W) for 2x2x2 Strassen, shapes (4,7), (4,7), (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a10+a11)b00 -> c10 += m2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # m3 = a00(b10-b00) -> c01 += m3, c11 -= m3
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # m4 = a11(b01-b11) -> c10 += m4, c00 -= m4
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[1, 3] = 1; W[0, 3] = -1
    
    # m5 = (a00+a01)b11 -> c01 += m5, c00 -= m5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # m6 = (a10-a00)b01 -> c11 += m6
    U[2, 5] = 1; U[0, 5] = -1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (a01-a11)b10 -> c00 += m7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_221():
    """Returns (U, V, W) for 2x2x1 schoolbook, shapes (4,2), (2,2), (2,2)."""
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    # c=0: a00*b00 -> c00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    # c=1: a10*b00 -> c10
    U[2, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    return U, V, W

def _combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two decompositions that share U but have disjoint supports in V (p) and W (p).
    U1: (n*m, r1), V1: (m*p1, r1), W1: (p1*n, r1)
    U2: (n*m, r2), V2: (m*p2, r2), W2: (p2*n, r2)
    Returns U: (n*m, r1+r2), V: (m*(p1+p2), r1+r2), W: ((p1+p2)*n, r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    p = p1 + p2
    U = np.zeros((n * m, r1 + r2))
    V = np.zeros((m * p, r1 + r2))
    W = np.zeros((p * n, r1 + r2))
    
    U[:, 0:r1] = U1
    U[:, r1:r1+r2] = U2
    
    # V mapping: V[j*p+k, c]
    # For k < p1: V[j*p1+k, c] = V1[j*p1+k, c]
    # For k >= p1: V[j*p1+(k-p1), c] = V2[j*p2+(k-p1), c+r1]
    for j in range(m):
        for k in range(p1):
            V[j*p + k, 0:r1] = V1[j*p1 + k, 0:r1]
        for k in range(p2):
            V[j*p + (k+p1), r1:r1+r2] = V2[j*p2 + k, 0:r2]
            
    # W mapping: W[k*n+i, c]
    for k in range(p1):
        for i in range(n):
            W[k*n + i, 0:r1] = W1[k*n + i, 0:r1]
    for k in range(p2):
        for i in range(n):
            W[(k+p1)*n + i, r1:r1+r2] = W2[k*n + i, 0:r2]
            
    return U, V, W

def _combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two decompositions that share U (but U is split in m) and have disjoint supports in V (m) and W (none, W is same p*n).
    Actually, for C = A1*B1 + A2*B2, the W parts add up.
    U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
    U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
    Returns U: (n*(m1+m2), r1+r2), V: ((m1+m2)*p, r1+r2), W: (p*n, r1+r2)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    m = m1 + m2
    U = np.zeros((n * m, r1 + r2))
    V = np.zeros((m * p, r1 + r2))
    W = np.zeros((p * n, r1 + r2))
    
    # U mapping: U[i*m+j, c]
    # For j < m1: U[i*m+j, c] = U1[i*m1+j, c]
    # For j >= m1: U[i*m+(j-m1), c] = U2[i*m2+(j-m1), c+r1]
    for i in range(n):
        for j in range(m1):
            U[i*m + j, 0:r1] = U1[i*m1 + j, 0:r1]
        for j in range(m2):
            U[i*m + (j+m1), r1:r1+r2] = U2[i*m2 + j, 0:r2]
            
    # V mapping: V[j*p+k, c]
    for j in range(m1):
        for k in range(p):
            V[j*p + k, 0:r1] = V1[j*p1 + k, 0:r1] # Wait, V1 is m1*p. V1[j*p+k]
            # Correction: V1 index is j*p+k. Here p is same.
            # V1 shape (m1*p, r1).
            # V2 shape (m2*p, r2).
            pass
    
    # Let's do it simpler.
    V[:, 0:r1] = 0
    V[:, r1:r1+r2] = 0
    V[0:m1*p, 0:r1] = V1
    V[m1*p:m*p, r1:r1+r2] = V2
    
    W[:, 0:r1] = W1
    W[:, r1:r1+r2] = W2
    
    return U, V, W

def _construct_rank50():
    """
    Constructs rank 50 decomposition for 2x4x8.
    Structure:
    1. Base 2x2x2 Strassen (rank 7).
    2. Base 2x2x1 Schoolbook (rank 2).
    3. 2x2x3 = 2x2x2 + 2x2x1 (rank 9).
    4. 2x2x5 = 2x2x2 + 2x2x3 (rank 16).
    5. 2x4x3 = 2x2x3 + 2x2x3 (rank 18).
    6. 2x4x5 = 2x2x5 + 2x2x5 (rank 32).
    7. 2x4x8 = 2x4x3 + 2x4x5 (rank 50).
    """
    n = 2
    
    # 1. 2x2x2 Strassen
    U_222, V_222, W_222 = _strassen_222()
    
    # 2. 2x2x1 Schoolbook
    U_221, V_221, W_221 = _schoolbook_221()
    
    # 3. 2x2x3 (Combine p=2 and p=1)
    U_223, V_223, W_223 = _combine_p(U_222, V_222, W_222, U_221, V_221, W_221, n=2, m=2, p1=2, p2=1)
    
    # 4. 2x2x5 (Combine p=2 and p=3)
    # We need U_222, V_222, W_222 (p=2) and U_223, V_223, W_223 (p=3)
    # Wait, the U for p=2 part and p=3 part must be compatible if we are just adding independent products?
    # No, _combine_p assumes U1 and U2 are independent columns in the rank space.
    # So U_225[:, 0:7] = U_222 and U_225[:, 7:16] = U_223.
    # This effectively computes A*B2 + A*B3.
    # This is valid for C = A * [B2, B3].
    U_225, V_225, W_225 = _combine_p(U_222, V_222, W_222, U_223, V_223, W_223, n=2, m=2, p1=2, p2=3)
    
    # 5. 2x4x3 (Combine m=2 and m=2)
    # C = A1*B1 + A2*B2.
    # We use two copies of 2x2x3.
    U_243, V_243, W_243 = _combine_m(U_223, V_223, W_223, U_223, V_223, W_223, n=2, m1=2, m2=2, p=3)
    
    # 6. 2x4x5 (Combine m=2 and m=2)
    # We use two copies of 2x2x5.
    U_245, V_245, W_245 = _combine_m(U_225, V_225, W_225, U_225, V_225, W_225, n=2, m1=2, m2=2, p=5)
    
    # 7. 2x4x8 (Combine p=3 and p=5)
    # C = A*B3 + A*B5.
    U_248, V_248, W_248 = _combine_p(U_243, V_243, W_243, U_245, V_245, W_245, n=2, m=4, p1=3, p2=5)
    
    # Convert to lists of ints
    U = [[int(x) for x in row] for row in U_248]
    V = [[int(x) for x in row] for row in V_248]
    W = [[int(x) for x in row] for row in W_248]
    
    return {"U": U, "V": V, "W": W}

def _construct_rank56():
    """
    Constructs rank 56 decomposition for 2x4x8.
    2x4x8 = 2x4x4 + 2x4x4.
    2x4x4 = 2x2x4 + 2x2x4.
    2x2x4 = 2x2x2 + 2x2x2.
    """
    n = 2
    U_222, V_222, W_222 = _strassen_222()
    
    # 2x2x4
    U_224, V_224, W_224 = _combine_p(U_222, V_222, W_222, U_222, V_222, W_222, n=2, m=2, p1=2, p2=2)
    
    # 2x4x4
    U_244, V_244, W_244 = _combine_m(U_224, V_224, W_224, U_224, V_224, W_224, n=2, m1=2, m2=2, p=4)
    
    # 2x4x8
    U_248, V_248, W_248 = _combine_p(U_244, V_244, W_244, U_244, V_244, W_244, n=2, m=4, p1=4, p2=4)
    
    U = [[int(x) for x in row] for row in U_248]
    V = [[int(x) for x in row] for row in V_248]
    W = [[int(x) for x in row] for row in W_248]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try rank 56
    if time.time() < deadline:
        try:
            res56 = _construct_rank56()
            if len(res56["U"][0]) < best_rank:
                if _verify(res56["U"], res56["V"], res56["W"], n, m, p):
                    best = res56
                    best_rank = len(res56["U"][0])
        except Exception:
            pass
            
    # Try rank 50
    if time.time() < deadline:
        try:
            res50 = _construct_rank50()
            if len(res50["U"][0]) < best_rank:
                if _verify(res50["U"], res50["V"], res50["W"], n, m, p):
                    best = res50
                    best_rank = len(res50["U"][0])
        except Exception:
            pass
            
    return best

# EVOLVE-BLOCK-END
