"""Matrix multiplication 2,4,8 rank-minimal exact tensor decomposition."""
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

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, shapes (4,7), (4,7), (4,7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]"""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: m1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # c=1: m2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # c=2: m3 = a00(b10-b00)
    U[0,2]=1
    V[2,2]=1; V[0,2]=-1
    W[1,2]=1; W[3,2]=-1
    
    # c=3: m4 = a11(b01-b11)
    U[3,3]=1
    V[1,3]=1; V[3,3]=-1
    W[0,3]=-1; W[2,3]=1
    
    # c=4: m5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # c=5: m6 = (a10-a00)b01
    U[2,5]=1; U[0,5]=-1
    V[1,5]=1
    W[3,5]=1
    
    # c=6: m7 = (a01-a11)b10
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_2x4x8_hierarchical():
    """
    Build 2x4x8 using hierarchical Strassen:
    2x4x8 = two 2x4x4 (concatenated in output k)
    2x4x4 = sum of two 2x2x4 (split input j)
    2x2x4 = sum of two 2x2x2 (split output k)
    2x2x2 = Strassen rank 7
    
    Total rank: 2 * 2 * 2 * 7 = 56
    """
    n, m, p = 2, 4, 8
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Build 2x2x4: sum of two 2x2x2 (k split into {0,1} and {2,3})
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # Part 1: k=0,1
    U_224[:, 0:7] = Us
    for j in range(2):
        for k in range(2):
            V_224[j*4+k, 0:7] = Vs[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[k*2+i, 0:7] = Ws[k*2+i, 0:7]
    
    # Part 2: k=2,3
    U_224[:, 7:14] = Us
    for j in range(2):
        for k in range(2):
            V_224[j*4+(k+2), 7:14] = Vs[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[(k+2)*2+i, 7:14] = Ws[k*2+i, 0:7]
    
    # Build 2x4x4: sum of two 2x2x4 (j split into {0,1} and {2,3})
    U_244 = np.zeros((8, 28))
    V_244 = np.zeros((16, 28))
    W_244 = np.zeros((8, 28))
    
    # Part 1: j=0,1
    U_244[0:4, 0:14] = U_224
    for j in range(2):
        for k in range(4):
            V_244[j*4+k, 0:14] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 0:14] = W_224[k*2+i, 0:14]
    
    # Part 2: j=2,3
    U_244[4:8, 14:28] = U_224
    for j in range(2):
        for k in range(4):
            V_244[(j+2)*4+k, 14:28] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 14:28] = W_224[k*2+i, 0:14]
    
    # Build 2x4x8: concatenation of two 2x4x4 (k split into {0..3} and {4..7})
    U_248 = np.zeros((8, 56))
    V_248 = np.zeros((32, 56))
    W_248 = np.zeros((16, 56))
    
    # Part 1: k=0..3
    U_248[:, 0:28] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+k, 0:28] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[k*2+i, 0:28] = W_244[k*2+i, 0:28]
    
    # Part 2: k=4..7
    U_248[:, 28:56] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+(k+4), 28:56] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[(k+4)*2+i, 28:56] = W_244[k*2+i, 0:28]
    
    return U_248, V_248, W_248

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p  # 64
    
    # Try hierarchical Strassen (rank 56)
    try:
        U, V, W = _build_2x4x8_hierarchical()
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 56
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
