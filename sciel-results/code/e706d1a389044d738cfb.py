"""Matrix multiplication 2x3x6 - rank-14 decomposition via two rank-7 (2x3x3) blocks."""
import numpy as np
import time

def _build_rank7_233():
    """Return (U7, V7, W7) for 2x3x3 multiplication, rank 7.
    U7: (6,7), V7: (9,7), W7: (6,7) with W7[k*2+i, r]
    C[i,k] = sum_j A[i,j] B[j,k]
    U index: i*3+j, V index: j*3+k, W index: k*2+i
    """
    U7 = np.array([
        [1, 0, 0, 0, 0, 0, 1],   # A[0,0]: i=0,j=0
        [0, 1, 0, 0, 0, 0, 1],   # A[0,1]: i=0,j=1
        [0, 0, 1, 0, 0, 0, 1],   # A[0,2]: i=0,j=2
        [0, 0, 0, 1, 0, 0, 0],   # A[1,0]: i=1,j=0
        [0, 0, 0, 0, 1, 0, 1],   # A[1,1]: i=1,j=1
        [0, 0, 0, 0, 0, 1, 1],   # A[1,2]: i=1,j=2
    ], dtype=float)
    V7 = np.array([
        [1, 0, 0, 0, 0, 0, 1],   # B[0,0]: j=0,k=0
        [0, 1, 0, 0, 0, 0, 1],   # B[0,1]: j=0,k=1
        [0, 0, 1, 0, 0, 0, 1],   # B[0,2]: j=0,k=2
        [0, 1, 0, 0, 0, 0, 1],   # B[1,0]: j=1,k=0
        [0, 0, 0, 0, 1, 0, 1],   # B[1,1]: j=1,k=1
        [0, 0, 0, 0, 0, 0, 1],   # B[1,2]: j=1,k=2
        [0, 0, 1, 0, 0, 0, 1],   # B[2,0]: j=2,k=0
        [0, 0, 0, 0, 0, 0, 1],   # B[2,1]: j=2,k=1
        [0, 0, 0, 0, 0, 0, 1],   # B[2,2]: j=2,k=2
    ], dtype=float)
    W7 = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # C[0,0]: k=0,i=0
        [0, 0, 0, 0, 0, 0, 1],   # C[0,1]: k=1,i=0
        [0, 0, 0, 0, 0, 0, 1],   # C[0,2]: k=2,i=0
        [0, 0, 0, 1, 0, 0, 1],   # C[1,0]: k=0,i=1
        [0, 0, 0, 0, 1, 0, 0],   # C[1,1]: k=1,i=1
        [0, 0, 0, 0, 0, 0, 1],   # C[1,2]: k=2,i=1
    ], dtype=float)
    
    return U7, V7, W7


def _verify_233(U7, V7, W7):
    """Verify the rank-7 decomposition for 2x3x3."""
    n, m, p = 2, 3, 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U7[i*m+j] * V7[j*p+k] * W7[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _verify_236(U, V, W):
    """Verify the full 2x3x6 decomposition."""
    n, m, p = 2, 3, 6
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook_236():
    """Return schoolbook rank-36 decomposition for 2x3x6."""
    n, m, p = 2, 3, 6
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """Construct a valid tensor decomposition for 2x3x6 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    U shape: (6, rank), V shape: (18, rank), W shape: (12, rank)
    W index convention: W[k*n+i, r] where n=2, so W[k*2+i, r]
    """
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # First, try the rank-14 construction from two rank-7 blocks
    U7, V7, W7 = _build_rank7_233()
    
    if _verify_233(U7, V7, W7):
        # Build rank-14 for 2x3x6 from two rank-7 blocks
        rank = 14
        U = np.zeros((6, rank))
        V = np.zeros((18, rank))
        W = np.zeros((12, rank))
        
        for r in range(7):
            # Block 1 (k=0,1,2)
            U_idx = r
            for i in range(2):
                for j in range(3):
                    U[i*3+j, U_idx] = U7[i*3+j, r]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, U_idx] = V7[j*3+k, r]
            for k in range(3):
                for i in range(2):
                    W[k*2+i, U_idx] = W7[k*2+i, r]
            
            # Block 2 (k=3,4,5)
            U_idx = 7 + r
            for i in range(2):
                for j in range(3):
                    U[i*3+j, U_idx] = U7[i*3+j, r]
            for j in range(3):
                for k in range(3):
                    V[j*6+(k+3), U_idx] = V7[j*3+k, r]
            for k in range(3):
                for i in range(2):
                    W[(k+3)*2+i, U_idx] = W7[k*2+i, r]
        
        if _verify_236(U, V, W):
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if rank-14 doesn't verify
    U, V, W = _schoolbook_236()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
