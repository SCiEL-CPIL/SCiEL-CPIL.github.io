import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def _build_target():
    """Build the target multiplication tensor once."""
    n, m, p = DIMENSIONS
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

_TARGET = None
def _get_target():
    global _TARGET
    if _TARGET is None:
        _TARGET = _build_target()
    return _TARGET

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = _get_target()
    # Use einsum for efficient computation
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    err = np.max(np.abs(T - target))
    return err <= tol

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Split into 4 blocks of 2x3 * 3x2.
    Each block uses Strassen-7 for 2x2*2x2 part + 4 rank-1 terms for 2x1*1x2 part.
    Total rank per block = 11. Total rank = 44.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Indices for A (2x3):
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        
        # Indices for B (3x2 block):
        # B[0,k0], B[1,k0], B[2,k0]
        # B[0,k1], B[1,k1], B[2,k1]
        # B index = j*p + k
        # b00 = 0*p + k0 = k0
        # b10 = 1*p + k0 = 8 + k0
        # b20 = 2*p + k0 = 16 + k0
        # b01 = 0*p + k1 = k1
        # b11 = 1*p + k1 = 8 + k1
        # b21 = 2*p + k1 = 16 + k1
        
        # Indices for C (2x2 block):
        # C[i,k]
        # C index = k*n + i
        # c00 = k0*2 + 0 = 2*k0
        # c01 = k1*2 + 0 = 2*k1
        # c10 = k0*2 + 1 = 2*k0 + 1
        # c11 = k1*2 + 1 = 2*k1 + 1

        # --- Strassen Part (2x2 * 2x2) ---
        # A part: A[:, 0:2] (cols 0,1) -> indices 0,1,3,4
        # B part: B[0:2, :] (rows 0,1) -> indices k0, 8+k0, k1, 8+k1
        # C part: C[:, k0:k1+1] -> indices 2*k0, 2*k0+1, 2*k1, 2*k1+1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1

        # --- Outer Product Part (2x1 * 1x2) ---
        # A part: A[:, 2] (col 2) -> indices 2, 5
        # B part: B[2, :] (row 2) -> indices 16+k0, 16+k1
        # C part: C[:, k0:k1+1] -> same C indices as above
        
        # term 1: a02 * b20 -> C[0, k0]
        t = col_offset
        U[2, t] = 1
        V[16+k0, t] = 1
        W[2*k0, t] = 1
        col_offset += 1
        
        # term 2: a02 * b21 -> C[0, k1]
        t = col_offset
        U[2, t] = 1
        V[16+k1, t] = 1
        W[2*k1, t] = 1
        col_offset += 1
        
        # term 3: a12 * b20 -> C[1, k0]
        t = col_offset
        U[5, t] = 1
        V[16+k0, t] = 1
        W[2*k0+1, t] = 1
        col_offset += 1
        
        # term 4: a12 * b21 -> C[1, k1]
        t = col_offset
        U[5, t] = 1
        V[16+k1, t] = 1
        W[2*k1+1, t] = 1
        col_offset += 1

    assert col_offset == 44
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Fallback: schoolbook
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            best_U, best_V, best_W = U44, V44, W44
            best_rank = 44
    except Exception:
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
