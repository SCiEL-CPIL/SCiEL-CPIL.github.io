import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if n * m > 0 else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Indexing: U[i*2+j], V[i*2+j], W[i*2+j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _block_decompose():
    """
    Decompose 3x3x8 matrix multiplication using block structure.
    A is 3x3, B is 3x8, C = A*B is 3x8.
    
    Split A: A00 (2x2), A01 (2x1), A10 (1x2), A11 (1x1)
    Split B: B_top (2x8) split into 4 blocks Bk (2x2), B_bot (1x8) split into 4 blocks Dk (1x2)
    
    For each k in 0..3:
      C[0:2, 2k:2k+2] = A00*Bk + A01*Dk
      C[2, 2k:2k+2] = A10*Bk + A11*Dk
    
    Ranks per block:
      A00*Bk: Strassen rank 7
      A01*Dk: outer product, rank 1
      A10*Bk: matrix-vector, rank 2
      A11*Dk: scalar times vector, rank 1
    Total per block: 11, total: 44
    """
    n, m, p = 3, 3, 8
    total_rank = 44
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for k in range(4):
        col_start = k * 2
        
        # 1) Strassen for A00*Bk: rank 7
        sU, sV, sW = _strassen_222()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * 3 + j          # A00[i,j]
                    v_idx = i * 8 + (col_start + j)  # Bk[i,j]
                    w_idx = (col_start + j) * 3 + i  # C[i, col_start+j]
                    U[u_idx, col + r] += sU[i*2+j, r]
                    V[v_idx, col + r] += sV[i*2+j, r]
                    W[w_idx, col + r] += sW[i*2+j, r]
        col += 7
        
        # 2) A01*Dk: rank 1 (outer product)
        # T[i,j] = A01[i] * Dk[j] for i in 0..1, j in 0..1
        # U[i*3+2, col] = 1, V[2*8+col_start+j, col] = 1, W[(col_start+j)*3+i, col] = 1
        for i in range(2):
            for j in range(2):
                u_idx = i * 3 + 2
                v_idx = 2 * 8 + (col_start + j)
                w_idx = (col_start + j) * 3 + i
                U[u_idx, col] += 1
                V[v_idx, col] += 1
                W[w_idx, col] += 1
        col += 1
        
        # 3) A10*Bk: rank 2
        # T[j] = A10[0]*Bk[0,j] + A10[1]*Bk[1,j] for j in 0..1
        # Term 1: U[2*3+0, col] = 1, V[0*8+col_start+j, col] = 1, W[(col_start+j)*3+2, col] = 1
        # Term 2: U[2*3+1, col+1] = 1, V[1*8+col_start+j, col+1] = 1, W[(col_start+j)*3+2, col+1] = 1
        for j in range(2):
            # Term 1: A10[0] * Bk[0,j]
            u_idx1 = 2 * 3 + 0
            v_idx1 = 0 * 8 + (col_start + j)
            w_idx1 = (col_start + j) * 3 + 2
            U[u_idx1, col] += 1
            V[v_idx1, col] += 1
            W[w_idx1, col] += 1
            
            # Term 2: A10[1] * Bk[1,j]
            u_idx2 = 2 * 3 + 1
            v_idx2 = 1 * 8 + (col_start + j)
            w_idx2 = (col_start + j) * 3 + 2
            U[u_idx2, col + 1] += 1
            V[v_idx2, col + 1] += 1
            W[w_idx2, col + 1] += 1
        col += 2
        
        # 4) A11*Dk: rank 1
        # T[j] = A11 * Dk[j] for j in 0..1
        # U[2*3+2, col] = 1, V[2*8+col_start+j, col] = 1, W[(col_start+j)*3+2, col] = 1
        for j in range(2):
            u_idx = 2 * 3 + 2
            v_idx = 2 * 8 + (col_start + j)
            w_idx = (col_start + j) * 3 + 2
            U[u_idx, col] += 1
            V[v_idx, col] += 1
            W[w_idx, col] += 1
        col += 1
    
    # Convert to lists
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _schoolbook():
    n, m, p = 3, 3, 8
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

def construct():
    n, m, p = DIMENSIONS
    # Try the block decomposition first (rank 44)
    result = _block_decompose()
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    # Fall back to schoolbook (rank 72)
    result = _schoolbook()
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    return result
