"""Matrix multiplication 4x4x8 tensor decomposition.
Constructs a rank-98 exact decomposition using the Kronecker product of
two Strassen 2x2x2 decompositions. Falls back to schoolbook rank-128
if verification fails.
"""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin


def _time_left():
    return max(0.0, _DEADLINE - time.time())


def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    
    Conventions:
      U[i, r] : A factor, A indexed by (i_row, i_col) -> U[i_row*2 + i_col, r]
      V[j, r] : B factor, B indexed by (j_row, j_col) -> V[j_row*2 + j_col, r]
      W[k, r] : C factor, C indexed by (k_row, k_col) -> W[k_row*2 + k_col, r]
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W


def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen.
    
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    U indexed by (i, j) -> i*4 + j
    V indexed by (j, k) -> j*4 + k
    W indexed by (k, i) -> k*4 + i
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    # U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
                    U4[i * 4 + j, c] = U2[(i // 2) * 2 + (j // 2), c0] * U2[(i % 2) * 2 + (j % 2), c1]
            for j in range(4):
                for k in range(4):
                    # V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
                    V4[j * 4 + k, c] = V2[(j // 2) * 2 + (k // 2), c0] * V2[(j % 2) * 2 + (k % 2), c1]
            for k in range(4):
                for i in range(4):
                    # W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
                    W4[k * 4 + i, c] = W2[(k // 2) * 2 + (i // 2), c0] * W2[(k % 2) * 2 + (i % 2), c1]
    return U4, V4, W4


def _build_4x4x8_strassen():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98.
    
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8):
    """Verify the tensor decomposition exactly.
    
    Checks if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    rank = U.shape[1]
    # Flatten U, V, W for faster access? No, direct indexing is fine for small sizes.
    # U: (16, R), V: (32, R), W: (32, R)
    
    # Pre-flatten for speed? 
    # U_flat[i, j, r] = U[i*m+j, r]
    # V_flat[j, k, r] = V[j*p+k, r]
    # W_flat[k, i, r] = W[k*n+i, r]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Vectorized: np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                prod = U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :]
                total = np.sum(prod)
                if total != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int16)
    V = np.zeros((m * p, rank), dtype=np.int16)
    W = np.zeros((p * n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    # Try the rank-98 Strassen-Kronecker decomposition
    try:
        if _time_left() > 10:
            U, V, W = _build_4x4x8_strassen()
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to schoolbook if Strassen fails or times out
    U, V, W = _schoolbook(4, 4, 8)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
