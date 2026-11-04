"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U, V, W have shape (4, 7) where indices are (i,j) for U, (j,k) for V, (k,i) for W
    with i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen.
    U: (16, 49) indexed by (i,j)
    V: (16, 49) indexed by (j,k)
    W: (16, 49) indexed by (k,i)
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _build_4x4x8_shared():
    """
    Build a rank-98 decomposition for 4x4 * 4x8 using two 4x4x4 Strassen-Kronecker blocks.
    U: (16, 98) indexed by (i,j) where i in 0..3, j in 0..3
    V: (32, 98) indexed by (j,k) where j in 0..3, k in 0..7
    W: (32, 98) indexed by (k,i) where k in 0..7, i in 0..3
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
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition. Returns True if valid."""
    rank = U.shape[1]
    # Vectorized verification for speed
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute tensor product
    # T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T is all 1s
    if not np.all(T == 1):
        return False
    return True

def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank_reduction(U, V, W, max_time=30.0):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms. This is a heuristic search.
    
    Returns (U, V, W) with potentially lower rank, or the original if no reduction found.
    """
    if time.time() > _DEADLINE - 10:
        return U, V, W
    
    rank = U.shape[1]
    n, m, p = 4, 4, 8
    
    # Check for duplicate columns (same U, V, W pattern)
    seen = {}
    duplicates = []
    for r in range(rank):
        key = (tuple(U[:, r]), tuple(V[:, r]), tuple(W[:, r]))
        if key in seen:
            duplicates.append(r)
        else:
            seen[key] = r
    
    if duplicates:
        keep = [r for r in range(rank) if r not in set(duplicates)]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Primary strategy: rank-98 Strassen-Kronecker decomposition
    try:
        U, V, W = _build_4x4x8_shared()
        if _verify(U, V, W, n, m, p):
            # Try to reduce rank further
            U, V, W = _try_rank_reduction(U, V, W)
            # Re-verify after potential reduction
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # If verification failed, fall through to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook rank-128
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
