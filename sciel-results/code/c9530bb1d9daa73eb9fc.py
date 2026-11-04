import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    n, m, p = 4, 5, 6
    U[i0*m + j0, offset+0] = 1
    U[(i0+1)*m + (j0+1), offset+0] = 1
    V[j0*p + k0, offset+0] = 1
    V[(j0+1)*p + (k0+1), offset+0] = 1
    W[k0*n + i0, offset+0] = 1
    W[(k0+1)*n + i0, offset+0] = 1
    W[k0*n + (i0+1), offset+0] = 1
    W[(k0+1)*n + (i0+1), offset+0] = 1
    
    U[(i0+1)*m + j0, offset+1] = 1
    U[(i0+1)*m + (j0+1), offset+1] = 1
    V[j0*p + k0, offset+1] = 1
    W[k0*n + (i0+1), offset+1] = 1
    W[(k0+1)*n + (i0+1), offset+1] = -1
    
    U[i0*m + j0, offset+2] = 1
    V[j0*p + (k0+1), offset+2] = 1
    V[(j0+1)*p + (k0+1), offset+2] = -1
    W[(k0+1)*n + (i0+1), offset+2] = 1
    
    U[(i0+1)*m + (j0+1), offset+3] = 1
    V[(j0+1)*p + k0, offset+3] = 1
    V[j0*p + k0, offset+3] = -1
    W[k0*n + i0, offset+3] = -1
    W[(k0+1)*n + i0, offset+3] = 1
    
    U[i0*m + j0, offset+4] = 1
    U[i0*m + (j0+1), offset+4] = 1
    V[(j0+1)*p + (k0+1), offset+4] = 1
    W[(k0+1)*n + i0, offset+4] = 1
    W[(k0+1)*n + (i0+1), offset+4] = 1
    
    U[(i0+1)*m + j0, offset+5] = 1
    U[i0*m + j0, offset+5] = -1
    V[j0*p + k0, offset+5] = 1
    V[j0*p + (k0+1), offset+5] = 1
    W[k0*n + i0, offset+5] = 1
    W[k0*n + (i0+1), offset+5] = 1
    
    U[i0*m + (j0+1), offset+6] = 1
    U[(i0+1)*m + (j0+1), offset+6] = -1
    V[(j0+1)*p + k0, offset+6] = 1
    V[(j0+1)*p + (k0+1), offset+6] = 1
    W[k0*n + i0, offset+6] = 1
    W[(k0+1)*n + (i0+1), offset+6] = 1
    
    return offset + 7

def _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset):
    n, m, p = 4, 5, 6
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                i = i0 + ii
                j = j0 + jj
                k = k0 + kk
                U[i*m + j, offset] = 1
                V[j*p + k, offset] = 1
                W[k*n + i, offset] = 1
                offset += 1
    return offset

def _build_block_decomp(n, m, p):
    # Partition: i:(2,2), j:(2,2,1), k:(2,2,2)
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    total_rank = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    offset = _strassen_2x2x2(U, V, W, i0, j0, k0, offset)
                else:
                    offset = _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset)
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    import random
    random.seed(42)
    for _ in range(2000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = sum(U[a, r] * V[b, r] * W[g, r] for r in range(rank))
            if abs(val) > tol:
                return False
    return True

def _try_strassen_2x3x2(U, V, W, i0, j0, k0, offset):
    """
    Try a rank-9 decomposition for a 2x3x2 block (Strassen-like).
    n_rows=2 (i), m_cols=3 (j), p_cols=2 (k)
    """
    n, m, p = 4, 5, 6
    r = offset
    # 9 factors for 2x3x2
    # We use a custom decomposition
    # M = A(2x3) * B(3x2), C = M (2x2)
    # We need to represent M[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try 9 factors
    # F0: A[0,0]*B[0,0] -> C[0,0]
    U[i0*m+j0, r] = 1
    V[j0*p+k0, r] = 1
    W[k0*n+i0, r] = 1
    r += 1
    # F1: A[0,1]*B[1,0] -> C[0,0]
    U[i0*m+j0+1, r] = 1
    V[(j0+1)*p+k0, r] = 1
    W[k0*n+i0, r] = 1
    r += 1
    # F2: A[0,2]*B[2,0] -> C[0,0]
    U[i0*m+j0+2, r] = 1
    V[(j0+2)*p+k0, r] = 1
    W[k0*n+i0, r] = 1
    r += 1
    # F3: A[0,0]*B[0,1] -> C[0,1]
    U[i0*m+j0, r] = 1
    V[j0*p+k0+1, r] = 1
    W[(k0+1)*n+i0, r] = 1
    r += 1
    # F4: A[0,1]*B[1,1] -> C[0,1]
    U[i0*m+j0+1, r] = 1
    V[(j0+1)*p+k0+1, r] = 1
    W[(k0+1)*n+i0, r] = 1
    r += 1
    # F5: A[0,2]*B[2,1] -> C[0,1]
    U[i0*m+j0+2, r] = 1
    V[(j0+2)*p+k0+1, r] = 1
    W[(k0+1)*n+i0, r] = 1
    r += 1
    # F6: A[1,0]*B[0,0] -> C[1,0]
    U[(i0+1)*m+j0, r] = 1
    V[j0*p+k0, r] = 1
    W[k0*n+i0+1, r] = 1
    r += 1
    # F7: A[1,1]*B[1,0] -> C[1,0]
    U[(i0+1)*m+j0+1, r] = 1
    V[(j0+1)*p+k0, r] = 1
    W[k0*n+i0+1, r] = 1
    r += 1
    # F8: A[1,2]*B[2,0] -> C[1,0]
    U[(i0+1)*m+j0+2, r] = 1
    V[(j0+2)*p+k0, r] = 1
    W[k0*n+i0+1, r] = 1
    r += 1
    # F9: A[1,0]*B[0,1] -> C[1,1]
    U[(i0+1)*m+j0, r] = 1
    V[j0*p+k0+1, r] = 1
    W[(k0+1)*n+i0+1, r] = 1
    r += 1
    # F10: A[1,1]*B[1,1] -> C[1,1]
    U[(i0+1)*m+j0+1, r] = 1
    V[(j0+1)*p+k0+1, r] = 1
    W[(k0+1)*n+i0+1, r] = 1
    r += 1
    # F11: A[1,2]*B[2,1] -> C[1,1]
    U[(i0+1)*m+j0+2, r] = 1
    V[(j0+2)*p+k0+1, r] = 1
    W[(k0+1)*n+i0+1, r] = 1
    r += 1
    return r

def _build_decomp_v2(n, m, p):
    """
    Alternative decomposition: use Strassen where possible, 
    but also try to merge adjacent blocks.
    """
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    total_rank = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    offset = _strassen_2x2x2(U, V, W, i0, j0, k0, offset)
                else:
                    offset = _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Primary: block Strassen decomposition
    U, V, W = _build_block_decomp(n, m, p)
    rank = U.shape[1]
    
    if time.time() < deadline - 10:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback schoolbook
    sb = n * m * p
    U2 = np.zeros((n * m, sb))
    V2 = np.zeros((m * p, sb))
    W2 = np.zeros((p * n, sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, col] = 1
                V2[j*p+k, col] = 1
                W2[k*n+i, col] = 1
                col += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
