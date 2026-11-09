import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3 = A00(B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11(B01-B11)
    U[3][3] += 1
    V[1][3] += 1; V[3][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _schoolbook_block(ni, mj, pk):
    """Return (U, V, W) as lists for schoolbook decomposition of (ni,mj,pk) block."""
    rank = ni * mj * pk
    U = [[0] * rank for _ in range(ni * mj)]
    V = [[0] * rank for _ in range(mj * pk)]
    W = [[0] * rank for _ in range(pk * ni)]
    c = 0
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U[i * mj + j][c] = 1
                V[j * pk + k][c] = 1
                W[k * ni + i][c] = 1
                c += 1
    return U, V, W

def _smart_block(ni, mj, pk):
    """Recursively decompose a block using Strassen for 2x2x2 and splitting 3s."""
    if ni == 2 and mj == 2 and pk == 2:
        return _strassen_2x2x2()
    
    # If any dimension is 3, split it
    if ni == 3:
        U1, V1, W1 = _smart_block(2, mj, pk)
        U2, V2, W2 = _smart_block(1, mj, pk)
        # Concatenate along columns
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        U = [[0]*rank for _ in range(ni*mj)]
        V = [[0]*rank for _ in range(mj*pk)]
        W = [[0]*rank for _ in range(pk*ni)]
        
        # First part (rows 0-1 of ni)
        for r in range(rank1):
            for i in range(2):
                U[i*mj + j][r] = U1[i*mj + j][r]
            for j in range(mj):
                for k in range(pk):
                    V[j*pk + k][r] = V1[j*pk + k][r]
                for i in range(2):
                    W[k*ni + i][r] = W1[k*ni + i][r]
        
        # Second part (row 2 of ni)
        for r in range(rank2):
            for i in range(1):
                U[(2+i)*mj + j][rank1 + r] = U2[i*mj + j][r]
            for j in range(mj):
                for k in range(pk):
                    V[j*pk + k][rank1 + r] = V2[j*pk + k][r]
                for i in range(1):
                    W[k*ni + (2+i)][rank1 + r] = W2[k*ni + i][r]
        
        return U, V, W
    
    if mj == 3:
        U1, V1, W1 = _smart_block(ni, 2, pk)
        U2, V2, W2 = _smart_block(ni, 1, pk)
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        U = [[0]*rank for _ in range(ni*mj)]
        V = [[0]*rank for _ in range(mj*pk)]
        W = [[0]*rank for _ in range(pk*ni)]
        
        for r in range(rank1):
            for i in range(ni):
                for j in range(2):
                    U[i*mj + j][r] = U1[i*2 + j][r]
            for j in range(2):
                for k in range(pk):
                    V[j*pk + k][r] = V1[j*pk + k][r]
            for k in range(pk):
                for i in range(ni):
                    W[k*ni + i][r] = W1[k*ni + i][r]
        
        for r in range(rank2):
            for i in range(ni):
                for j in range(1):
                    U[i*mj + (2+j)][rank1 + r] = U2[i*1 + j][r]
            for j in range(1):
                for k in range(pk):
                    V[(2+j)*pk + k][rank1 + r] = V2[j*pk + k][r]
            for k in range(pk):
                for i in range(ni):
                    W[k*ni + i][rank1 + r] = W2[k*ni + i][r]
        
        return U, V, W
    
    if pk == 3:
        U1, V1, W1 = _smart_block(ni, mj, 2)
        U2, V2, W2 = _smart_block(ni, mj, 1)
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        U = [[0]*rank for _ in range(ni*mj)]
        V = [[0]*rank for _ in range(mj*pk)]
        W = [[0]*rank for _ in range(pk*ni)]
        
        for r in range(rank1):
            for i in range(ni):
                for j in range(mj):
                    U[i*mj + j][r] = U1[i*mj + j][r]
            for j in range(mj):
                for k in range(2):
                    V[j*pk + k][r] = V1[j*2 + k][r]
            for k in range(2):
                for i in range(ni):
                    W[k*ni + i][r] = W1[k*ni + i][r]
        
        for r in range(rank2):
            for i in range(ni):
                for j in range(mj):
                    U[i*mj + j][rank1 + r] = U2[i*mj + j][r]
            for j in range(mj):
                for k in range(1):
                    V[j*pk + (2+k)][rank1 + r] = V2[j*1 + k][r]
            for k in range(1):
                for i in range(ni):
                    W[(2+k)*ni + i][rank1 + r] = W2[k*ni + i][r]
        
        return U, V, W
    
    # Fallback: schoolbook for remaining cases (e.g., 1x1x1, 1x2x2, etc.)
    return _schoolbook_block(ni, mj, pk)

def _embed_block(U_block, V_block, W_block, i0, j0, k0, ni, mj, pk, U_full, V_full, W_full, col_offset):
    """Embed a block decomposition into the full 5x5x5 decomposition."""
    rank_block = len(U_block[0])
    n, m, p = 5, 5, 5
    
    for li in range(ni):
        for lj in range(mj):
            global_u_row = (i0 + li) * m + (j0 + lj)
            for r in range(rank_block):
                U_full[global_u_row][col_offset + r] += U_block[li * mj + lj][r]
    
    for lj in range(mj):
        for lk in range(pk):
            global_v_row = (j0 + lj) * p + (k0 + lk)
            for r in range(rank_block):
                V_full[global_v_row][col_offset + r] += V_block[lj * pk + lk][r]
    
    for lk in range(pk):
        for li in range(ni):
            global_w_row = (k0 + lk) * n + (i0 + li)
            for r in range(rank_block):
                W_full[global_w_row][col_offset + r] += W_block[lk * ni + li][r]
    
    return col_offset + rank_block

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is exact. Returns True if valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def construct():
    _start = time.time()
    _deadline = _start + 140
    
    n, m, p = DIMENSIONS
    
    # Define blocks: (I_start, J_start, K_start, ni, mj, pk)
    blocks = [
        (0, 0, 0, 2, 2, 2),  # (2,2,2)
        (0, 0, 2, 2, 2, 3),  # (2,2,3)
        (0, 2, 0, 2, 3, 2),  # (2,3,2)
        (0, 2, 2, 2, 3, 3),  # (2,3,3)
        (2, 0, 0, 3, 2, 2),  # (3,2,2)
        (2, 0, 2, 3, 2, 3),  # (3,2,3)
        (2, 2, 0, 3, 3, 2),  # (3,3,2)
        (2, 2, 2, 3, 3, 3),  # (3,3,3)
    ]
    
    # Get decompositions for each block using _smart_block
    decomps = []
    for idx, (i0, j0, k0, ni, mj, pk) in enumerate(blocks):
        decomps.append(_smart_block(ni, mj, pk))
    
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, (i0, j0, k0, ni, mj, pk) in enumerate(blocks):
        U_b, V_b, W_b = decomps[idx]
        col_offset = _embed_block(U_b, V_b, W_b, i0, j0, k0, ni, mj, pk, U_full, V_full, W_full, col_offset)
    
    # Verify
    if not _verify(U_full, V_full, W_full, n, m, p):
        # Fall back to schoolbook
        U_full, V_full, W_full = _schoolbook(n, m, p)
    
    return {"U": U_full, "V": V_full, "W": W_full}
