import time
import numpy as np

DIMENSIONS = [5, 5, 5]

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

def construct():
    _start = time.time()
    _deadline = _start + 140
    
    n, m, p = DIMENSIONS
    
    # Define blocks: (I_start, J_start, K_start, ni, mj, pk)
    blocks = [
        (0, 0, 0, 2, 2, 2),  # (2,2,2) - use Strassen
        (0, 0, 2, 2, 2, 3),  # (2,2,3)
        (0, 2, 0, 2, 3, 2),  # (2,3,2)
        (0, 2, 2, 2, 3, 3),  # (2,3,3)
        (2, 0, 0, 3, 2, 2),  # (3,2,2)
        (2, 0, 2, 3, 2, 3),  # (3,2,3)
        (2, 2, 0, 3, 3, 2),  # (3,3,2)
        (2, 2, 2, 3, 3, 3),  # (3,3,3)
    ]
    
    # Get decompositions for each block
    decomps = []
    for idx, (i0, j0, k0, ni, mj, pk) in enumerate(blocks):
        if idx == 0:  # First block: 2x2x2, use Strassen
            decomps.append(_strassen_2x2x2())
        else:
            decomps.append(_schoolbook_block(ni, mj, pk))
    
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
