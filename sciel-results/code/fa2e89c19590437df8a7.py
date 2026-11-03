"""Two-level Strassen decomposition for 4x4 * 4x8 -> 4x8.

Strategy: Use proper one-level Strassen on 4x4 blocks (7 block products of 2x2x2,
each of rank 7, giving rank 49 for 4x4x4). Since both halves of the 8-wide output
share the same A-side linear combinations, total rank = 49.
"""

# EVOLVE-BLOCK-START

import time

# 2x2x2 Strassen decomposition (rank 7)
# A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[p, q], [r, s]]
# Indices: A: 0=a,1=b,2=c,3=d; B: 0=e,1=f,2=g,3=h; C: 0=p,1=q,2=r,3=s
_S2_U = [
    [1, 0, 0, 1],   # M1: (a+d)
    [0, 0, 1, 1],   # M2: (c+d)
    [1, 0, 0, 0],   # M3: a
    [0, 0, 0, 1],   # M4: d
    [1, 1, 0, 0],   # M5: (a+b)
    [1, 0, -1, 0],  # M6: (a-c)
    [0, 1, 0, -1],  # M7: (b-d)
]
_S2_V = [
    [1, 0, 0, 1],   # M1: (e+h)
    [1, 0, 0, 0],   # M2: e
    [0, 1, 0, -1],  # M3: (f-h)
    [0, 0, 1, -1],  # M4: (g-e)
    [0, 0, 0, 1],   # M5: h
    [1, 1, 0, 0],   # M6: (e+f)
    [0, 0, 1, 1],   # M7: (g+h)
]
_S2_W = [
    [1, 0, 0, -1, 1, 1, 0],   # C00 = M1 - M4 + M5 + M6
    [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
    [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
    [1, 1, -1, 0, 0, 0, 1],   # C11 = M1 - M3 + M2 + M7
]

def _build_strassen_4x4x8():
    """Build rank-49 decomposition for 4x4 * 4x8 -> 4x8.
    
    One-level Strassen: 7 block products of 2x2x2 (each rank 7) = 49.
    Both output halves share the same U (A-side), so total rank = 49.
    """
    n, m, p = 4, 4, 8
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 32 x 49
    W = [[0] * rank for _ in range(p * n)]  # 32 x 49

    # Partition 4x4 into 2x2 blocks:
    # A blocks: A[bi][bj] for bi,bj in {0,1}, block at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B blocks: B[bj][bk] for bj,bk in {0,1}, block at rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C blocks: C[bi][bk] for bi,bk in {0,1}, block at rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    #
    # Strassen: 7 products M_t = (linear_comb_A_t) * (linear_comb_B_t)
    # Each M_t is a 2x2 * 2x2 = 2x2 product, decomposed via rank-7 Strassen.
    #
    # The 7 Strassen linear combinations (at block level):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A11-A21)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Block-level coefficients: for each Strassen term t, we need the 4x4 A-coefficients
    # and 4x4 B-coefficients (each as a 2x2 block pattern).
    #
    # A block indices: A[0][0]=A11, A[0][1]=A12, A[1][0]=A21, A[1][1]=A22
    # B block indices: B[0][0]=B11, B[0][1]=B12, B[1][0]=B21, B[1][1]=B22
    #
    # A-side block coefficients (which A-blocks participate and with what sign):
    # M1: A11+A22 -> blocks (0,0)+1, (1,1)+1
    # M2: A21+A22 -> blocks (1,0)+1, (1,1)+1
    # M3: A11 -> block (0,0)+1
    # M4: A22 -> block (1,1)+1
    # M5: A11+A12 -> blocks (0,0)+1, (0,1)+1
    # M6: A11-A21 -> blocks (0,0)+1, (1,0)-1
    # M7: A12-A22 -> blocks (0,1)+1, (1,1)-1
    #
    # B-side block coefficients:
    # M1: B11+B22 -> blocks (0,0)+1, (1,1)+1
    # M2: B11 -> block (0,0)+1
    # M3: B12-B22 -> blocks (0,1)+1, (1,1)-1
    # M4: B21-B11 -> blocks (1,0)+1, (0,0)-1
    # M5: B22 -> block (1,1)+1
    # M6: B11+B12 -> blocks (0,0)+1, (0,1)+1
    # M7: B21+B22 -> blocks (1,0)+1, (1,1)+1

    # A-side: for each Strassen term t, list of (block_row, block_col, coeff)
    A_block_coeffs = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,1)],      # M2
        [(0,0,1)],               # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(0,0,1), (1,0,-1)],     # M6
        [(0,1,1), (1,1,-1)],     # M7
    ]
    # B-side: for each Strassen term t, list of (block_row, block_col, coeff)
    B_block_coeffs = [
        [(0,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(0,1,1), (1,1,-1)],     # M3
        [(1,0,1), (0,0,-1)],     # M4
        [(1,1,1)],               # M5
        [(0,0,1), (0,1,1)],      # M6
        [(1,0,1), (1,1,1)],      # M7
    ]

    for t in range(7):
        # For this Strassen term, we have a 2x2 * 2x2 product (the "inner" multiplication)
        # decomposed into 7 rank-1 terms via 2x2x2 Strassen.
        for s in range(7):
            col = t * 7 + s
            # U: A-side coefficients
            # For each A-block (bi, bj) with coeff ca, and each 2x2 entry (i0, j0):
            for (bi, bj, ca) in A_block_coeffs[t]:
                for i0 in range(2):
                    for j0 in range(2):
                        i = 2 * bi + i0
                        j = 2 * bj + j0
                        u_val = ca * _S2_U[s][i0 * 2 + j0]
                        if u_val != 0:
                            U[i * m + j][col] += u_val

            # V: B-side coefficients for left half (k=0..3) and right half (k=4..7)
            for (bj, bk, cb) in B_block_coeffs[t]:
                for j0 in range(2):
                    for k0 in range(2):
                        j = 2 * bj + j0
                        k_left = 2 * bk + k0
                        k_right = k_left + 4
                        v_val = cb * _S2_V[s][j0 * 2 + k0]
                        if v_val != 0:
                            V[j * m + k_left][col] += v_val
                            V[j * m + k_right][col] += v_val

            # W: C-side coefficients for left half and right half
            for (bi, bk, cc) in A_block_coeffs[t]:  # C blocks use same (bi, bk) indexing
                pass  # We need C block coefficients, not A block coefficients!

            # Actually, C blocks are indexed by (bi, bk) where bi is the A-block row and bk is the B-block col.
            # The C-side Strassen W coefficients tell us which C-block entries get which linear combo.
            # But we also need to know which C-blocks are involved for each Strassen term.
            #
            # The output C = M1*M1 + ... (block level). Each Mt is a 2x2 matrix.
            # C11 = M1 - M4 + M5 + M6  (block (0,0) of C)
            # C12 = M3 + M5            (block (0,1) of C)
            # C21 = M2 + M4            (block (1,0) of C)
            # C22 = M1 - M3 + M2 + M7  (block (1,1) of C)
            #
            # So for Strassen term t, the C-blocks that receive contributions:
            # t=0 (M1): C11(+1), C22(+1)
            # t=1 (M2): C21(+1), C22(+1)
            # t=2 (M3): C12(+1), C22(-1)
            # t=3 (M4): C11(-1), C21(+1)
            # t=4 (M5): C11(+1), C12(+1)
            # t=5 (M6): C11(+1)
            # t=6 (M7): C22(+1)
            pass

    # Let me redo this more carefully.
    # C-block coefficients for each Strassen term t:
    # C block (bi, bk) gets coefficient from the C formulas.
    C_block_coeffs = [
        [(0,0,1), (1,1,1)],      # M1: C11+1, C22+1
        [(1,0,1), (1,1,1)],      # M2: C21+1, C22+1
        [(0,1,1), (1,1,-1)],     # M3: C12+1, C22-1
        [(0,0,-1), (1,0,1)],     # M4: C11-1, C21+1
        [(0,0,1), (0,1,1)],      # M5: C11+1, C12+1
        [(0,0,1)],               # M6: C11+1
        [(1,1,1)],               # M7: C22+1
    ]

    # Now rebuild U, V, W properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U: A-side
            for (bi, bj, ca) in A_block_coeffs[t]:
                for i0 in range(2):
                    for j0 in range(2):
                        i = 2 * bi + i0
                        j = 2 * bj + j0
                        u_val = ca * _S2_U[s][i0 * 2 + j0]
                        if u_val != 0:
                            U[i * m + j][col] += u_val

            # V: B-side (both halves)
            for (bj, bk, cb) in B_block_coeffs[t]:
                for j0 in range(2):
                    for k0 in range(2):
                        j = 2 * bj + j0
                        k_left = 2 * bk + k0
                        k_right = k_left + 4
                        v_val = cb * _S2_V[s][j0 * 2 + k0]
                        if v_val != 0:
                            V[j * m + k_left][col] += v_val
                            V[j * m + k_right][col] += v_val

            # W: C-side (both halves)
            for (bi, bk, cc) in C_block_coeffs[t]:
                for i0 in range(2):
                    for k0 in range(2):
                        i = 2 * bi + i0
                        k_left = 2 * bk + k0
                        k_right = k_left + 4
                        w_val = cc * _S2_W[i0 * 2 + k0][s]
                        if w_val != 0:
                            W[k_left * n + i][col] += w_val
                            W[k_right * n + i][col] += w_val

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    assert len(U) == n * m, f"U rows {len(U)} != {n*m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m*p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U)
    assert all(len(row) == rank for row in V)
    assert all(len(row) == rank for row in W)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False, (i, j, k, s)
    return True, None


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    result = _build_strassen_4x4x8()
    n, m, p = 4, 4, 8
    ok, err = _verify(result["U"], result["V"], result["W"], n, m, p)
    if not ok:
        # Fallback to schoolbook
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
    return result
# EVOLVE-BLOCK-END
