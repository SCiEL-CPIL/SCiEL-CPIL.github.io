"""Optimized matrix multiplication 3x4x7 decomposition using Strassen for 3x3x3 core."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    t0 = time.time()

    # Block decomposition strategy
    # n=3 -> 2+1
    # m=4 -> 2+2
    # p=7 -> 2+2+3
    
    # Total rank calculation:
    # 4 blocks of 2x2x2 (Strassen, rank 7): 4 * 7 = 28
    # 2 blocks of 2x2x3 (Schoolbook, rank 12): 2 * 12 = 24
    # 4 blocks of 1x2x2 (Schoolbook, rank 4): 4 * 4 = 16
    # 2 blocks of 1x2x3 (Schoolbook, rank 6): 2 * 6 = 12
    # Total: 28 + 24 + 16 + 12 = 80
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1..M7
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 1, 0, 1, 0],   # (1,0)
        [1, 1, 0, 1, 0, 0, -1]   # (1,1)
    ]
    # Wait, let's use the standard Strassen mapping derived earlier.
    # U (A matrix, 2x2):
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, A00(-1)
    # M7: A01, A11(-1)
    
    sU = [
        [1, 0, 1, 0, 1, -1, 0],  # (0,0)
        [0, 0, 0, 0, 1, 0, 1],   # (0,1)
        [0, 1, 0, 1, 0, 1, 0],   # (1,0)
        [1, 1, 0, 1, 0, 0, -1]   # (1,1)
    ]
    
    # V (B matrix, 2x2):
    # M1: B00, B11
    # M2: B00
    # M3: B10, B11(-1)
    # M4: B01, B00(-1)
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    
    sV = [
        [1, 1, 0, -1, 0, 1, 0],  # (0,0)
        [0, 0, 1, 1, 0, 1, 1],   # (0,1)
        [0, 0, -1, 0, 1, 0, 0],  # (1,0)
        [1, 0, 0, 0, 0, 0, 1]    # (1,1)
    ]
    
    # W (C matrix, 2x2, index k*2+i):
    # C00 (k=0, i=0): M1+M4-M5+M7
    # C10 (k=0, i=1): M2+M4
    # C01 (k=1, i=0): M3+M5
    # C11 (k=1, i=1): M1-M2+M3+M6
    
    sW = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    def add_strassen(n0, m0, p0):
        nonlocal col
        # U: rows n0..n0+1, cols m0..m0+1
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[(n0 + i) * m + (m0 + j)][col + c] += sU[i * 2 + j][c]
        
        # V: rows m0..m0+1, cols p0..p0+1
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[(m0 + j) * p + (p0 + k)][col + c] += sV[j * 2 + k][c]
        
        # W: rows p0..p0+1, cols n0..n0+1
        # W index: k*n + i
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[(p0 + k) * n + (n0 + i)][col + c] += sW[k * 2 + i][c]
        
        col += 7

    def add_schoolbook(n0, m0, p0, hn, hm, hp):
        nonlocal col
        for i in range(hn):
            for j in range(hm):
                for k in range(hp):
                    U[(n0 + i) * m + (m0 + j)][col] = 1
                    V[(m0 + j) * p + (p0 + k)][col] = 1
                    W[(p0 + k) * n + (n0 + i)][col] = 1
                    col += 1

    # Define blocks
    # n splits: [0, 2), [2, 3)
    # m splits: [0, 2), [2, 4)
    # p splits: [0, 2), [2, 4), [4, 7)
    
    n_splits = [(0, 2), (2, 1)]
    m_splits = [(0, 2), (2, 2)]
    p_splits = [(0, 2), (2, 2), (4, 3)]
    
    for n0, hn in n_splits:
        for m0, hm in m_splits:
            for p0, hp in p_splits:
                if hn == 2 and hm == 2 and hp == 2:
                    add_strassen(n0, m0, p0)
                else:
                    add_schoolbook(n0, m0, p0, hn, hm, hp)

    # Verify
    if time.time() - t0 < 110:
        ok = True
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[ii * m + jj][c] * V[jj * p + kk][c] * W[kk * n + ii][c]
                    if val != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    c = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][c] = 1
                V_fb[jj * p + kk][c] = 1
                W_fb[kk * n + ii][c] = 1
                c += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
