"""Optimized matrix multiplication 3x4x7 decomposition using improved block ranks."""
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
    
    # Ranks:
    # 2x2x2: 7 (Strassen)
    # 2x2x3: 11 (Strassen 2x2x2 + Schoolbook 2x2x1)
    # 1x2x2: 4 (Schoolbook)
    # 1x2x3: 6 (Schoolbook)
    
    # Total rank:
    # 4 * 7 (2x2x2) = 28
    # 2 * 11 (2x2x3) = 22
    # 4 * 4 (1x2x2) = 16
    # 2 * 6 (1x2x3) = 12
    # Total: 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    sU = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    sV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 1],
        [0, 0, -1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1]
    ]
    sW = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    def add_strassen(n0, m0, p0):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[(n0 + i) * m + (m0 + j)][col + c] += sU[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[(m0 + j) * p + (p0 + k)][col + c] += sV[j * 2 + k][c]
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

    def add_mixed_223(n0, m0, p0):
        nonlocal col
        # Part 1: Strassen for first 2 cols of p
        add_strassen(n0, m0, p0)
        # Part 2: Schoolbook for 3rd col of p (local index 2)
        # i in 0..1, j in 0..1, k=2
        for i in range(2):
            for j in range(2):
                U[(n0 + i) * m + (m0 + j)][col] = 1
                V[(m0 + j) * p + (p0 + 2)][col] = 1
                W[(p0 + 2) * n + (n0 + i)][col] = 1
                col += 1

    n_splits = [(0, 2), (2, 1)]
    m_splits = [(0, 2), (2, 2)]
    p_splits = [(0, 2), (2, 2), (4, 3)]
    
    for n0, hn in n_splits:
        for m0, hm in m_splits:
            for p0, hp in p_splits:
                if hn == 2 and hm == 2 and hp == 2:
                    add_strassen(n0, m0, p0)
                elif hn == 2 and hm == 2 and hp == 3:
                    add_mixed_223(n0, m0, p0)
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
