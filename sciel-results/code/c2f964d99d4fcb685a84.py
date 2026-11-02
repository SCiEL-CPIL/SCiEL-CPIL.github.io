"""Matrix multiplication 2,2,4 — rank-14 Strassen decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    R = 14

    # Pre-allocate as lists of lists of ints
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]

    # --- Left 2x2 block (cols 0,1 of B) ---
    # r=0: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # r=1: (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # r=2: a00*(b11-b01)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # r=3: a11*(b10-b00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # r=4: (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # r=5: (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # r=6: (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # --- Right 2x2 block (cols 2,3 of B) ---
    # r=7: (a00+a11)*(b02+b13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1

    # r=8: (a10+a11)*b02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1

    # r=9: a00*(b03-b13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1

    # r=10: a11*(b12-b02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1

    # r=11: (a00+a01)*b13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1

    # r=12: (a10-a00)*(b02+b03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1

    # r=13: (a01-a11)*(b12+b13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    # Quick verification over integer arithmetic
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for r in range(R):
                    acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if acc != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if ok:
        return {"U": U, "V": V, "W": W}

    # Fallback: naive rank-16
    rank = nm * mp  # not quite; proper naive is n*m*p = 16
    rank = n * m * p
    Uf = [[0] * rank for _ in range(nm)]
    Vf = [[0] * rank for _ in range(mp)]
    Wf = [[0] * rank for _ in range(pn)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i * m + j][idx] = 1
                Vf[j * p + k][idx] = 1
                Wf[k * n + i][idx] = 1
                idx += 1
    return {"U": Uf, "V": Vf, "W": Wf}
# EVOLVE-BLOCK-END
