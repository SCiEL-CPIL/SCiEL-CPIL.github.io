import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 (32 terms)
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2, Strassen on 2x2x2 blocks (56 terms)
    # Split j into 2 blocks of 2, k into 4 blocks of 2
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                # U factor: maps local (i', j') -> global (i, j)
                for li in range(4):
                    if sU[t][li]:
                        li_p = li // 2
                        lj_p = li % 2
                        gi = 1 + li_p
                        gj = js + lj_p
                        U[gi * m + gj][col] += sU[t][li]
                # V factor: maps local (j', k') -> global (j, k)
                for lv in range(4):
                    if sV[t][lv]:
                        lj_p = lv // 2
                        lk_p = lv % 2
                        gj = js + lj_p
                        gk = ks + lk_p
                        V[gj * p + gk][col] += sV[t][lv]
                # W factor: maps local (i', k') -> global (k, i) per W[k*n+i]
                for lw in range(4):
                    if sW[t][lw]:
                        li_p = lw // 2
                        lk_p = lw % 2
                        gi = 1 + li_p
                        gk = ks + lk_p
                        W[gk * n + gi][col] += sW[t][lw]
                col += 1

    return {"U": U, "V": V, "W": W}
