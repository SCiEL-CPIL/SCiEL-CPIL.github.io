import time

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    rank = 44

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen for 2x2x2: 7 multiplications
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B
    strassen = [
        # (A_entries, B_entries, C_entries)
        # m1 = (a00+a11)(b00+b11) -> c00+=m1, c11+=m1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # m2 = (a10+a11)b00 -> c10+=m2, c11+=m2
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),
        # m3 = a00(b10-b11) -> c00-=m3, c01+=m3
        ([(0,1)], [(2,1),(3,-1)], [(0,-1),(1,1)]),
        # m4 = a11(b01-b00) -> c10-=m4, c11+=m4
        ([(3,1)], [(1,1),(0,-1)], [(2,-1),(3,1)]),
        # m5 = (a00+a01)a10 -> c00+=m5, c10+=m5
        ([(0,1),(1,1)], [(2,1)], [(0,1),(2,1)]),
        # m6 = (a10-a00)(b00+b01) -> c01-=m6, c11-=m6
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,-1),(3,-1)]),
        # m7 = (a01-a11)(b10+b11) -> c00+=m7, c11-=m7
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),
    ]

    r = 0
    # Four 2x2x2 Strassen blocks for the 4x4x4 part
    for block in range(4):
        for sa, sb, sc in strassen:
            for (li, coeff) in sa:
                U[block * 2 + li][r] += coeff
            for (li, coeff) in sb:
                V[block * 2 + li][r] += coeff
            for (li, coeff) in sc:
                W[block * 2 + li][r] += coeff
            r += 1

    # 4x4x1 schoolbook for the 5th column of B (index 4)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][r] = 1.0
            V[j * 5 + 4][r] = 1.0
            W[4 * 4 + i][r] = 1.0
            r += 1

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(rank):
                    s += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}")

    return {"U": U, "V": V, "W": W}
