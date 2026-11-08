import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    rank = 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2: loc = col*2 + row for 2x2 blocks
    # U: A block, V: B block, W: C block
    # M0=(A00+A11)(B00+B11): C00+=, C11+=
    # M1=(A10+A11)B00:        C10+=, C11-=
    # M2=A00(B10-B11):         C11+=
    # M3=A11(B01-B00):         C00+=, C11+=
    # M4=(A00+A01)B11:         C01+=
    # M5=(A10-A00)(B00+B01):   C00-=, C01+=
    # M6=(A01-A11)(B10+B11):   C10+=, C00-=

    S = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)],       [(1,1),(3,-1)]),
        ([(0,1)],       [(2,1),(3,-1)],[(3,1)]),
        ([(3,1)],       [(1,1),(0,-1)],[(0,1),(3,1)]),
        ([(0,1),(1,1)], [(3,1)],       [(2,1)]),
        ([(2,1),(0,-1)],[(0,1),(1,1)], [(0,-1),(2,1)]),
        ([(1,1),(3,-1)],[(2,1),(3,1)], [(1,1),(0,-1)]),
    ]

    def emit(base, i0, j0, k0):
        for t in range(7):
            r = base + t
            ut, vt, wt = S[t]
            for (loc, val) in ut:
                c, rr = divmod(loc, 2)
                U[(i0 + rr) * m + (j0 + c)][r] += val
            for (loc, val) in vt:
                c, kk = divmod(loc, 2)
                V[(j0 + c) * p + (k0 + kk)][r] += val
            for (loc, val) in wt:
                c, rr = divmod(loc, 2)
                W[(k0 + kk) * n + (i0 + rr)][r] += val

    for jc in (0, 2):
        for kc in (0, 2, 4, 6):
            base = (jc // 2) * 28 + (kc // 2) * 7
            emit(base, 0, jc, kc)

    return {"U": U, "V": V, "W": W}
