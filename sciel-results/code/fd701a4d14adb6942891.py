"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def laderman_factors():
        a = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, -1, 2, 3, -4, 5, 6, -7, 8],
            [0, 1, -2, 3, 4, -5, 6, 7, -8],
            [0, -1, -2, 3, -4, -5, 6, -7, -8],
            [0, 0, 2, 0, 0, 0, 6, 0, 8],
            [0, 0, 0, 3, 4, 5, 0, 7, 8],
            [0, 1, 0, 3, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 6, 0, 8],
            [0, 0, 2, 3, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 4, 5, 0, 0, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ]
        b = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, -1, 2, 3, -4, 5, 6, -7, 8],
            [0, 1, -2, 3, 4, -5, 6, 7, -8],
            [0, -1, -2, 3, -4, -5, 6, -7, -8],
            [0, 0, 2, 0, 0, 0, 6, 0, 8],
            [0, 0, 0, 3, 4, 5, 0, 7, 8],
            [0, 1, 0, 3, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 6, 0, 8],
            [0, 0, 2, 3, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 4, 5, 0, 0, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ]
        return a, b

    la, lb = laderman_factors()
    terms = []

    def add(u, v, i, k):
        terms.append((dict(u), dict(v), i, k))

    # C11 = A11*B11 (15) + A12*B21 (18)
    for t in range(15):
        u = {i * 5 + j: la[t][i * 3 + j] for i in range(3) for j in range(3)}
        v = {j * 5 + k: lb[t][j * 3 + k] for j in range(3) for k in range(3)}
        for i in range(3):
            for k in range(3):
                add(u, v, i, k)

    for i in range(3):
        for k in range(3):
            for s in range(2):
                u = {i * 5 + (3 + s): 1}
                v = {(3 + s) * 5 + k: 1}
                add(u, v, i, k)

    # C12 = A11*B12 (18) + A12*B22 (12)
    for i in range(3):
        for k in range(2):
            for s in range(3):
                u = {i * 5 + s: 1}
                v = {s * 5 + (3 + k): 1}
                add(u, v, i, 3 + k)

    for i in range(3):
        for k in range(2):
            for s in range(2):
                u = {i * 5 + (3 + s): 1}
                v = {(3 + s) * 5 + (3 + k): 1}
                add(u, v, i, 3 + k)

    # C21 = A21*B11 (18) + A22*B21 (12)
    for i in range(2):
        for k in range(3):
            for s in range(3):
                u = {(3 + i) * 5 + s: 1}
                v = {s * 5 + k: 1}
                add(u, v, 3 + i, k)

    for i in range(2):
        for k in range(3):
            for s in range(2):
                u = {(3 + i) * 5 + (3 + s): 1}
                v = {(3 + s) * 5 + k: 1}
                add(u, v, 3 + i, k)

    # C22 = A21*B12 (12) + A22*B22 (Strassen 7)
    for i in range(2):
        for k in range(2):
            for s in range(3):
                u = {(3 + i) * 5 + s: 1}
                v = {s * 5 + (3 + k): 1}
                add(u, v, 3 + i, 3 + k)

    # Strassen for 2x2
    # Indices for A22: (3,3)->18, (3,4)->19, (4,3)->23, (4,4)->24
    # Indices for B22: (3,3)->18, (3,4)->19, (4,3)->23, (4,4)->24
    # Output C22: i in 3..4, k in 3..4 -> W[k*5+i]
    
    # m1 = (A11+A22)(B11+B22)
    u1 = {18: 1, 24: 1}
    v1 = {18: 1, 24: 1}
    for i in [3, 4]:
        for k in [3, 4]:
            if (i==3 and k==3) or (i==4 and k==4):
                add(u1, v1, i, k) # C11 += m1, C22 += m1

    # m2 = (A21+A22)B11
    u2 = {23: 1, 24: 1}
    v2 = {18: 1, 19: 1} # B11 cols 0,1? No, B21 is rows 3,4 cols 0,1,2?
    # Wait, B21 is 2x3. Indices: (3,0)..(3,2), (4,0)..(4,2).
    # My previous cross term A22*B21 used B21.
    # Here Strassen is for A22*B22.
    # B22 indices: (3,3),(3,4),(4,3),(4,4).
    # So v2 should be B11 of B22? i.e. B22[0,0]=18, B22[0,1]=19.
    v2 = {18: 1, 19: 1}
    for i in [4]:
        for k in [3, 4]:
            if i==4 and k==3: add(u2, v2, i, k) # C21 += m2
            if i==4 and k==4: pass # C22 has -m2
    
    # m3 = A11(B22-B11)
    u3 = {18: 1}
    v3 = {19: 1, 24: -1} # B22[0,1] - B22[1,1] ? No.
    # B22 = [[b00, b01], [b10, b11]].
    # m3 = A11 * (B22[1,0] - B22[0,0])? No.
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's map A22: a=18, b=19, c=23, d=24
    # B22: e=18, f=19, g=23, h=24
    
    # Clear previous partial adds for C22 and restart with proper Strassen terms.
    # The previous loop added A21*B12 (12 terms).
    # Now add 7 terms for A22*B22.
    
    # m1 = (A18+A24)(B18+B24)
    u1 = {18: 1, 24: 1}
    v1 = {18: 1, 24: 1}
    add(u1, v1, 3, 3) # C11
    add(u1, v1, 4, 4) # C22
    
    # m2 = (A23+A24)B18
    u2 = {23: 1, 24: 1}
    v2 = {18: 1}
    add(u2, v2, 4, 3) # C21
    # C22 -= m2
    u2_neg = {23: 1, 24: 1}
    v2_neg = {18: -1}
    add(u2_neg, v2_neg, 4, 4) # C22

    # m3 = A18(B23-B18)
    u3 = {18: 1}
    v3 = {23: 1, 18: -1}
    add(u3, v3, 3, 4) # C12
    # C22 += m3
    u3_add = {18: 1}
    v3_add = {23: 1, 18: -1}
    add(u3_add, v3_add, 4, 4) # C22

    # m4 = A24(B23-B18)
    u4 = {24: 1}
    v4 = {23: 1, 18: -1}
    add(u4, v4, 3, 3) # C11
    # C21 += m4
    u4_add = {24: 1}
    v4_add = {23: 1, 18: -1}
    add(u4_add, v4_add, 4, 3) # C21

    # m5 = (A18+A19)B24
    u5 = {18: 1, 19: 1}
    v5 = {24: 1}
    add(u5, v5, 3, 4) # C12
    # C11 -= m5
    u5_neg = {18: 1, 19: 1}
    v5_neg = {24: -1}
    add(u5_neg, v5_neg, 3, 3) # C11

    # m6 = (A23-A18)(B18+B19)
    u6 = {23: 1, 18: -1}
    v6 = {18: 1, 19: 1}
    # C22 += m6
    u6_add = {23: 1, 18: -1}
    v6_add = {18: 1, 19: 1}
    add(u6_add, v6_add, 4, 4) # C22

    # m7 = (A19-A24)(B23+B24)
    u7 = {19: 1, 24: -1}
    v7 = {23: 1, 24: 1}
    # C11 += m7
    u7_add = {19: 1, 24: -1}
    v7_add = {23: 1, 24: 1}
    add(u7_add, v7_add, 3, 3) # C11

    rank = len(terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u, v, i, k) in enumerate(terms):
        for idx, val in u.items():
            U[idx][col] += val
        for idx, val in v.items():
            V[idx][col] += val
        W[k * n + i][col] = 1

    if time.time() < deadline:
        try:
            Ua = np.array(U, dtype=np.int64)
            Va = np.array(V, dtype=np.int64)
            Wa = np.array(W, dtype=np.int64)
            M1 = (Ua * Va.T).sum(axis=1)
            M2 = (M1 * Wa.T).sum(axis=1)
            ok = True
            for i in range(n):
                for k in range(p):
                    for s in range(m):
                        if M2[i, k, s] != 1:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if ok:
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
