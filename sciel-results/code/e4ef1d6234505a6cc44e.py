"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Strassen(2x2) on A[0:2,0:2] (7 terms) + 5 schoolbook terms for remaining A entries.
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 9 x 12
    V = [[0] * rank for _ in range(m * p)]  # 24 x 12
    W = [[0] * rank for _ in range(p * n)]  # 24 x 12

    def Uidx(i, j):
        return i * m + j
    def Vidx(j, k):
        return j * p + k
    def Widx(k, i):
        return k * n + i

    def lin_sum(d1, d2, s2=1):
        out = dict(d1)
        for idx, val in d2.items():
            out[idx] = out.get(idx, 0) + s2 * val
        return out

    def lin_neg(d):
        return {idx: -val for idx, val in d.items()}

    # --- Strassen 2x2 (A[0:2,0:2], B rows j=0,1) ---
    a = {Uidx(0, 0): 1}
    b = {Uidx(0, 1): 1}
    c = {Uidx(1, 0): 1}
    d = {Uidx(1, 1): 1}
    B0 = {Vidx(0, k): 1 for k in range(8)}
    B1 = {Vidx(1, k): 1 for k in range(8)}

    a_plus_b = lin_sum(a, b)
    c_plus_d = lin_sum(c, d)
    c_minus_d = lin_sum(c, lin_neg(d))
    d_minus_c = lin_sum(d, lin_neg(c))
    a_plus_d = lin_sum(a, d)
    c_minus_a = lin_sum(c, lin_neg(a))
    d_plus_b = lin_sum(d, b)
    b_minus_c = lin_sum(b, lin_neg(c))

    # C00 = M5+M4-M2+M6 ; C01 = M1+M2 ; C10 = M3+M4 ; C11 = M1-M3+M5-M7
    C00 = {4: 1, 3: 1, 1: -1, 5: 1}
    C01 = {0: 1, 1: 1}
    C10 = {2: 1, 3: 1}
    C11 = {0: 1, 2: -1, 4: 1, 6: -1}

    Uforms = [a_plus_b, a_plus_b, a, b, a_plus_d, c_minus_a, b_minus_c]
    Vforms = [c_plus_d, d, c_minus_d, d_minus_c, b, d_plus_b, a_plus_d]

    for r in range(7):
        for idx, val in Uforms[r].items():
            U[idx][r] = val
        for idx, val in Vforms[r].items():
            V[idx][r] = val
        for k in range(8):
            W[Widx(k, 0)][r] = C00.get(r, 0) + C01.get(r, 0)
            W[Widx(k, 1)][r] = C10.get(r, 0) + C11.get(r, 0)

    # --- Schoolbook for remaining A entries (5 terms) ---
    # Remaining (i,j): (2,0),(2,1),(2,2),(0,2),(1,2)
    remaining = [(2, 0), (2, 1), (2, 2), (0, 2), (1, 2)]
    for t, (i, j) in enumerate(remaining):
        r = 7 + t
        U[Uidx(i, j)][r] = 1
        for k in range(8):
            V[Vidx(j, k)][r] = 1
            W[Widx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
