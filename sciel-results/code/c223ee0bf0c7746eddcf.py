"""Matrix multiplication 3x4x6 via two disjoint Strassen blocks (rank 14)."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14

    U = [[0.0] * rank for _ in range(n * m)]   # shape (12, 14)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (24, 14)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (18, 14)

    # Strassen 7-term pattern for a 2x2x2 sub-problem.
    # Local A indices: 00,01,10,11 -> (li, lj)
    # Local B indices: 00,01,10,11 -> (lj, lk)
    # Local C indices: 00,01,10,11 -> (lk, li)
    # Term t has U-coeffs on A, V-coeffs on B, and W-coeffs on C.
    # U_coeffs[t] = dict local (li,lj) -> coeff
    # V_coeffs[t] = dict local (lj,lk) -> coeff
    # W_coeffs[t] = dict local (lk,li) -> coeff

    U_coeffs = [
        {(0,0):1, (1,1):1},
        {(1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,0):1},
    ]
    V_coeffs = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(1,1):1},
        {(0,0):1},
        {(0,0):1, (1,0):-1},
    ]
    # C = sum_t W_t * M_t, where M_t = (sum U A)*(sum V B)
    # C00 = M1 + M4 - M2 - M7
    # C01 = M3 + M5
    # C10 = M6 + M7
    # C11 = M1 - M3 + M2 + M4
    W_coeffs = [
        {(0,0):1, (1,1):1},          # t0: M1 -> C00 + C11
        {(1,0):-1, (1,1):1},         # t1: M2 -> C21? no. M2 -> C11 (+), C21? Let's recheck.
        # Recompute mapping below.
    ]
    # Let me recompute W_coeffs properly from the C formulas.
    # t0 (M1): C00 +=1, C11 +=1
    # t1 (M2): C11 +=1, C00 -=1
    # t2 (M3): C01 +=1, C11 -=1
    # t3 (M4): C00 +=1, C11 +=1
    # t4 (M5): C01 +=1
    # t5 (M6): C10 +=1
    # t6 (M7): C00 -=1, C10 +=1
    W_coeffs = [
        {(0,0):1, (1,1):1},
        {(0,0):-1, (1,1):1},
        {(0,1):1, (1,1):-1},
        {(0,0):1, (1,1):1},
        {(0,1):1},
        {(1,0):1},
        {(0,0):-1, (1,0):1},
    ]

    def add_block(offset_u, offset_v, offset_w, i_map, j_map, k_map, base_col):
        # i_map: local li -> global i ; j_map: local lj -> global j ; k_map: local lk -> global k
        for t in range(7):
            col = base_col + t
            for (li, lj), cu in U_coeffs[t].items():
                gi, gj = i_map[li], j_map[lj]
                U[offset_u + gi * m + gj][col] += cu
            for (lj, lk), cv in V_coeffs[t].items():
                gj, gk = j_map[lj], k_map[lk]
                V[offset_v + gj * p + gk][col] += cv
            for (lk, li), cw in W_coeffs[t].items():
                gk, gi = k_map[lk], i_map[li]
                W[offset_w + gk * n + gi][col] += cw

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # local li=0->i=0, li=1->i=1 ; lj=0->j=0, lj=1->j=1 ; lk=0->k=0, lk=1->k=1
    add_block(
        offset_u=0, offset_v=0, offset_w=0,
        i_map={0:0, 1:1}, j_map={0:0, 1:1}, k_map={0:0, 1:1},
        base_col=0,
    )

    # Block 2: i in {1,2}, j in {2,3}, k in {4,5}
    # local li=0->i=1, li=1->i=2 ; lj=0->j=2, lj=1->j=3 ; lk=0->k=4, lk=1->k=5
    add_block(
        offset_u=0, offset_v=0, offset_w=0,
        i_map={0:1, 1:2}, j_map={0:2, 1:3}, k_map={0:4, 1:5},
        base_col=7,
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
