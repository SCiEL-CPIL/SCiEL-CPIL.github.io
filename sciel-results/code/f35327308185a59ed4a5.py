"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0

    # 1. Generate T_{1,2,2} decomposition (Rank 4)
    # A: 1x2, B: 2x2, C: 1x2
    # Terms: a0*b00->c0, a1*b10->c0, a0*b01->c1, a1*b11->c1
    T_122_U = []
    T_122_V = []
    T_122_W = []
    # Local A indices: 0, 1
    # Local B indices: 0,1 (row 0), 2,3 (row 1)
    # Local C indices: 0, 1
    T_122_U.append([1, 0]); T_122_V.append([1, 0, 0, 0]); T_122_W.append([1, 0]) # a0*b00
    T_122_U.append([0, 1]); T_122_V.append([0, 0, 1, 0]); T_122_W.append([1, 0]) # a1*b10
    T_122_U.append([1, 0]); T_122_V.append([0, 1, 0, 0]); T_122_W.append([0, 1]) # a0*b01
    T_122_U.append([0, 1]); T_122_V.append([0, 0, 0, 1]); T_122_W.append([0, 1]) # a1*b11

    # 2. Generate T_{2,2,2} decomposition (Rank 7, Strassen)
    # A: 2x2, B: 2x2, C: 2x2
    # Indices: 00, 01, 10, 11
    T_222_U = []
    T_222_V = []
    T_222_W = []
    
    # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
    T_222_U.append([1, 0, 0, 1]); T_222_V.append([1, 0, 0, 1]); T_222_W.append([1, 0, 0, 1])
    # M2 = (A10+A11)B00 -> C00-1, C10+1
    T_222_U.append([0, 0, 1, 1]); T_222_V.append([1, 0, 0, 0]); T_222_W.append([-1, 0, 1, 0])
    # M3 = A00(B10-B11) -> C11-1
    T_222_U.append([1, 0, 0, 0]); T_222_V.append([0, 0, 1, -1]); T_222_W.append([0, 0, 0, -1])
    # M4 = A11(B00-B10) -> C00+1, C11+1
    T_222_U.append([0, 0, 0, 1]); T_222_V.append([1, 0, -1, 0]); T_222_W.append([1, 0, 0, 1])
    # M5 = (A00+A01)B11 -> C01+1, C11+1
    T_222_U.append([1, 1, 0, 0]); T_222_V.append([0, 0, 0, 1]); T_222_W.append([0, 1, 0, 1])
    # M6 = (A10-A00)(B00+B01) -> C10+1, C11-1
    T_222_U.append([-1, 0, 1, 0]); T_222_V.append([1, 1, 0, 0]); T_222_W.append([0, 0, 1, -1])
    # M7 = (A01-A11)(B10+B11) -> C00-1, C01+1
    T_222_U.append([0, 1, 0, -1]); T_222_V.append([0, 0, 1, 1]); T_222_W.append([-1, 1, 0, 0])

    # 3. Construct T_{2,4,4} (Rank 28)
    # Kronecker product of T_{2,2,2} and T_{1,2,2}
    T_244_U = []
    T_244_V = []
    T_244_W = []
    
    for u1, v1, w1 in zip(T_222_U, T_222_V, T_222_W):
        for u2, v2, w2 in zip(T_122_U, T_122_V, T_122_W):
            # U: 2x2 (4) x 1x2 (2) -> 2x4 (8)
            # Index mapping:
            # u1 index: r1*2 + c1
            # u2 index: c2 (r2 is always 0)
            # Global u index: r1*4 + c1*2 + c2
            u_comb = [0] * 8
            for i1 in range(4):
                r1 = i1 // 2
                c1 = i1 % 2
                for i2 in range(2):
                    c2 = i2
                    idx = r1 * 4 + c1 * 2 + c2
                    u_comb[idx] = u1[i1] * u2[i2]
            
            # V: 2x2 (4) x 2x2 (4) -> 4x4 (16)
            # v1 index: r1*2 + c1
            # v2 index: r2*2 + c2
            # Global v index: (r1*2 + r2)*4 + (c1*2 + c2)
            v_comb = [0] * 16
            for i1 in range(4):
                r1 = i1 // 2
                c1 = i1 % 2
                for i2 in range(4):
                    r2 = i2 // 2
                    c2 = i2 % 2
                    idx = (r1 * 2 + r2) * 4 + (c1 * 2 + c2)
                    v_comb[idx] = v1[i1] * v2[i2]
            
            # W: 2x2 (4) x 1x2 (2) -> 2x4 (8)
            # w1 index: r1*2 + c1
            # w2 index: c2
            # Global w index: r1*4 + c1*2 + c2
            w_comb = [0] * 8
            for i1 in range(4):
                r1 = i1 // 2
                c1 = i1 % 2
                for i2 in range(2):
                    c2 = i2
                    idx = r1 * 4 + c1 * 2 + c2
                    w_comb[idx] = w1[i1] * w2[i2]
            
            T_244_U.append(u_comb)
            T_244_V.append(v_comb)
            T_244_W.append(w_comb)

    # 4. Construct T_{1,4,4} (Rank 16)
    # A: 1x4, B: 4x4, C: 1x4
    T_144_U = []
    T_144_V = []
    T_144_W = []
    for j in range(4):
        for k in range(4):
            u = [0] * 4
            v = [0] * 16
            w = [0] * 4
            u[j] = 1
            v[j * 4 + k] = 1
            w[k] = 1
            T_144_U.append(u)
            T_144_V.append(v)
            T_144_W.append(w)

    # 5. Assemble Global Decomposition
    # Block 1: A1(2x4) * B1(4x4) -> C1(2x4, rows 0-1, cols 0-3)
    # Use T_244
    for u, v, w in zip(T_244_U, T_244_V, T_244_W):
        # U: local 2x4 (8) -> global 3x4 (12), rows 0-1
        for l_idx, val in enumerate(u):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 4 + c
            U[g_idx][col_idx] = val
        # V: local 4x4 (16) -> global 4x8 (32), cols 0-3
        for l_idx, val in enumerate(v):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + c
            V[g_idx][col_idx] = val
        # W: local 2x4 (8) -> global 3x8 (24), rows 0-1, cols 0-3
        for l_idx, val in enumerate(w):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + c
            W[g_idx][col_idx] = val
        col_idx += 1

    # Block 2: A1(2x4) * B2(4x4) -> C2(2x4, rows 0-1, cols 4-7)
    # Use T_244
    for u, v, w in zip(T_244_U, T_244_V, T_244_W):
        for l_idx, val in enumerate(u):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 4 + c
            U[g_idx][col_idx] = val
        for l_idx, val in enumerate(v):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + (c + 4)
            V[g_idx][col_idx] = val
        for l_idx, val in enumerate(w):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + (c + 4)
            W[g_idx][col_idx] = val
        col_idx += 1

    # Block 3: A2(1x4) * B1(4x4) -> C3(1x4, row 2, cols 0-3)
    # Use T_144
    for u, v, w in zip(T_144_U, T_144_V, T_144_W):
        for l_idx, val in enumerate(u):
            c = l_idx
            g_idx = 2 * 4 + c
            U[g_idx][col_idx] = val
        for l_idx, val in enumerate(v):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + c
            V[g_idx][col_idx] = val
        for l_idx, val in enumerate(w):
            c = l_idx
            g_idx = 2 * 8 + c
            W[g_idx][col_idx] = val
        col_idx += 1

    # Block 4: A2(1x4) * B2(4x4) -> C4(1x4, row 2, cols 4-7)
    # Use T_144
    for u, v, w in zip(T_144_U, T_144_V, T_144_W):
        for l_idx, val in enumerate(u):
            c = l_idx
            g_idx = 2 * 4 + c
            U[g_idx][col_idx] = val
        for l_idx, val in enumerate(v):
            r = l_idx // 4
            c = l_idx % 4
            g_idx = r * 8 + (c + 4)
            V[g_idx][col_idx] = val
        for l_idx, val in enumerate(w):
            c = l_idx
            g_idx = 2 * 8 + (c + 4)
            W[g_idx][col_idx] = val
        col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
