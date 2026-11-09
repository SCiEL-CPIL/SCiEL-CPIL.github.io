"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 98 decomposition using 2x Strassen 4x4
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to fill one 4x4 Strassen decomposition
    # Part 1: B columns 0-3, C columns 0-3
    # Part 2: B columns 4-7, C columns 4-7
    
    # Define Strassen 2x2 coefficients
    # u_A[t][a][b], v_B[t][c][d], w_C[t][a][d]
    u_A = [
        [[1, 0], [0, 1]], # M1: x00 + x11
        [[0, 0], [1, 1]], # M2: x10 + x11
        [[1, 0], [0, 0]], # M3: x00
        [[0, 0], [0, 1]], # M4: x11
        [[1, 1], [0, 0]], # M5: x00 + x01
        [[-1, 0], [1, 0]],# M6: x10 - x00
        [[0, 1], [-1, 1]] # M7: x01 - x11 (Wait, x01 - x11 is [0,1] - [0,0; 0,1]? No. x01 is (0,1), x11 is (1,1). So row 0 col 1 is 1, row 1 col 1 is -1. Yes.)
    ]
    
    v_B = [
        [[1, 0], [0, 1]], # M1: y00 + y11
        [[1, 0], [0, 0]], # M2: y00
        [[0, 1], [0, -1]],# M3: y01 - y11
        [[-1, 0], [1, 0]],# M4: y10 - y00
        [[0, 0], [0, 1]], # M5: y11
        [[1, 1], [0, 0]], # M6: y00 + y01
        [[0, 1], [0, 1]]  # M7: y10 + y11
    ]
    
    w_C = [
        [[1, 0], [0, 1]], # M1: z00 + z11
        [[0, 0], [1, -1]],# M2: z10 - z11
        [[0, 1], [0, 1]], # M3: z01 + z11
        [[1, 0], [1, 0]], # M4: z00 + z10
        [[-1, 1], [0, 0]],# M5: -z00 + z01
        [[0, 0], [0, 1]], # M6: z11
        [[1, 0], [0, 0]]  # M7: z00
    ]
    
    # Define Block Strassen coefficients for 4x4
    # cA[s][I][J][a][b], cB[s][J][K][c][d]
    # s=0..6, I,J,K in 0..1, a,b,c,d in 0..1
    
    # Initialize cA and cB to 0
    cA = [[[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    cB = [[[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0: P1 = (A00 + A11)(B00 + B11)
    for a in range(2):
        for b in range(2):
            cA[0][0][0][a][b] = 1
            cA[0][1][1][a][b] = 1
            cB[0][0][0][a][b] = 1
            cB[0][1][1][a][b] = 1
            
    # s=1: P2 = (A10 + A11)B00
    for a in range(2):
        for b in range(2):
            cA[1][1][0][a][b] = 1
            cA[1][1][1][a][b] = 1
            cB[1][0][0][a][b] = 1
            
    # s=2: P3 = A00(B01 - B11)
    for a in range(2):
        for b in range(2):
            cA[2][0][0][a][b] = 1
            cB[2][0][1][a][b] = 1
            cB[2][1][1][a][b] = -1
            
    # s=3: P4 = A11(B10 - B00)
    for a in range(2):
        for b in range(2):
            cA[3][1][1][a][b] = 1
            cB[3][1][0][a][b] = 1
            cB[3][0][0][a][b] = -1
            
    # s=4: P5 = (A00 + A01)B11
    for a in range(2):
        for b in range(2):
            cA[4][0][0][a][b] = 1
            cA[4][0][1][a][b] = 1
            cB[4][1][1][a][b] = 1
            
    # s=5: P6 = (A10 - A00)(B00 + B01)
    for a in range(2):
        for b in range(2):
            cA[5][1][0][a][b] = 1
            cA[5][0][0][a][b] = -1
            cB[5][0][0][a][b] = 1
            cB[5][0][1][a][b] = 1
            
    # s=6: P7 = (A01 - A11)(B10 + B11)
    for a in range(2):
        for b in range(2):
            cA[6][0][1][a][b] = 1
            cA[6][1][1][a][b] = -1
            cB[6][1][0][a][b] = 1
            cB[6][1][1][a][b] = 1

    # Reconstruction coefficients d[s][I][K]
    d = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    # C00 = P0 + P3 - P4 + P6
    d[0][0][0] = 1
    d[3][0][0] = 1
    d[4][0][0] = -1
    d[6][0][0] = 1
    # C01 = P2 + P4
    d[2][0][1] = 1
    d[4][0][1] = 1
    # C10 = P1 + P3
    d[1][1][0] = 1
    d[3][1][0] = 1
    # C11 = P0 - P1 + P2 + P5
    d[0][1][1] = 1
    d[1][1][1] = -1
    d[2][1][1] = 1
    d[5][1][1] = 1

    def fill_part(b_offset, c_offset):
        for s in range(7):
            for t in range(7):
                g_idx = s * 7 + t
                # Fill U
                for I in range(2):
                    for J in range(2):
                        for a in range(2):
                            for b in range(2):
                                coeff_A = cA[s][I][J][a][b]
                                coeff_u = u_A[t][a][b]
                                if coeff_A != 0 and coeff_u != 0:
                                    val = coeff_A * coeff_u
                                    i = 2 * I + a
                                    j = 2 * J + b
                                    U[i * m + j][g_idx + b_offset] += val
                                    
                # Fill V
                for J in range(2):
                    for K in range(2):
                        for c in range(2):
                            for d in range(2):
                                coeff_B = cB[s][J][K][c][d]
                                coeff_v = v_B[t][c][d]
                                if coeff_B != 0 and coeff_v != 0:
                                    val = coeff_B * coeff_v
                                    j = 2 * J + c
                                    k = 2 * K + d
                                    V[j * p + k + c_offset][g_idx + b_offset] += val
                                    
                # Fill W
                for I in range(2):
                    for K in range(2):
                        coeff_d = d[s][I][K]
                        if coeff_d != 0:
                            for a in range(2):
                                for d_idx in range(2):
                                    coeff_w = w_C[t][a][d_idx]
                                    if coeff_w != 0:
                                        val = coeff_d * coeff_w
                                        i = 2 * I + a
                                        k = 2 * K + d_idx
                                        W[k * n + i + c_offset][g_idx + b_offset] += val

    # Part 1: offset 0
    fill_part(0, 0)
    
    # Part 2: offset 49 for U/V/W indices, but B/C offsets are 4
    # Wait, U is same A, so no offset in U rows, but new columns.
    # The function fill_part adds to g_idx + b_offset.
    # For Part 2, we want columns 49..97.
    # And B columns 4..7, C columns 4..7.
    fill_part(49, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
