"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 4x4x6 multiplication into 8 blocks of 2x2x3 multiplication.
    # Each 2x2x3 multiplication is decomposed into a 2x2x2 part (rank 7) and a 2x2x1 part (rank 4).
    # Total rank = 8 * (7 + 4) = 88.
    
    # Local indices for 2x2x3:
    # i' in 0..1, j' in 0..1, k' in 0..2
    # U local row: i'*2 + j' (0..3)
    # V local row: j'*3 + k' (0..5)
    # W local row: k'*2 + i' (0..5)
    
    # Global indices:
    # I in 0..1, J in 0..1, K in 0..1
    # i = 2*I + i', j = 2*J + j', k = 3*K + k'
    
    # Global U row: i*4 + j = (2I+i')*4 + (2J+j') = 8I + 4i' + 2J + j'
    # Global V row: j*6 + k = (2J+j')*6 + (3K+k') = 12J + 6j' + 3K + k'
    # Global W row: k*4 + i = (3K+k')*4 + (2I+i') = 12K + 4k' + 2I + i'
    
    rank = 88
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    col = 0
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # Add 7 terms for 2x2x2 part (columns k'=0,1)
                # Strassen decomposition for A[2x2] * B[2x2]
                # A entries: a(0,0), b(0,1), c(1,0), d(1,1)
                # B entries: e(0,0), f(0,1), g(1,0), h(1,1)
                # Note: In local 2x2x2, k' is 0 or 1.
                
                # M1 = (a+d)(e+h)
                U[8*I + 4*0 + 2*J + 0][col] = 1 # a
                U[8*I + 4*1 + 2*J + 0][col] = 1 # d? No, d is i'=1, j'=0. Row 8I + 4 + 2J.
                # Let's be precise with rows.
                # Row for a (i'=0, j'=0): 8I + 2J
                # Row for b (i'=0, j'=1): 8I + 2J + 1
                # Row for c (i'=1, j'=0): 8I + 4 + 2J
                # Row for d (i'=1, j'=1): 8I + 4 + 2J + 1
                
                # V rows for e (j'=0, k'=0): 12J + 3K
                # V rows for f (j'=0, k'=1): 12J + 3K + 1
                # V rows for g (j'=1, k'=0): 12J + 6 + 3K
                # V rows for h (j'=1, k'=1): 12J + 6 + 3K + 1
                
                # W rows for C00 (k'=0, i'=0): 12K + 2I
                # W rows for C01 (k'=1, i'=0): 12K + 4 + 2I
                # W rows for C10 (k'=0, i'=1): 12K + 1 + 2I
                # W rows for C11 (k'=1, i'=1): 12K + 4 + 1 + 2I
                
                r_a = 8*I + 2*J
                r_b = 8*I + 2*J + 1
                r_c = 8*I + 4 + 2*J
                r_d = 8*I + 4 + 2*J + 1
                
                r_e = 12*J + 3*K
                r_f = 12*J + 3*K + 1
                r_g = 12*J + 6 + 3*K
                r_h = 12*J + 6 + 3*K + 1
                
                r_C00 = 12*K + 2*I
                r_C01 = 12*K + 4 + 2*I
                r_C10 = 12*K + 1 + 2*I
                r_C11 = 12*K + 4 + 1 + 2*I
                
                # Term 1: (a+d)(e+h)
                U[r_a][col] += 1
                U[r_d][col] += 1
                V[r_e][col] += 1
                V[r_h][col] += 1
                W[r_C00][col] += 1
                W[r_C11][col] += 1
                col += 1
                
                # Term 2: (c+d)e
                U[r_c][col] += 1
                U[r_d][col] += 1
                V[r_e][col] += 1
                W[r_C10][col] += 1
                W[r_C11][col] += -1
                col += 1
                
                # Term 3: a(f-h)
                U[r_a][col] += 1
                V[r_f][col] += 1
                V[r_h][col] += -1
                W[r_C01][col] += 1
                W[r_C11][col] += 1
                col += 1
                
                # Term 4: d(g-e)
                U[r_d][col] += 1
                V[r_g][col] += 1
                V[r_e][col] += -1
                W[r_C00][col] += 1
                W[r_C10][col] += 1
                col += 1
                
                # Term 5: (a+b)h
                U[r_a][col] += 1
                U[r_b][col] += 1
                V[r_h][col] += 1
                W[r_C00][col] += -1
                W[r_C01][col] += 1
                col += 1
                
                # Term 6: (c-a)(e+f)
                U[r_c][col] += 1
                U[r_a][col] += -1
                V[r_e][col] += 1
                V[r_f][col] += 1
                W[r_C11][col] += 1
                col += 1
                
                # Term 7: (b-d)(g+h)
                U[r_b][col] += 1
                U[r_d][col] += -1
                V[r_g][col] += 1
                V[r_h][col] += 1
                W[r_C00][col] += 1
                W[r_C11][col] += 1
                col += 1
                
                # Add 4 terms for 2x2x1 part (column k'=2)
                # B_right is column k'=2.
                # V rows for x (j'=0, k'=2): 12J + 6*0 + 3K + 2 = 12J + 3K + 2
                # V rows for y (j'=1, k'=2): 12J + 6*1 + 3K + 2 = 12J + 6 + 3K + 2
                # W rows for C02 (k'=2, i'=0): 12K + 4*2 + 2I = 12K + 8 + 2I
                # W rows for C12 (k'=2, i'=1): 12K + 4*2 + 2I + 1 = 12K + 8 + 2I + 1
                
                r_x = 12*J + 3*K + 2
                r_y = 12*J + 6 + 3*K + 2
                r_C02 = 12*K + 8 + 2*I
                r_C12 = 12*K + 8 + 2*I + 1
                
                # Term 8: a*x
                U[r_a][col] += 1
                V[r_x][col] += 1
                W[r_C02][col] += 1
                col += 1
                
                # Term 9: b*y
                U[r_b][col] += 1
                V[r_y][col] += 1
                W[r_C02][col] += 1
                col += 1
                
                # Term 10: c*x
                U[r_c][col] += 1
                V[r_x][col] += 1
                W[r_C12][col] += 1
                col += 1
                
                # Term 11: d*y
                U[r_d][col] += 1
                V[r_y][col] += 1
                W[r_C12][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
