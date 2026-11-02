"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    def add_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # Local indices 0..3 for 2x2 matrices
        # A: a00, a01, a10, a11
        # B: b00, b01, b10, b11
        # C: c00, c01, c10, c11
        
        # Map local (r, c) to global index
        def a_idx(lr, lc):
            return (a_rows[lr]) * m + (a_cols[lc])
        def b_idx(lr, lc):
            return (b_rows[lr]) * p + (b_cols[lc])
        def c_idx(lr, lc):
            return (c_cols[lc]) * n + (c_rows[lr]) # W[k*n+i, r] -> index is k*n+i

        # Strassen terms
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        terms = [
            # (U_coeff, U_indices, V_coeff, V_indices, W_coeff, W_indices)
            # We need to distribute the sums into U, V, W
            # U has one entry per term? No, U, V, W are matrices.
            # Each term is a rank-1 update.
            # Term t: U[:, t], V[:, t], W[:, t]
            
            # m1
            (1, [a_idx(0,0), a_idx(1,1)], 1, [b_idx(0,0), b_idx(1,1)], 1, [c_idx(0,0), c_idx(1,1)]),
            # Note: The W indices for m1 are c00 and c11.
            # But the coefficient in C is +1 for both.
            
            # m2
            (1, [a_idx(1,0), a_idx(1,1)], 1, [b_idx(0,0)], 1, [c_idx(1,0)]),
            # m2 contributes to c10.
            
            # m3
            (1, [a_idx(0,0)], 1, [b_idx(0,1)], 1, [c_idx(0,1)]),
            (1, [a_idx(0,0)], -1, [b_idx(1,1)], 1, [c_idx(0,1)]),
            # m3 = a00*b01 - a00*b11.
            
            # m4
            (1, [a_idx(1,1)], 1, [b_idx(1,0)], 1, [c_idx(0,0)]),
            (1, [a_idx(1,1)], -1, [b_idx(0,0)], 1, [c_idx(0,0)]),
            # m4 = a11*b10 - a11*b00.
            
            # m5
            (1, [a_idx(0,0), a_idx(0,1)], 1, [b_idx(1,1)], -1, [c_idx(0,0)]),
            # m5 contributes -1 to c00.
            
            # m6
            (1, [a_idx(1,0)], 1, [b_idx(0,0)], 1, [c_idx(1,1)]),
            (1, [a_idx(1,0)], 1, [b_idx(0,1)], 1, [c_idx(1,1)]),
            (1, [a_idx(0,0)], -1, [b_idx(0,0)], 1, [c_idx(1,1)]),
            (1, [a_idx(0,0)], -1, [b_idx(0,1)], 1, [c_idx(1,1)]),
            # m6 = (a10-a00)(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01.
            
            # m7
            (1, [a_idx(0,1)], 1, [b_idx(1,0)], 1, [c_idx(0,0)]),
            (1, [a_idx(0,1)], 1, [b_idx(1,1)], 1, [c_idx(0,0)]),
            (1, [a_idx(1,1)], -1, [b_idx(1,0)], 1, [c_idx(0,0)]),
            (1, [a_idx(1,1)], -1, [b_idx(1,1)], 1, [c_idx(0,0)]),
            # m7 = (a01-a11)(b10+b11).
        ]
        
        # This manual expansion is error-prone. 
        # It's better to define the 7 multiplications and their contributions to C.
        
        # Let's restart the Strassen part with a clearer structure.
        # We will create 7 columns in U, V, W.
        
        # m1 = (a00+a11)(b00+b11)
        # C00 += m1, C11 += m1
        # m2 = (a10+a11)b00
        # C10 += m2
        # m3 = a00(b01-b11)
        # C01 += m3
        # m4 = a11(b10-b00)
        # C00 += m4
        # m5 = (a00+a01)b11
        # C00 -= m5
        # m6 = (a10-a00)(b00+b01)
        # C11 += m6
        # m7 = (a01-a11)(b10+b11)
        # C00 += m7
        
        # We need to add these to the global col.
        
        # Term 1: m1
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # Term 2: m2
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        col += 1
        
        # Term 3: m3
        U[a_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += -1
        W[c_idx(0,1)][col] += 1
        col += 1
        
        # Term 4: m4
        U[a_idx(1,1)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(0,0)][col] += -1
        W[c_idx(0,0)][col] += 1
        col += 1
        
        # Term 5: m5
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += -1
        col += 1
        
        # Term 6: m6
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] += -1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # Term 7: m7
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] += -1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        col += 1

    # Helper to add a 1x2x2 trivial decomposition
    def add_trivial_1x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        nonlocal col
        # A: a0, a1
        # B: b00, b01, b10, b11
        # C: c0, c1
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        # Term 1: a0*b00 -> c0
        U[a_row * m + a_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        W[c_cols[0] * n + c_row][col] += 1
        col += 1
        
        # Term 2: a1*b10 -> c0
        U[a_row * m + a_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[0]][col] += 1
        W[c_cols[0] * n + c_row][col] += 1
        col += 1
        
        # Term 3: a0*b01 -> c1
        U[a_row * m + a_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[1]][col] += 1
        W[c_cols[1] * n + c_row][col] += 1
        col += 1
        
        # Term 4: a1*b11 -> c1
        U[a_row * m + a_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        W[c_cols[1] * n + c_row][col] += 1
        col += 1

    # Define the 8 sub-problems
    
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # 5. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_trivial_1x2(2, [0,1], [0,1], [0,1], 2, [0,1])
    
    # 6. A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    add_trivial_1x2(2, [2,3], [2,3], [0,1], 2, [0,1])
    
    # 7. A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    add_trivial_1x2(2, [0,1], [0,1], [2,3], 2, [2,3])
    
    # 8. A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    add_trivial_1x2(2, [2,3], [2,3], [2,3], 2, [2,3])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
