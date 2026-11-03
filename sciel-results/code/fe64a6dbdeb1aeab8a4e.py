"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # We will use rank 60:
    # 28 terms for the first 4 columns (Strassen 4x4x4)
    # 32 terms for the last 2 columns (Schoolbook 4x4x2)
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 16 x 60
    V = [[0] * rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * rank for _ in range(p * n)]  # 24 x 60
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for W
    def add_term(term_idx, u_pos, v_pos, w_pos, coeff):
        U[u_pos][term_idx] = 1
        V[v_pos][term_idx] = 1
        W[w_pos][term_idx] = coeff

    # Part 1: Strassen for 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # Strassen blocks for A and B (4x4)
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # Strassen formulas for 2x2 blocks
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output C blocks:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # We need to expand each Mk (2x2 product) into 4 scalar terms.
    # Mk = X * Y.
    # (Mk)[r,s] = sum_t X[r,t] Y[t,s]
    # Each term X[r,t] Y[t,s] contributes to C[i,j][r,s] with coefficient from formula.
    
    # Define block offsets
    # A block (br, bc) is at rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # B block (br, bc) is at rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # C block (br, bc) is at rows 2*br..2*br+1, cols 2*bc..2*bc+1
    
    # Helper to get U index for A element (row, col)
    def u_idx_a(r, c): return r * m + c
    # Helper to get V index for B element (row, col)
    def v_idx_b(r, c): return r * p + c
    # Helper to get W index for C element (row, col)
    def w_idx_c(r, c): return c * n + r  # W[k*n+i] -> k is col of C, i is row of C
    
    term = 0
    
    # Define the 7 Strassen multiplications
    # Each is a list of (X_block, X_sign) and (Y_block, Y_sign)
    # But X and Y are 2x2 matrices formed by linear combos of blocks.
    # It's easier to define the scalar terms directly.
    
    # M1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11, Y = B00 + B11
    # Terms:
    # A00[r,t] * B00[t,s]
    # A00[r,t] * B11[t,s]
    # A11[r,t] * B00[t,s]
    # A11[r,t] * B11[t,s]
    
    def add_strassen_term(x_blocks, y_blocks, c_blocks, signs, current_term):
        # x_blocks: list of (block_id, sign)
        # y_blocks: list of (block_id, sign)
        # c_blocks: list of (block_id, sign) where this M is added
        # We iterate over all scalar products
        for r in range(2):
            for s in range(2):
                for t in range(2):
                    # Sum over x_blocks and y_blocks
                    # Actually, each (x_block, y_block) pair gives a term.
                    # (A_x) * (B_y)
                    for bx, sx in x_blocks:
                        for by, sy in y_blocks:
                            # A element at block bx, pos (r,t)
                            a_row = 2 * bx[0] + r
                            a_col = 2 * bx[1] + t
                            u_pos = u_idx_a(a_row, a_col)
                            
                            # B element at block by, pos (t,s)
                            b_row = 2 * by[0] + t
                            b_col = 2 * by[1] + s
                            v_pos = v_idx_b(b_row, b_col)
                            
                            # C element at block cj, pos (r,s)
                            # The term is added to all C blocks in c_blocks
                            coeff = sx * sy
                            for cj, sc in c_blocks:
                                c_row = 2 * cj[0] + r
                                c_col = 2 * cj[1] + s
                                w_pos = w_idx_c(c_row, c_col)
                                # Add to W
                                W[w_pos][current_term] += coeff * sc
                            
                            # U and V are set to 1 (or sign)
                            # Since U and V are multiplied, we can put sign in U or V or W.
                            # Let's put sign in W for now, but U,V should be 1?
                            # No, if U=1, V=1, then W must be coeff.
                            # But if we have multiple terms for the same U,V, we can't just add W.
                            # Here each scalar term is unique in (u_pos, v_pos)?
                            # No, different (r,s,t) might map to same u,v?
                            # u_pos depends on (bx, r, t). v_pos depends on (by, t, s).
                            # For fixed r,s,t, bx,by, the u,v are fixed.
                            # So we can set U=1, V=1, and W accumulates.
                            
                            U[u_pos][current_term] = 1
                            V[v_pos][current_term] = 1
                            
        return current_term + 1

    # M1
    # X: A00(1), A11(1)
    # Y: B00(1), B11(1)
    # C: C00(1), C11(1)
    term = add_strassen_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], None, term)
    
    # M2
    # X: A10(1), A11(1)
    # Y: B00(1)
    # C: C10(1), C11(1)
    term = add_strassen_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)], None, term)
    
    # M3
    # X: A00(1)
    # Y: B01(1), B11(-1)
    # C: C01(1)
    term = add_strassen_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)], None, term)
    
    # M4
    # X: A11(1)
    # Y: B10(1), B00(-1)
    # C: C00(-1), C10(1)
    term = add_strassen_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)], None, term)
    
    # M5
    # X: A00(1), A01(1)
    # Y: B11(1)
    # C: C01(1), C11(-1)
    term = add_strassen_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (1,1,-1)], None, term)
    
    # M6
    # X: A10(1), A00(-1)
    # Y: B00(1), B01(1)
    # C: C00(1)
    term = add_strassen_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)], None, term)
    
    # M7
    # X: A01(1), A11(-1)
    # Y: B10(1), B11(1)
    # C: C00(-1)
    term = add_strassen_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1)], None, term)
    
    # Part 2: Schoolbook for 4x4x2 (cols 4-5 of B, cols 4-5 of C)
    # Rank 32
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # k is 0,1 corresponding to global col 4,5
                global_k = 4 + k
                u_pos = u_idx_a(i, j)
                v_pos = v_idx_b(j, global_k)
                w_pos = w_idx_c(i, global_k)
                
                U[u_pos][term] = 1
                V[v_pos][term] = 1
                W[w_pos][term] = 1
                term += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
