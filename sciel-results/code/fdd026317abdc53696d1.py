"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # We decompose 4x4x5 into 4x4x4 and 4x4x1
    
    # Part 1: 4x4x4 using Strassen (Rank 28)
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    
    rank1 = 28
    rank2 = 16
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x 44
    V = [[0] * total_rank for _ in range(m * p)] # 20 x 44
    W = [[0] * total_rank for _ in range(p * n)] # 20 x 44
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w, col):
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w

    # --- Part 1: 4x4x4 Strassen ---
    # Strassen for 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map block indices to global indices
    # A block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Global A index: i*4 + j
    
    # Define the 7 Strassen multiplications
    # Each is a 2x2 matrix product X*Y
    # X and Y are linear combinations of A and B blocks
    
    # We will iterate through the 7 terms, and for each, expand the 2x2 product into 4 scalar mults.
    
    strassen_terms = [
        # (X_block_coeffs, Y_block_coeffs)
        # X is 2x2, Y is 2x2
        # Coeffs are tuples (block_i, block_j, coeff)
        # Block indices: 00, 01, 10, 11
    ]
    
    # M1
    # X = A00 + A11
    # Y = B00 + B11
    # M2
    # X = A10 + A11
    # Y = B00
    # M3
    # X = A00
    # Y = B01 - B11
    # M4
    # X = A11
    # Y = B10 - B00
    # M5
    # X = A00 + A01
    # Y = B11
    # M6
    # X = A10 - A00
    # Y = B00 + B01
    # M7
    # X = A01 - A11
    # Y = B10 + B11
    
    # Output C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the 28 terms.
    # Each term is a scalar multiplication: A_entry * B_entry -> C_entry
    
    col = 0
    for t in range(7):
        # Determine X and Y block compositions
        if t == 0:
            # M1
            X_blocks = {0: 1, 3: 1} # A00, A11 (0,0 and 1,1)
            Y_blocks = {0: 1, 3: 1} # B00, B11
            C_coeffs = {0: 1, 1: 0, 2: 0, 3: 1} # C00, C01, C10, C11 coeffs for M1
            # C00 += M1, C11 += M1
        elif t == 1:
            # M2
            X_blocks = {2: 1, 3: 1} # A10, A11
            Y_blocks = {0: 1} # B00
            C_coeffs = {0: 0, 1: 0, 2: 1, 3: -1} # C10 += M2, C11 -= M2
        elif t == 2:
            # M3
            X_blocks = {0: 1} # A00
            Y_blocks = {1: 1, 3: -1} # B01, B11
            C_coeffs = {0: 0, 1: 1, 2: 0, 3: 1} # C01 += M3, C11 += M3
        elif t == 3:
            # M4
            X_blocks = {3: 1} # A11
            Y_blocks = {2: 1, 0: -1} # B10, B00
            C_coeffs = {0: 1, 1: 0, 2: 1, 3: 0} # C00 += M4, C10 += M4
        elif t == 4:
            # M5
            X_blocks = {0: 1, 1: 1} # A00, A01
            Y_blocks = {3: 1} # B11
            C_coeffs = {0: -1, 1: 1, 2: 0, 3: 0} # C00 -= M5, C01 += M5
        elif t == 5:
            # M6
            X_blocks = {2: 1, 0: -1} # A10, A00
            Y_blocks = {0: 1, 1: 1} # B00, B01
            C_coeffs = {0: 0, 1: 0, 2: 0, 3: 1} # C11 += M6
        elif t == 6:
            # M7
            X_blocks = {1: 1, 3: -1} # A01, A11
            Y_blocks = {2: 1, 3: 1} # B10, B11
            C_coeffs = {0: 1, 1: 0, 2: 0, 3: 0} # C00 += M7
            
        # Expand X*Y
        # X is 2x2, Y is 2x2
        # X_{ab} = sum_{bk} X_blocks[bk] * A_{2*bk_i + a, 2*bk_j + b}
        # Y_{bc} = sum_{bk} Y_blocks[bk] * B_{2*bk_i + b, 2*bk_j + c}
        
        # We iterate over the 4 elements of the product matrix M = X*Y
        for r in range(2): # row of M
            for c in range(2): # col of M
                # M_{rc} = sum_{k=0}^1 X_{rk} Y_{kc}
                for k in range(2):
                    # X_{rk} involves A entries
                    # Y_{kc} involves B entries
                    # We need to combine the linear combinations
                    
                    # Collect A coeffs for X_{rk}
                    a_coeffs = []
                    for bk, coeff in X_blocks.items():
                        bi, bj = divmod(bk, 2)
                        # A block (bi, bj) is at rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1
                        # Entry (r, k) of this block is A_{2*bi + r, 2*bj + k}
                        ai = 2 * bi + r
                        aj = 2 * bj + k
                        a_coeffs.append((ai * 4 + aj, coeff))
                    
                    # Collect B coeffs for Y_{kc}
                    b_coeffs = []
                    for bk, coeff in Y_blocks.items():
                        bi, bj = divmod(bk, 2)
                        # B block (bi, bj) is at rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1
                        # Entry (k, c) of this block is B_{2*bi + k, 2*bj + c}
                        bi_idx = 2 * bi + k
                        bj_idx = 2 * bj + c
                        b_coeffs.append((bi_idx * 4 + bj_idx, coeff))
                    
                    # Combine
                    # This creates rank |a_coeffs| * |b_coeffs| terms
                    for a_idx, a_coeff in a_coeffs:
                        for b_idx, b_coeff in b_coeffs:
                            # Output C entry
                            # M is at block position (r, c) of the 2x2 block matrix
                            # So C block (r, c) gets contribution M_{rc}
                            # But we also have C_coeffs for the whole M term
                            # Wait, the C_coeffs apply to the whole 2x2 matrix M.
                            # So C_{2*r + r', 2*c + c'} gets C_coeffs[r*2+c] * M_{rc}
                            # Here r' and c' are the local indices within the C block.
                            # But M_{rc} is a scalar.
                            # The output is a 2x2 block.
                            # The term M_{rc} contributes to C_{2*r, 2*c}, C_{2*r, 2*c+1}, etc?
                            # No. M is the product of two 2x2 blocks.
                            # M is a 2x2 matrix.
                            # M_{rc} is an element of M.
                            # This element M_{rc} contributes to the C matrix at position (2*r + r, 2*c + c)?
                            # No. The blocks are 2x2.
                            # C_block(r, c) = M.
                            # So C_{2*r + i, 2*c + j} += M_{ij}.
                            # Here we are calculating M_{rc} (using r, c as row/col of M).
                            # So this scalar contributes to C_{2*r + r, 2*c + c}?
                            # No, the indices of M are 0..1.
                            # Let's call M's indices mr, mc.
                            # We are calculating M[mr, mc].
                            # This value is added to C_block(r, c) at position (mr, mc).
                            # So global C index is (2*r + mr, 2*c + mc).
                            
                            # In our loop, r and c are the block indices (0 or 1).
                            # k is the summation index.
                            # We are calculating M[r, c].
                            # So global C row = 2*r + r? No.
                            # The loop variable r is the row of M.
                            # So global C row = 2*r + r.
                            # Global C col = 2*c + c.
                            
                            c_row = 2 * r + r
                            c_col = 2 * c + c
                            
                            c_idx = c_col * 4 + c_row # W index: k*n + i. Here k=c_col, i=c_row.
                            
                            # Wait, W index convention: W[k*n + i].
                            # k is column of C, i is row of C.
                            # So c_idx = c_col * 4 + c_row.
                            
                            coeff = a_coeff * b_coeff * C_coeffs[r * 2 + c]
                            
                            if coeff != 0:
                                add_term(a_idx, b_idx, c_idx, coeff, 1, 1, col)
                                
                            if time.time() > deadline:
                                return {"U": U, "V": V, "W": W}
                            
                            col += 1
                            if col >= rank1:
                                break
                        if col >= rank1: break
                    if col >= rank1: break
                if col >= rank1: break
            if col >= rank1: break
        
    # --- Part 2: 4x4x1 Schoolbook ---
    # C_col4 = A * B_col4
    # C_{i, 4} = sum_{j=0}^3 A_{i, j} B_{j, 4}
    for i in range(4):
        for j in range(4):
            # A index: i*4 + j
            a_idx = i * 4 + j
            # B index: j*5 + 4
            b_idx = j * 5 + 4
            # C index: 4*4 + i
            c_idx = 16 + i
            
            add_term(a_idx, b_idx, c_idx, 1, 1, 1, col)
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            if col >= total_rank:
                break
        if col >= total_rank: break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
