"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
Uses recursive Strassen algorithm structure.
"""
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-49 decomposition of 4x4 matrix multiplication.
    This is the best known upper bound using recursive Strassen (7*7).
    A rank-48 decomposition is not known to exist for exact bilinear decomposition over Q.
    
    Shapes:
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    # Initialize factor matrices with zeros
    # U maps A entries to intermediate values
    # V maps B entries to intermediate values
    # W maps intermediate products to C entries
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to get index in flattened A (row-major: i*m + j)
    def a_idx(i, j):
        return i * m + j
        
    # Helper to get index in flattened B (row-major: j*p + k)
    def b_idx(j, k):
        return j * p + k
        
    # Helper to get index in flattened C (row-major: i*p + k)
    # Note: Task description says "W[k*n+i,r]" which implies column-major for C?
    # Let's check the seed program's w_idx: `return k * n + i`.
    # Standard matrix C is usually i (row) and k (col).
    # If W index is `k * n + i`, then for fixed k (col), i (row) varies fastest.
    # This is column-major order for C.
    # However, the seed program's verification logic:
    # `C_computed = np.einsum('ir,kr,kir->ik', u_vals, v_vals, W_r)`
    # Here W_r is reshaped to (4, 4, 49). The indices in einsum are 'kir' -> k, i, r.
    # So W[ k, i, r ] corresponds to the coefficient of product r for C[i, k].
    # In the 1D array W, the index is `k * n + i`.
    # So W_flat[ k * n + i ] is the coefficient for C[i, k].
    def w_idx(i, k):
        return k * n + i

    # Recursive Strassen definitions
    # We divide 4x4 matrices into 2x2 blocks of size 2x2.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # We apply Strassen at the outer level (7 multiplications of 2x2 blocks)
    # Then apply Strassen again at the inner level (7 multiplications of 1x1 scalars)
    # Total rank = 7 * 7 = 49.
    
    # Strassen coefficients for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer term t (0..6) and inner term k (0..6), we have a product term.
    # Let's define the structure of the 7 outer terms.
    # Each outer term is defined by a linear combination of A-blocks and B-blocks.
    # A-blocks are 2x2 matrices. A linear combination of blocks is a 2x2 matrix of linear forms.
    # Specifically, a term is of the form X_t(A) * Y_t(B).
    # X_t is a 2x2 matrix where each entry is a linear combination of entries of A.
    # Y_t is a 2x2 matrix where each entry is a linear combination of entries of B.
    
    # Let's represent the 7 outer Strassen factors as sets of (A_coeff, B_coeff, C_target) tuples?
    # No, it's easier to iterate through the 49 products.
    
    # Define the 7 "outer" linear combinations for A and B.
    # Each combination is a 2x2 array of 2x2 matrices of coefficients.
    # Actually, let's just hardcode the 49 products directly by expanding the recursion.
    
    # Recursive step 1:
    # M_t = (sum_{i,j} a_{t,i,j} A_{i,j}) * (sum_{k,l} b_{t,k,l} B_{k,l})
    # where A_{i,j} is a 2x2 block of A, and B_{k,l} is a 2x2 block of B.
    # The result M_t contributes to C blocks.
    
    # Recursive step 2 (inside M_t):
    # The product of two 2x2 blocks is computed via 7 scalar multiplications.
    # Let A_block = [[A00, A01], [A10, A11]] (local indices 0,1)
    # Let B_block = [[B00, B01], [B10, B11]] (local indices 0,1)
    # The 7 scalar products are:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)(B00)
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)(B11)
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # The result of the 2x2 block product is:
    # C00_local = m1 + m4 - m5 + m7
    # C01_local = m3 + m5
    # C10_local = m2 + m4
    # C11_local = m1 - m2 + m3 + m6
    
    # We need to map these local indices back to global indices.
    
    # Let's define the outer Strassen terms explicitly.
    # Term t=0: (A00+A11) * (B00+B11)
    # Term t=1: (A10+A11) * (B00)
    # Term t=2: A00 * (B01-B11)
    # Term t=3: A11 * (B10-B00)
    # Term t=4: (A00+A01) * (B11)
    # Term t=5: (A10-A00) * (B00+B01)
    # Term t=6: (A01-A11) * (B10+B11)
    
    # A-block global indices:
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # B-block global indices:
    # B00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # B11: rows 2,3 cols 2,3
    
    # C-block global indices:
    # C00: rows 0,1 cols 0,1
    # C01: rows 0,1 cols 2,3
    # C10: rows 2,3 cols 0,1
    # C11: rows 2,3 cols 2,3
    
    # For each outer term t, we have a linear form on A (2x2 block) and B (2x2 block).
    # Let's represent the "effective" A block for term t as a 2x2 matrix of linear combinations of A's entries.
    # Same for B.
    
    # We will build U, V, W by iterating t in 0..6 and k in 0..6.
    
    # Structure of outer terms:
    # t=0: A_eff = A00 + A11, B_eff = B00 + B11
    # t=1: A_eff = A10 + A11, B_eff = B00
    # t=2: A_eff = A00,      B_eff = B01 - B11
    # t=3: A_eff = A11,      B_eff = B10 - B00
    # t=4: A_eff = A00 + A01,B_eff = B11
    # t=5: A_eff = A10 - A00,B_eff = B00 + B01
    # t=6: A_eff = A01 - A11,B_eff = B10 + B11
    
    # Let's define helper to get the global row/col offsets for a 2x2 block.
    # Block (br, bc) where br, bc in {0, 1}
    # Row offset = br * 2, Col offset = bc * 2
    
    # We'll define the coefficients for the inner Strassen (k=0..6) on the effective blocks.
    # The inner Strassen operates on 2x2 blocks of size 2x2? No.
    # The outer terms are 2x2 blocks. The multiplication of two 2x2 blocks requires 7 scalar multiplications IF the blocks are 1x1? No.
    # Wait. The recursive Strassen for 4x4 splits into 2x2 blocks.
    # The multiplication of two 2x2 matrices (the blocks) is performed via Strassen, which uses 7 multiplications of their elements?
    # No, Strassen reduces 8 multiplications to 7. The elements of the 2x2 blocks are scalars (from the 4x4 perspective).
    # So, for each outer term t, we compute the product of two 2x2 matrices (X_t and Y_t) using Strassen.
    # This yields 7 scalar products.
    # Each scalar product corresponds to one rank-1 term in the decomposition.
    
    # So, total 7 outer terms * 7 inner terms = 49 terms.
    
    # Let's define the 7 inner Strassen factors for a generic 2x2 matrix multiplication X*Y = Z.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    # Inner factors (k=0..6):
    # k=0: (x00+x11) * (y00+y11)
    # k=1: (x10+x11) * y00
    # k=2: x00 * (y01-y11)
    # k=3: x11 * (y10-y00)
    # k=4: (x00+x01) * y11
    # k=5: (x10-x00) * (y00+y01)
    # k=6: (x01-x11) * (y10+y11)
    
    # Result Z:
    # z00 = m0 + m3 - m4 + m6
    # z01 = m2 + m4
    # z10 = m1 + m3
    # z11 = m0 - m1 + m2 + m5
    
    # Now we map x and y entries to global A and B entries.
    # For a given outer term t, X_t is a 2x2 matrix where each entry is a linear combination of global A entries.
    # Specifically, X_t[i, j] = sum_{a} c_{t,i,j,a} A[a]
    # We can represent the coefficient of A[a] in X_t[i,j] as a scalar.
    
    # Let's pre-compute the coefficients for U and V.
    
    # List of outer terms: (A_block_combos, B_block_combos)
    # A_block_combos: list of (block_id, coeff)
    # block_id: 0=A00, 1=A01, 2=A10, 3=A11
    
    outer_A = [
        [(0, 1), (3, 1)],   # t=0: A00 + A11
        [(2, 1), (3, 1)],   # t=1: A10 + A11
        [(0, 1)],           # t=2: A00
        [(3, 1)],           # t=3: A11
        [(0, 1), (1, 1)],   # t=4: A00 + A01
        [(2, 1), (0, -1)],  # t=5: A10 - A00
        [(1, 1), (3, -1)],  # t=6: A01 - A11
    ]
    
    outer_B = [
        [(0, 1), (3, 1)],   # t=0: B00 + B11
        [(0, 1)],           # t=1: B00
        [(1, 1), (3, -1)],  # t=2: B01 - B11
        [(2, 1), (0, -1)],  # t=3: B10 - B00
        [(3, 1)],           # t=4: B11
        [(0, 1), (1, 1)],   # t=5: B00 + B01
        [(2, 1), (3, 1)],   # t=6: B10 + B11
    ]
    
    # Block offsets:
    # block_id 0 (00): row_off 0, col_off 0
    # block_id 1 (01): row_off 0, col_off 2
    # block_id 2 (10): row_off 2, col_off 0
    # block_id 3 (11): row_off 2, col_off 2
    
    block_offsets = {
        0: (0, 0),
        1: (0, 2),
        2: (2, 0),
        3: (2, 2)
    }
    
    # Inner Strassen coefficients for X (A side)
    # k=0: x00+x11 -> coeffs for x00:1, x11:1
    # k=1: x10+x11 -> coeffs for x10:1, x11:1
    # k=2: x00 -> coeffs for x00:1
    # k=3: x11 -> coeffs for x11:1
    # k=4: x00+x01 -> coeffs for x00:1, x01:1
    # k=5: x10-x00 -> coeffs for x10:1, x00:-1
    # k=6: x01-x11 -> coeffs for x01:1, x11:-1
    
    inner_X_coeffs = [
        {(0,0): 1, (1,1): 1}, # k=0
        {(1,0): 1, (1,1): 1}, # k=1
        {(0,0): 1},           # k=2
        {(1,1): 1},           # k=3
        {(0,0): 1, (0,1): 1}, # k=4
        {(1,0): 1, (0,0): -1},# k=5
        {(0,1): 1, (1,1): -1} # k=6
    ]
    
    # Inner Strassen coefficients for Y (B side)
    # k=0: y00+y11
    # k=1: y00
    # k=2: y01-y11
    # k=3: y10-y00
    # k=4: y11
    # k=5: y00+y01
    # k=6: y10+y11
    
    inner_Y_coeffs = [
        {(0,0): 1, (1,1): 1}, # k=0
        {(0,0): 1},           # k=1
        {(0,1): 1, (1,1): -1},# k=2
        {(1,0): 1, (0,0): -1},# k=3
        {(1,1): 1},           # k=4
        {(0,0): 1, (0,1): 1}, # k=5
        {(1,0): 1, (1,1): 1}  # k=6
    ]
    
    # Inner Strassen contributions to Z (C side)
    # z00 = m0 + m3 - m4 + m6
    # z01 = m2 + m4
    # z10 = m1 + m3
    # z11 = m0 - m1 + m2 + m5
    
    inner_Z_targets = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, # m0 (k=0) -> z00? No, m0 is one of the products.
        # We need to map the product index k to the Z entries it contributes to.
        # m0 (k=0) contributes to z00 (+1) and z11 (+1)
        # m1 (k=1) contributes to z10 (+1) and z11 (-1)
        # m2 (k=2) contributes to z01 (+1) and z11 (+1)
        # m3 (k=3) contributes to z00 (+1) and z10 (+1)
        # m4 (k=4) contributes to z00 (-1) and z01 (+1)
        # m5 (k=5) contributes to z11 (+1)
        # m6 (k=6) contributes to z00 (+1)
    ]
    
    inner_Z_contrib = [
        {(0,0): 1, (1,1): 1},   # k=0 (m0)
        {(1,0): 1, (1,1): -1},  # k=1 (m1)
        {(0,1): 1, (1,1): 1},   # k=2 (m2)
        {(0,0): 1, (1,0): 1},   # k=3 (m3)
        {(0,0): -1, (0,1): 1},  # k=4 (m4)
        {(1,1): 1},             # k=5 (m5)
        {(0,0): 1}              # k=6 (m6)
    ]
    
    # Outer Strassen contributions to C blocks
    # C00 = M1 + M4 - M5 + M7  (M1=t0, M4=t3, M5=t4, M7=t6)
    # C01 = M3 + M5             (M3=t2, M5=t4)
    # C10 = M2 + M4             (M2=t1, M4=t3)
    # C11 = M1 - M2 + M3 + M6   (M1=t0, M2=t1, M3=t2, M6=t5)
    
    outer_C_contrib = [
        {(0,0): 1, (1,1): 1},   # t=0 (M1) -> C00(+1), C11(+1)
        {(1,0): 1, (1,1): -1},  # t=1 (M2) -> C10(+1), C11(-1)
        {(0,1): 1, (1,1): 1},   # t=2 (M3) -> C01(+1), C11(+1)
        {(0,0): 1, (1,0): 1},   # t=3 (M4) -> C00(+1), C10(+1)
        {(0,0): -1, (0,1): 1},  # t=4 (M5) -> C00(-1), C01(+1)
        {(1,1): 1},             # t=5 (M6) -> C11(+1)
        {(0,0): 1}              # t=6 (M7) -> C00(+1)
    ]
    
    # Build U, V, W
    col = 0
    for t in range(7):
        # Get A and B block definitions for this outer term
        a_blocks = outer_A[t]
        b_blocks = outer_B[t]
        
        for k in range(7):
            # Get inner coefficients
            x_coeffs = inner_X_coeffs[k]
            y_coeffs = inner_Y_coeffs[k]
            
            # Build U and V for this term
            # U[i, j, col] corresponds to coeff of A[i,j] in the X part of term (t,k)
            # X is a 2x2 matrix. X[ii, jj] = sum_{(block, coeff) in a_blocks} coeff * A_block[ii, jj]
            # A_block[ii, jj] has global indices (row_off+ii, col_off+jj)
            
            for (x_ii, x_jj), x_coeff in x_coeffs.items():
                # This x_entry is formed by linear combination of A blocks
                for (block_id, block_coeff) in a_blocks:
                    if block_coeff == 0: continue
                    total_coeff = x_coeff * block_coeff
                    if total_coeff == 0: continue
                    
                    ro, co = block_offsets[block_id]
                    global_i = ro + x_ii
                    global_j = co + x_jj
                    
                    U[global_i * m + global_j, col] += total_coeff
            
            # Similarly for V
            for (y_ii, y_jj), y_coeff in y_coeffs.items():
                for (block_id, block_coeff) in b_blocks:
                    if block_coeff == 0: continue
                    total_coeff = y_coeff * block_coeff
                    if total_coeff == 0: continue
                    
                    ro, co = block_offsets[block_id]
                    # B indices: row is j (of A), col is k (of C)
                    # B_block[ii, jj] corresponds to B[ro+ii, co+jj]
                    global_j = ro + y_ii
                    global_k = co + y_jj
                    
                    V[global_j * p + global_k, col] += total_coeff
            
            # Build W
            # W[global_i, global_k, col]
            # Contribution from outer term t and inner term k
            # Outer contrib: dict (C_block_row, C_block_col) -> coeff
            # Inner contrib: dict (Z_block_row, Z_block_col) -> coeff
            # Z_block_row/col are local 0,1 indices.
            # Global C index:
            # C_block (br, bc) contains C[br*2 + li, bc*2 + lk]
            
            outer_contrib = outer_C_contrib[t]
            inner_contrib = inner_Z_contrib[k]
            
            for (c_br, c_bc), c_coeff in outer_contrib.items():
                for (z_li, z_lj), z_coeff in inner_contrib.items():
                    total_coeff = c_coeff * z_coeff
                    if total_coeff == 0: continue
                    
                    global_i = c_br * 2 + z_li
                    global_k = c_bc * 2 + z_lj
                    
                    W[w_idx(global_i, global_k), col] += total_coeff
            
            col += 1
            
    assert col == 49, f"Rank mismatch: {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
