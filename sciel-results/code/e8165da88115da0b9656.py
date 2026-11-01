import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We split the multiplication A(2x3) * B(3x10) -> C(2x10)
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x10); B_bottom (1x10)]
    # C = A_left * B_top + A_right * B_bottom
    
    # Rank of A_left * B_top (2x2 * 2x10)
    # We split B_top into five 2x2 blocks? No, B_top is 2x10.
    # We can split the 10 columns into five 2-column blocks.
    # Each block is 2x2 * 2x2 -> 2x2.
    # So 5 blocks of 2x2x2.
    # Rank 5 * 7 = 35.
    
    # Rank of A_right * B_bottom (2x1 * 1x10)
    # This is outer product. Rank 2 * 10 = 20.
    
    total_rank = 35 + 20
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col_idx
        U[u_idx][col_idx] = u_val
        V[v_idx][col_idx] = v_val
        W[w_idx][col_idx] = w_val
        col_idx += 1

    # Part 1: A_left (2x2) * B_top (2x10)
    # A_left indices in A: 
    # A is 2x3. 
    # A_left is columns 0,1. Rows 0,1.
    # Indices in U (size 6): 
    # Row 0: 0, 1
    # Row 1: 3, 4
    # So A_left uses U indices 0, 1, 3, 4.
    
    # B_top is rows 0,1 of B. B is 3x10.
    # B_top is 2x10.
    # Indices in V (size 30):
    # Row 0: 0..9
    # Row 1: 10..19
    
    # We process 5 blocks of width 2.
    # Block b (0..4) covers columns 2*b, 2*b+1 of B_top.
    # B indices in V:
    # Row 0: 2*b, 2*b+1
    # Row 1: 10 + 2*b, 10 + 2*b+1
    
    # Output C is 2x10.
    # W indices (size 20):
    # Row 0: 0..9
    # Row 1: 10..19
    # Block b covers columns 2*b, 2*b+1 of C.
    # W indices:
    # Row 0: 2*b, 2*b+1
    # Row 1: 10 + 2*b, 10 + 2*b+1
    
    for b in range(5):
        # Map local 2x2 indices to global indices
        # A_local (0,0) -> U[0], (0,1) -> U[1], (1,0) -> U[3], (1,1) -> U[4]
        # B_local (0,0) -> V[2*b], (0,1) -> V[2*b+1], (1,0) -> V[10+2*b], (1,1) -> V[10+2*b+1]
        # C_local (0,0) -> W[2*b], (0,1) -> W[2*b+1], (1,0) -> W[10+2*b], (1,1) -> W[10+2*b+1]
        
        a00 = 0; a01 = 1; a10 = 3; a11 = 4
        b00 = 2*b; b01 = 2*b+1; b10 = 10+2*b; b11 = 10+2*b+1
        c00 = 2*b; c01 = 2*b+1; c10 = 10+2*b; c11 = 10+2*b+1
        
        # Strassen terms
        # M1 = (a00+a11)(b00+b11)
        # This expands to 4 terms in the tensor decomposition?
        # No, the tensor decomposition uses the intermediate values as linear combinations.
        # The standard tensor decomposition for Strassen uses 7 rank-1 terms.
        # The terms are:
        # 1. (a00+a11) * (b00+b11) -> contributes to C
        # But in the U,V,W matrices, we put the coefficients.
        # The "variables" in the tensor are the entries of A, B, C.
        # The decomposition is T = sum u_c v_c w_c.
        # For Strassen, the 7 terms correspond to the 7 multiplications.
        # The entries of U, V, W are the coefficients of the linear combinations.
        
        # Let's define the 7 terms explicitly for one block.
        # Term 1: (a00+a11)(b00+b11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c01:-1, c10:-1, c11:1 (Wait, C = M1 + M4 - M5 + M7 etc)
        # It is easier to define the 7 rank-1 tensors directly.
        
        # T1 = (a00+a11) (b00+b11) (c00 - c01 - c10 + c11) ? No.
        # The Strassen algorithm computes M1...M7 and then combines them for C.
        # The tensor decomposition is the sum of the 7 products.
        # Each product is rank 1 in the "multiplication" sense, but the linear combinations make it rank 1 in the tensor sense?
        # Yes, each of the 7 terms is a rank-1 tensor in the space of (A, B, C) if we consider the linear combinations as the basis vectors.
        
        # Term 1: u1 = a00+a11, v1 = b00+b11, w1 = c00+c11-c01-c10 ?
        # Let's check: (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
        # This contributes to C00, C01, C10, C11.
        # C00 gets a00b00 + a11b11.
        # C01 gets a00b11 + a11b00.
        # C10 gets a00b00 + a11b11.
        # C11 gets a00b11 + a11b00.
        # So w1 should be c00 + c10 + c01 + c11? No, the signs depend on the combination.
        # Actually, the standard decomposition lists the 7 terms as:
        # 1. (a+d)(e+h) -> C11, C22
        # 2. (c+d)e -> C21
        # 3. a(f-h) -> C12
        # 4. d(g-e) -> C11, C21
        # 5. (a+b)h -> C12, C11
        # 6. (c-a)(e+f) -> C22
        # 7. (b-d)(g+h) -> C11, C22
        
        # We will add these 7 terms.
        
        # Term 1: (a00+a11)(b00+b11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c11:1, c01:-1, c10:-1 (Check: (a+d)(e+h) = ae+ah+de+dh. C11=ae+dh, C22=ae+dh. C12=af+bg. C21=ce+dg.
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C11, +1 to C22.
        # So W: c00:1, c11:1.
        
        add_term(a00, b00, c00, 1, 1, 1) # Placeholder, need to handle linear combos
        # This add_term function is too simple for linear combinations.
        # I need to add multiple non-zeros to U, V, W for one column.
        
    # I need to rewrite the add_term logic to support linear combinations.
    pass

    return {"U": U, "V": V, "W": W}
