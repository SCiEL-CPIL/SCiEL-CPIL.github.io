"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(x_coeffs, y_coeffs, z_coeffs):
        nonlocal col
        # x_coeffs: dict (i, j) -> coeff
        # y_coeffs: dict (j, k) -> coeff
        # z_coeffs: dict (i, k) -> coeff
        
        for (i, j), c in x_coeffs.items():
            if c != 0:
                U[i * m + j][col] += c
        for (j, k), c in y_coeffs.items():
            if c != 0:
                V[j * p + k][col] += c
        for (i, k), c in z_coeffs.items():
            if c != 0:
                W[k * n + i][col] += c
        col += 1

    # Strassen 2x2 coefficients
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y01 - y11)
    # S4 = x11(y10 - y00)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # Mapping of Strassen step s to Z coefficients (z00, z01, z10, z11)
    # z00 is C[0,0] of the block, z01 is C[0,1], z10 is C[1,0], z11 is C[1,1]
    strassen_z = {
        1: [1, 0, 0, 1],   # S1: z00+=1, z11+=1
        2: [0, 0, 1, -1],  # S2: z10+=1, z11-=1
        3: [0, 1, 0, 1],   # S3: z01+=1, z11+=1
        4: [1, 0, 1, 0],   # S4: z00+=1, z10+=1
        5: [-1, 1, 0, 0],  # S5: z00-=1, z01+=1
        6: [0, 0, 0, 1],   # S6: z11+=1
        7: [1, 0, 0, 0]    # S7: z00+=1
    }
    
    # X coefficients for each M
    # M1: X = A1
    # A1 is A[0:2, 0:2]
    # Indices: (0,0), (0,1), (1,0), (1,1)
    x_m1 = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # For S1, etc. We need specific X for each S.
    
    # It is better to define X and Y for each S explicitly.
    
    def process_m(x_base, y_base, z_blocks):
        # x_base: dict (local 2x2 idx) -> coeff? No, X changes per S.
        # Let's define X and Y for each S.
        pass

    # Let's just list all 28 terms manually or via loops.
    
    # M1 = A1 (B1 + B4)
    # A1: A00, A01, A10, A11
    # B1+B4: B00+B22, B01+B23, B10+B32, B11+B33
    
    # M1 S1: X=(A00+A11), Y=(B00+B22+B11+B33)
    # M1 S2: X=(A10+A11), Y=(B00+B22)
    # M1 S3: X=A00, Y=(B01+B23 - (B11+B33))
    # M1 S4: X=A11, Y=(B10+B32 - (B00+B22))
    # M1 S5: X=(A00+A01), Y=(B11+B33)
    # M1 S6: X=(A10-A00), Y=(B00+B22+B01+B23)
    # M1 S7: X=(A01-A11), Y=(B10+B32+B11+B33)
    
    # Z for M1 affects C11 (block 0,0 of C)
    # C11 indices: (0,0), (0,1), (1,0), (1,1)
    
    # Helper to get X coeffs for Strassen step s given X matrix entries
    def get_x_coeffs(s, x00, x01, x10, x11):
        if s == 1: return {(0,0): 1, (1,1): 1}
        if s == 2: return {(1,0): 1, (1,1): 1}
        if s == 3: return {(0,0): 1}
        if s == 4: return {(1,1): 1}
        if s == 5: return {(0,0): 1, (0,1): 1}
        if s == 6: return {(1,0): 1, (0,0): -1}
        if s == 7: return {(0,1): 1, (1,1): -1}
        return {}

    def get_y_coeffs(s, y00, y01, y10, y11):
        if s == 1: return {(0,0): 1, (1,1): 1}
        if s == 2: return {(0,0): 1}
        if s == 3: return {(0,1): 1, (1,1): -1}
        if s == 4: return {(1,0): 1, (0,0): -1}
        if s == 5: return {(1,1): 1}
        if s == 6: return {(0,0): 1, (0,1): 1}
        if s == 7: return {(1,0): 1, (1,1): 1}
        return {}

    def add_strassen_term(x_map, y_map, z_block_idx, z_scale):
        # x_map: dict (local 0..1, 0..1) -> global (i,j)
        # y_map: dict (local 0..1, 0..1) -> global (j,k)
        # z_block_idx: 0 for C11, 1 for C12
        # z_scale: multiplier for Z coeffs
        
        for s in range(1, 8):
            x_local = get_x_coeffs(s, 0, 0, 0, 0) # dummy, we use maps
            # We need to map local indices to global indices using x_map, y_map
            # x_map[li, lj] = (gi, gj)
            
            x_coeffs = {}
            for (li, lj), c in get_x_coeffs(s, 0, 0, 0, 0).items():
                gi, gj = x_map[(li, lj)]
                x_coeffs[(gi, gj)] = c
                
            y_coeffs = {}
            for (li, lj), c in get_y_coeffs(s, 0, 0, 0, 0).items():
                gj, gk = y_map[(li, lj)]
                y_coeffs[(gj, gk)] = c
                
            z_coeffs = {}
            z_vals = strassen_z[s]
            # z_vals[0] -> z00 (local 0,0), z_vals[1] -> z01 (local 0,1), etc.
            # Map local z to global z
            # z_block_idx 0: C[0:2, 0:2]
            # z_block_idx 1: C[0:2, 2:4]
            
            if z_block_idx == 0:
                # C rows 0,1 cols 0,1
                z_map = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
            else:
                # C rows 0,1 cols 2,3
                z_map = {(0,0): (0,2), (0,1): (0,3), (1,0): (1,2), (1,1): (1,3)}
                
            for (li, lj), c in z_map.items():
                if z_vals[li * 2 + lj] != 0:
                    z_coeffs[c] = z_vals[li * 2 + lj] * z_scale

            add_term(x_coeffs, y_coeffs, z_coeffs)

    # Define maps for M1, M3, M5, M7
    
    # M1: X=A1, Y=B1+B4
    # A1 local (0,0)->A(0,0), (0,1)->A(0,1), (1,0)->A(1,0), (1,1)->A(1,1)
    x_map_m1 = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
    # Y local (0,0)->B(0,0)+B(2,2), etc.
    # Wait, Y is a matrix. The entries of Y are linear combos.
    # My get_y_coeffs returns coeffs for y00, y01...
    # So I need to pass the actual global indices for y00, y01, y10, y11.
    
    # Let's refactor add_strassen_term to take explicit global indices for X and Y entries.
    
    def add_strassen_term_explicit(x_entries, y_entries, z_block_idx, z_scale):
        # x_entries: dict (li, lj) -> (gi, gj)
        # y_entries: dict (li, lj) -> (gj, gk)
        
        for s in range(1, 8):
            x_coeffs = {}
            for (li, lj), c in get_x_coeffs(s, 0, 0, 0, 0).items():
                gi, gj = x_entries[(li, lj)]
                x_coeffs[(gi, gj)] = c
                
            y_coeffs = {}
            for (li, lj), c in get_y_coeffs(s, 0, 0, 0, 0).items():
                gj, gk = y_entries[(li, lj)]
                y_coeffs[(gj, gk)] = c
                
            z_coeffs = {}
            z_vals = strassen_z[s]
            if z_block_idx == 0:
                z_map = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
            else:
                z_map = {(0,0): (0,2), (0,1): (0,3), (1,0): (1,2), (1,1): (1,3)}
                
            for (li, lj), c in z_map.items():
                if z_vals[li * 2 + lj] != 0:
                    z_coeffs[c] = z_vals[li * 2 + lj] * z_scale

            add_term(x_coeffs, y_coeffs, z_coeffs)

    # M1: X=A1
    x_m1 = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
    # Y=B1+B4
    # y00 = B00+B22, y01 = B01+B23, y10 = B10+B32, y11 = B11+B33
    # We need to handle sums in Y.
    # get_y_coeffs returns coeffs for y00 etc.
    # If y00 is B00+B22, then coeff 1 for y00 means 1 for B00 and 1 for B22.
    # So y_entries should map local (li,lj) to a list of global (gj,gk) or a dict.
    
    # Let's change y_entries to dict (li,lj) -> dict (gj,gk) -> coeff
    # And x_entries to dict (li,lj) -> dict (gi,gj) -> coeff
    
    def add_strassen_term_complex(x_entries, y_entries, z_block_idx, z_scale):
        for s in range(1, 8):
            x_coeffs = {}
            for (li, lj), c in get_x_coeffs(s, 0, 0, 0, 0).items():
                for (gi, gj), cc in x_entries[(li, lj)].items():
                    x_coeffs[(gi, gj)] = x_coeffs.get((gi, gj), 0) + c * cc
                    
            y_coeffs = {}
            for (li, lj), c in get_y_coeffs(s, 0, 0, 0, 0).items():
                for (gj, gk), cc in y_entries[(li, lj)].items():
                    y_coeffs[(gj, gk)] = y_coeffs.get((gj, gk), 0) + c * cc
                    
            z_coeffs = {}
            z_vals = strassen_z[s]
            if z_block_idx == 0:
                z_map = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
            else:
                z_map = {(0,0): (0,2), (0,1): (0,3), (1,0): (1,2), (1,1): (1,3)}
                
            for (li, lj), c in z_map.items():
                if z_vals[li * 2 + lj] != 0:
                    z_coeffs[c] = z_vals[li * 2 + lj] * z_scale

            add_term(x_coeffs, y_coeffs, z_coeffs)

    # M1
    x_m1 = {
        (0,0): {(0,0): 1}, (0,1): {(0,1): 1},
        (1,0): {(1,0): 1}, (1,1): {(1,1): 1}
    }
    y_m1 = {
        (0,0): {(0,0): 1, (2,2): 1}, (0,1): {(0,1): 1, (2,3): 1},
        (1,0): {(1,0): 1, (3,2): 1}, (1,1): {(1,1): 1, (3,3): 1}
    }
    add_strassen_term_complex(x_m1, y_m1, 0, 1)

    # M3
    # X = A1
    x_m3 = x_m1
    # Y = B2 - B4
    # y00 = B02 - B22, y01 = B03 - B23, y10 = B12 - B32, y11 = B13 - B33
    y_m3 = {
        (0,0): {(0,2): 1, (2,2): -1}, (0,1): {(0,3): 1, (2,3): -1},
        (1,0): {(1,2): 1, (3,2): -1}, (1,1): {(1,3): 1, (3,3): -1}
    }
    add_strassen_term_complex(x_m3, y_m3, 1, 1)

    # M5
    # X = A1 + A2
    # x00 = A00+A02, x01 = A01+A03, x10 = A10+A12, x11 = A11+A13
    x_m5 = {
        (0,0): {(0,0): 1, (0,2): 1}, (0,1): {(0,1): 1, (0,3): 1},
        (1,0): {(1,0): 1, (1,2): 1}, (1,1): {(1,1): 1, (1,3): 1}
    }
    # Y = B4
    y_m5 = {
        (0,0): {(2,2): 1}, (0,1): {(2,3): 1},
        (1,0): {(3,2): 1}, (1,1): {(3,3): 1}
    }
    # Z: C11 scale -1, C12 scale +1
    # We need to call add_strassen_term_complex twice?
    # No, Z is summed.
    # My add_strassen_term_complex takes one z_block_idx.
    # I need to combine Z from two blocks.
    
    # Let's modify add_term to accept multiple z contributions?
    # Or just call it twice with different z_scale and block?
    # But the X and Y are the same.
    # The rank term is the same scalar product.
    # So we can just add the Z coeffs.
    
    # Let's create a helper to get Z coeffs for a given s, block, scale
    def get_z_coeffs(s, z_block_idx, z_scale):
        z_coeffs = {}
        z_vals = strassen_z[s]
        if z_block_idx == 0:
            z_map = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
        else:
            z_map = {(0,0): (0,2), (0,1): (0,3), (1,0): (1,2), (1,1): (1,3)}
            
        for (li, lj), c in z_map.items():
            if z_vals[li * 2 + lj] != 0:
                z_coeffs[c] = z_vals[li * 2 + lj] * z_scale
        return z_coeffs

    def add_strassen_term_multi_z(x_entries, y_entries, z_specs):
        # z_specs: list of (z_block_idx, z_scale)
        for s in range(1, 8):
            x_coeffs = {}
            for (li, lj), c in get_x_coeffs(s, 0, 0, 0, 0).items():
                for (gi, gj), cc in x_entries[(li, lj)].items():
                    x_coeffs[(gi, gj)] = x_coeffs.get((gi, gj), 0) + c * cc
                    
            y_coeffs = {}
            for (li, lj), c in get_y_coeffs(s, 0, 0, 0, 0).items():
                for (gj, gk), cc in y_entries[(li, lj)].items():
                    y_coeffs[(gj, gk)] = y_coeffs.get((gj, gk), 0) + c * cc
                    
            z_coeffs = {}
            for z_block_idx, z_scale in z_specs:
                z_part = get_z_coeffs(s, z_block_idx, z_scale)
                for (gi, gk), cc in z_part.items():
                    z_coeffs[(gi, gk)] = z_coeffs.get((gi, gk), 0) + cc

            add_term(x_coeffs, y_coeffs, z_coeffs)

    # Re-run M1, M3 with new function
    # M1
    add_strassen_term_multi_z(x_m1, y_m1, [(0, 1)])
    # M3
    add_strassen_term_multi_z(x_m3, y_m3, [(1, 1)])
    # M5
    add_strassen_term_multi_z(x_m5, y_m5, [(0, -1), (1, 1)])
    
    # M7
    # X = A2
    x_m7 = {
        (0,0): {(0,2): 1}, (0,1): {(0,3): 1},
        (1,0): {(1,2): 1}, (1,1): {(1,3): 1}
    }
    # Y = B3 + B4
    # y00 = B20+B22, y01 = B21+B23, y10 = B30+B32, y11 = B31+B33
    y_m7 = {
        (0,0): {(2,0): 1, (2,2): 1}, (0,1): {(2,1): 1, (2,3): 1},
        (1,0): {(3,0): 1, (3,2): 1}, (1,1): {(3,1): 1, (3,3): 1}
    }
    add_strassen_term_multi_z(x_m7, y_m7, [(0, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
