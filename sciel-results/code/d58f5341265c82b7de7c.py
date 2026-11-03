"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 56.
    # Part 1: 4x4x4 multiplication (for first 4 cols of B)
    # Part 2: 4x4x3 multiplication (for last 3 cols of B)
    # We use a rank 28 decomposition for 4x4x4 (Strassen with schoolbook inner)
    # and reuse it for 4x4x3 by padding.
    
    # Helper to generate schoolbook 2x2x2 decomposition
    # Returns U, V, W lists of length 4 (rank 4)
    def schoolbook_2x2x2():
        u = [[0]*4 for _ in range(4)] # 2x2
        v = [[0]*4 for _ in range(4)] # 2x2
        w = [[0]*4 for _ in range(4)] # 2x2
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u[i*2 + j][col] = 1
                    v[j*2 + k][col] = 1
                    w[k*2 + i][col] = 1
                    col += 1
        return u, v, w

    # Helper to generate Strassen 4x4x4 decomposition using schoolbook 2x2x2
    # Returns U, V, W of rank 28
    def strassen_4x4x4():
        # 7 Strassen steps
        # Each step is a 2x2x2 multiplication
        # Total rank 7 * 4 = 28
        
        u_global = [[0]*28 for _ in range(16)] # 4x4
        v_global = [[0]*28 for _ in range(16)] # 4x4
        w_global = [[0]*28 for _ in range(16)] # 4x4
        
        # Strassen formulas for 4x4 = [2x2 blocks]
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to express each Mi as a linear combination of A-blocks and B-blocks.
        # Then use the schoolbook 2x2x2 decomposition for each Mi.
        
        # For each Strassen step s (0..6), we have a rank-4 decomposition.
        # The U factor for step s will have non-zeros in the A-blocks involved.
        # The V factor for step s will have non-zeros in the B-blocks involved.
        # The W factor for step s will be the standard W for 2x2x2, mapped to the 4x4 output blocks.
        
        # Mapping of 2x2 block indices to 4x4 indices:
        # Block (bi, bj) where bi, bj in {0,1}
        # A-block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # In U (size 4x4, flattened 16):
        # Index for A[i, j] is i*4 + j.
        # For block (bi, bj), the local indices (li, lj) map to global (2*bi+li, 2*bj+lj).
        
        # For a given Strassen step, we have coefficients for A-blocks and B-blocks.
        # e.g. M1 = (A11+A22)(B11+B22)
        # A-coeffs: A11: 1, A22: 1
        # B-coeffs: B11: 1, B22: 1
        
        # We will define the 7 steps as (A_coeffs, B_coeffs, W_offset)
        # A_coeffs: list of (block_index, coeff)
        # B_coeffs: list of (block_index, coeff)
        
        steps = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)]),
            # M4
            ([(3, 1)], [(2, 1), (0, -1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ]
        
        # Block indices:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        base_u, base_v, base_w = schoolbook_2x2x2()
        
        for s in range(7):
            a_coeffs, b_coeffs = steps[s]
            
            # For each of the 4 rank-1 terms in the schoolbook 2x2x2:
            for c in range(4):
                global_c = s * 4 + c
                
                # Update U
                # U local index: i*2 + j
                # Global A index: (2*bi + li)*4 + (2*bj + lj)
                for (bi, coeff) in a_coeffs:
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            val = base_u[local_idx][c]
                            if val != 0:
                                global_i = 2 * bi + li
                                global_j = 2 * (bi if False else 0) # Wait, block index mapping
                                # Block 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                                # bi = block // 2, bj = block % 2
                                bj = bi % 2
                                bi = bi // 2
                                global_j = 2 * bj + lj
                                global_u_idx = global_i * 4 + global_j
                                u_global[global_u_idx][global_c] += coeff * val
                                
                # Update V
                for (bi, coeff) in b_coeffs:
                    bj = bi % 2
                    bi = bi // 2
                    for lj in range(2): # j in local 2x2
                        for lk in range(2): # k in local 2x2
                            local_idx = lj * 2 + lk
                            val = base_v[local_idx][c]
                            if val != 0:
                                global_j = 2 * bi + lj
                                global_k = 2 * bj + lk
                                global_v_idx = global_j * 4 + global_k
                                v_global[global_v_idx][global_c] += coeff * val
                                
                # Update W
                # W local index: k*2 + i
                # The output block for this Strassen step is added to specific C-blocks.
                # For M1, it contributes to C11, C12, C21, C22?
                # Strassen reconstruction:
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                # So each Mi contributes to multiple C-blocks with coefficients.
                
                # We need to define the W contribution for each Mi.
                # This part is tricky. The W factor in the decomposition corresponds to the output C.
                # In the schoolbook 2x2x2, W[k*2+i] = 1.
                # This means the local 2x2 result is added to the local 2x2 C-block.
                # But in Strassen, the result of Mi is added to *multiple* C-blocks.
                
                # To handle this, we can't just use the base_w directly.
                # We need to define the W factors for the Strassen steps explicitly.
                
                # Let's define the W contribution for each step s.
                # W_s is a 4x4 matrix (flattened 16) for each of the 4 rank-1 terms in the schoolbook.
                # Actually, the W factor is the same for all 4 rank-1 terms of a given Strassen step?
                # No, the schoolbook W depends on k and i.
                # The Strassen result is a 2x2 matrix.
                # The (li, lk) element of the Strassen result contributes to C.
                
                # Let's define the mapping from Strassen step s and local (li, lk) to global C-blocks.
                # C11 (block 0): M1 + M4 - M5 + M7
                # C12 (block 1): M3 + M5
                # C21 (block 2): M2 + M4
                # C22 (block 3): M1 - M2 + M3 + M6
                
                # For a given local index (li, lk) in the 2x2 result of step s:
                # It contributes to C-block (bi, bj) with coefficient coeff.
                # The global C index is (2*bi + li)*4 + (2*bj + lk)?
                # No, C is 4x4.
                # C[i, k] where i, k in 0..3.
                # Block (bi, bj) covers i in 2*bi..2*bi+1, k in 2*bj..2*bj+1.
                # Local li, lk map to global i = 2*bi + li, k = 2*bj + lk.
                
                # We need a table: contributions[s][li, lk] = list of (block_idx, coeff)
                
                # Let's hardcode the contributions.
                # M1: C11(+1), C22(+1)
                # M2: C21(+1), C22(-1)
                # M3: C12(+1), C22(+1)
                # M4: C11(+1), C21(+1)
                # M5: C11(-1), C12(+1)
                # M6: C22(+1)
                # M7: C11(+1)
                
                # Wait, the formulas are:
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # So for M1: C11(+1), C22(+1)
                # For M2: C21(+1), C22(-1)
                # For M3: C12(+1), C22(+1)
                # For M4: C11(+1), C21(+1)
                # For M5: C11(-1), C12(+1)
                # For M6: C22(+1)
                # For M7: C11(+1)
                
                # For each local (li, lk) in the 2x2 result of Mi:
                # It adds to C[2*bi+li, 2*bj+lk] for each (bi, bj, coeff) in the contribution list.
                
                # We need to build W_global[global_c][k*4+i] += coeff * base_w[lk*2+li][c]
                
                # Let's define the contribution lists.
                contrib = {
                    0: [(0, 1), (3, 1)], # M1
                    1: [(2, 1), (3, -1)], # M2
                    2: [(1, 1), (3, 1)], # M3
                    3: [(0, 1), (2, 1)], # M4
                    4: [(0, -1), (1, 1)], # M5
                    5: [(3, 1)], # M6
                    6: [(0, 1)], # M7
                }
                
                for (bi, bj) in [(0,0), (0,1), (1,0), (1,1)]: # Iterate over local 2x2 result indices?
                    # No, bi, bj here are local indices li, lk.
                    # Let's use li, lk.
                    pass

                # Correct loop:
                for li in range(2):
                    for lk in range(2):
                        local_w_idx = lk * 2 + li
                        base_val = base_w[local_w_idx][c]
                        if base_val == 0:
                            continue
                        for (b_idx, coeff) in contrib[s]:
                            bi = b_idx // 2
                            bj = b_idx % 2
                            global_i = 2 * bi + li
                            global_k = 2 * bj + lk
                            global_w_idx = global_k * 4 + global_i
                            w_global[global_w_idx][global_c] += coeff * base_val

        return u_global, v_global, w_global

    u1, v1, w1 = strassen_4x4x4()
    u2, v2, w2 = strassen_4x4x4()
    
    # Combine for 4x4x7
    # U: 4x4 (16 rows)
    # V: 4x7 (28 rows)
    # W: 7x4 (28 rows)
    
    U = [[0]*56 for _ in range(16)]
    V = [[0]*56 for _ in range(28)]
    W = [[0]*56 for _ in range(28)]
    
    # Part 1: cols 0-3 of B
    for i in range(16):
        for c in range(28):
            U[i][c] = u1[i][c]
            # V index: j*7 + k. Here k is 0..3.
            # u1/v1/w1 are for 4x4x4.
            # v1 has 16 rows (4x4).
            # We need to map v1's rows to V's rows.
            # v1 row index j*4+k.
            # V row index j*7+k.
            # So for each row r in v1:
            # j = r // 4, k = r % 4.
            # V[r*7 + k][c] = v1[r][c]
            pass
            
    for r in range(16):
        j = r // 4
        k = r % 4
        for c in range(28):
            V[j*7 + k][c] = v1[r][c]
            
    # W1: 16 rows (4x4).
    # W index: k*4 + i.
    # W row index k*4+i.
    # For 4x4x7, W has 28 rows (7x4).
    # Index k*4+i where k is 0..6.
    # For part 1, k is 0..3.
    for r in range(16):
        k = r // 4
        i = r % 4
        for c in range(28):
            W[k*4 + i][c] = w1[r][c]
            
    # Part 2: cols 4-6 of B
    # Use u2, v2, w2.
    # Offset c by 28.
    for i in range(16):
        for c in range(28):
            U[i][c+28] = u2[i][c]
            
    for r in range(16):
        j = r // 4
        k = r % 4
        # k in v2 is 0..3.
        # In B, these correspond to cols 4, 5, 6.
        # So global k' = k + 4.
        k_prime = k + 4
        for c in range(28):
            V[j*7 + k_prime][c+28] = v2[r][c]
            
    for r in range(16):
        k = r // 4
        i = r % 4
        # k in w2 is 0..3.
        # In C, these correspond to cols 4, 5, 6.
        # So global k' = k + 4.
        k_prime = k + 4
        for c in range(28):
            W[k_prime*4 + i][c+28] = w2[r][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
