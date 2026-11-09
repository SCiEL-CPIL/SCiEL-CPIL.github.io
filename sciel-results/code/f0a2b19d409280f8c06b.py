"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank) = (8, rank)
    # V shape (m*p, rank) = (16, rank)
    # W shape (p*n, rank) = (8, rank)
    
    # We will build 28 columns.
    # 4 blocks of 7 Strassen multiplications.
    
    def add_strassen_block(u_idx, v_idx, w_idx):
        """
        Adds 7 rank-1 components corresponding to a 2x2x2 Strassen multiplication.
        u_idx: dict mapping 'x00', 'x01', 'x10', 'x11' to global U indices.
        v_idx: dict mapping 'y00', 'y01', 'y10', 'y11' to global V indices.
        w_idx: dict mapping 'z00', 'z01', 'z10', 'z11' to global W indices.
        """
        # Define the 7 Strassen products
        # Each product is a list of (u_key, coeff), (v_key, coeff), (w_key, coeff)
        # Actually, we just need to populate U, V, W vectors for each of the 7 steps.
        
        steps = []
        
        # 1. (x00 + x11)(y00 + y11)
        steps.append([
            [('x00', 1), ('x11', 1)],
            [('y00', 1), ('y11', 1)],
            [('z00', 1), ('z11', 1)]
        ])
        
        # 2. (x10 + x11)y00
        steps.append([
            [('x10', 1), ('x11', 1)],
            [('y00', 1)],
            [('z10', 1)] # z10 = M2 + M7. Wait, M2 contributes to z10.
                         # Let's check Strassen formulas again.
                         # z00 = M1 - M4 + M5 - M7
                         # z01 = M3 + M4
                         # z10 = M2 + M7
                         # z11 = M1 - M2 + M3 + M6
        ])
        # Correction on W contributions for each M:
        # M1: z00 (+1), z11 (+1)
        # M2: z10 (+1), z11 (-1)
        # M3: z01 (+1), z11 (+1)
        # M4: z00 (-1), z01 (+1)
        # M5: z00 (+1)
        # M6: z11 (+1)
        # M7: z00 (-1), z10 (+1)
        
        # Let's re-verify standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A12 - A11)(B11 + B12)
        # M7 = (A21 - A22)(B12 + B22)
        
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M4
        # C21 = M2 + M7
        # C22 = M1 - M2 + M3 + M6
        
        # My previous mapping:
        # x00=A11, x01=A12, x10=A21, x11=A22
        # y00=B11, y01=B12, y10=B21, y11=B22
        # z00=C11, z01=C12, z10=C21, z11=C22
        
        # Let's update the steps with correct W coefficients.
        
        steps = []
        
        # M1
        steps.append([
            [('x00', 1), ('x11', 1)],
            [('y00', 1), ('y11', 1)],
            [('z00', 1), ('z11', 1)]
        ])
        
        # M2
        steps.append([
            [('x10', 1), ('x11', 1)],
            [('y00', 1)],
            [('z10', 1), ('z11', -1)]
        ])
        
        # M3
        steps.append([
            [('x00', 1)],
            [('y10', 1), ('y11', -1)],
            [('z01', 1), ('z11', 1)]
        ])
        
        # M4
        steps.append([
            [('x11', 1)],
            [('y01', 1), ('y00', -1)],
            [('z00', -1), ('z01', 1)]
        ])
        
        # M5
        steps.append([
            [('x00', 1), ('x01', 1)],
            [('y11', 1)],
            [('z00', 1)]
        ])
        
        # M6
        steps.append([
            [('x01', 1), ('x00', -1)],
            [('y00', 1), ('y01', 1)],
            [('z11', 1)]
        ])
        
        # M7
        steps.append([
            [('x10', 1), ('x11', -1)],
            [('y01', 1), ('y11', 1)],
            [('z00', -1), ('z10', 1)]
        ])
        
        for u_part, v_part, w_part in steps:
            # Create new column vectors
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for key, val in u_part:
                idx = u_idx[key]
                u_vec[idx] = val
                
            for key, val in v_part:
                idx = v_idx[key]
                v_vec[idx] = val
                
            for key, val in w_part:
                idx = w_idx[key]
                w_vec[idx] = val
                
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)

    U = []
    V = []
    W = []
    
    # Block 1: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # A indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C indices: (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3  (W[k*n+i])
    u1 = {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5}
    v1 = {'y00': 0, 'y01': 1, 'y10': 4, 'y11': 5}
    w1 = {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    add_strassen_block(u1, v1, w1)
    
    # Block 2: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # A indices: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B indices: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # C indices: Same as Block 1
    u2 = {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7}
    v2 = {'y00': 8, 'y01': 9, 'y10': 12, 'y11': 13}
    w2 = {'z00': 0, 'z01': 2, 'z10': 1, 'z11': 3}
    add_strassen_block(u2, v2, w2)
    
    # Block 3: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # A indices: Same as Block 1
    # B indices: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # C indices: (0,2)->4, (0,3)->6, (1,2)->5, (1,3)->7
    u3 = {'x00': 0, 'x01': 1, 'x10': 4, 'x11': 5}
    v3 = {'y00': 2, 'y01': 3, 'y10': 6, 'y11': 7}
    w3 = {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    add_strassen_block(u3, v3, w3)
    
    # Block 4: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    # A indices: Same as Block 2
    # B indices: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    # C indices: Same as Block 3
    u4 = {'x00': 2, 'x01': 3, 'x10': 6, 'x11': 7}
    v4 = {'y00': 10, 'y01': 11, 'y10': 14, 'y11': 15}
    w4 = {'z00': 4, 'z01': 6, 'z10': 5, 'z11': 7}
    add_strassen_block(u4, v4, w4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
