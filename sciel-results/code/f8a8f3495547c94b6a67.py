"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-38 decomposition for 2x4 * 4x5 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 sub-blocks and direct (rank 4) for 2x2x1 sub-blocks.
    Structure:
      A (2x4) = [A1 A2], each 2x2
      B (4x5) = [B1; B2], each 2x5
      C = A1*B1 + A2*B2
      For each 2x2 * 2x5: split 2x5 into 2x2, 2x2, 2x1
        Strassen (7) + Strassen (7) + direct (4) = 18
      Total: 18 + 18 = 36... but wait, let me recount.
      
      Actually, let me re-derive. The known best for 2x4x5 might be 38.
      Let me use a simpler block approach:
      A (2x4), B (4x5), C (2x5)
      Split A into two 2x2 blocks, B into two 2x5 blocks.
      C = A1*B1 + A2*B2
      
      For 2x2 * 2x5:
        Split B into columns: B = [B_c0, B_c1, B_c2, B_c3, B_c4]
        This gives 5 separate 2x2 * 1x2 products... no.
        
      Let me think differently. 
      
      Actually the cell finding says rank 38 was achieved. Let me use:
      Split the 4 dimension into 2+2.
      A: 2x4 -> [A1, A2] each 2x2
      B: 4x5 -> [B1; B2] each 2x5
      C = A1*B1 + A2*B2
      
      Now for 2x2 * 2x5:
        We can split the 5 into 2+2+1
        B1 = [B11, B12, B13] where B11, B12 are 2x2, B13 is 2x1
        A1*B1 = A1*B11 + A1*B12 + A1*B13
        A1*B11: 2x2x2 -> Strassen rank 7
        A1*B12: 2x2x2 -> Strassen rank 7  
        A1*B13: 2x2x1 -> direct rank 4
        Subtotal: 18
        
      Similarly A2*B2: 18
      Total: 36
      
      But the known result is 38. Let me check if 36 is valid.
      Actually, I'll just implement the 36-rank construction and verify.
      If it's correct, great. If not, I'll fall back.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U: (n*m, rank) = (8, R)
    # V: (m*p, rank) = (20, R)  
    # W: (p*n, rank) = (10, R) with W[k*n + i, c]
    
    # We'll build a list of (u_row, v_row, w_row, coefficient) tuples
    # where u_row is index into U rows (0..7), v_row into V rows (0..19), w_row into W rows (0..9)
    
    entries = []
    
    def add_entry(u_idx, v_idx, w_idx, coeff=1):
        entries.append((u_idx, v_idx, w_idx, coeff))
    
    # Strassen for 2x2x2
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # C = X*Y
    # M1 = (x0+x3)(y0+y3)
    # M2 = (x2+x3)*y0
    # M3 = x0*(y1-y3)
    # M4 = x3*(y2-y0)
    # M5 = (x0+x1)*y3
    # M6 = (x2-x0)*(y0+y1)
    # M7 = (x1-x3)*(y2+y3)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_222(base_u, base_v, base_w, u_rows, v_rows, w_rows):
        """
        Add Strassen entries for a 2x2 * 2x2 block.
        base_u: starting index for U rows of this block (2 rows: u0, u1)
        base_v: starting index for V rows (4 rows: v0, v1, v2, v3)
        base_w: starting index for W rows (2 rows: w0, w1)
        u_rows: [u0, u1] - the 2 U row indices
        v_rows: [v0, v1, v2, v3] - the 4 V row indices (for Y: y0,y1,y2,y3)
        w_rows: [w0, w1] - the 2 W row indices (for C: c00,c01 or similar)
        """
        x0, x1, x2, x3 = u_rows  # Wait, X is 2x2, so X has 4 elements but only 2 U rows?
        # Actually U maps A entries. For 2x2 block of A, there are 4 elements but U has only 2 rows for this block.
        # This is getting complicated with the global indexing.
        
        # Let me use a different approach: build the full tensor and find a low-rank decomposition.
        # Actually, let me just hardcode the known rank-38 decomposition.
        pass
    
    # Let me take a different, more systematic approach.
    # Build the multiplication tensor and use a known decomposition.
    
    # Actually, let me just implement the schoolbook for now as a fallback,
    # and try to construct the rank-38 version.
    
    # For 2x4 * 4x5:
    # Let A be 2x4, B be 4x5, C be 2x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split j into j1 (0,1) and j2 (2,3)
    # C[i,k] = sum_{j1=0,1} A[i,j1]*B[j1,k] + sum_{j2=0,1} A[i,j2+2]*B[j2+2,k]
    
    # Define A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B1 = B[0:2, :] (2x5), B2 = B[2:4, :] (2x5)
    
    # C = A1 * B1 + A2 * B2
    
    # Now decompose A1 * B1 (2x2 * 2x5) and A2 * B2 (2x2 * 2x5)
    
    # For 2x2 * 2x5:
    # Split the 5 columns into 2+2+1
    # B = [B_a, B_b, B_c] where B_a, B_b are 2x2, B_c is 2x1
    # A*B = A*B_a + A*B_b + A*B_c
    # A*B_a: 2x2x2 -> rank 7 (Strassen)
    # A*B_b: 2x2x2 -> rank 7 (Strassen)
    # A*B_c: 2x2x1 -> rank 4 (direct)
    # Total: 18
    
    # Similarly for A2 * B2: 18
    # Grand total: 36
    
    # Let me implement this. I'll track which global row each local element corresponds to.
    
    # Global indexing:
    # U rows: i*m + j where i in 0..1 (row of A), j in 0..3 (col of A)
    # V rows: j*p + k where j in 0..3 (row of B), k in 0..4 (col of B)
    # W rows: k*n + i where k in 0..4 (col of C), i in 0..1 (row of C)
    
    # For A1 (rows i=0,1, cols j=0,1):
    #   A1[0,0] -> U row 0*4+0 = 0
    #   A1[0,1] -> U row 0*4+1 = 1
    #   A1[1,0] -> U row 1*4+0 = 4
    #   A1[1,1] -> U row 1*4+1 = 5
    
    # For A2 (rows i=0,1, cols j=2,3):
    #   A2[0,0] -> A[0,2] -> U row 0*4+2 = 2
    #   A2[0,1] -> A[0,3] -> U row 0*4+3 = 3
    #   A2[1,0] -> A[1,2] -> U row 1*4+2 = 6
    #   A2[1,1] -> A[1,3] -> U row 1*4+3 = 7
    
    # For B1 (rows j=0,1, cols k=0..4):
    #   B1[0,0] -> V row 0*5+0 = 0
    #   B1[0,1] -> V row 0*5+1 = 1
    #   B1[0,2] -> V row 0*5+2 = 2
    #   B1[0,3] -> V row 0*5+3 = 3
    #   B1[0,4] -> V row 0*5+4 = 4
    #   B1[1,0] -> V row 1*5+0 = 5
    #   B1[1,1] -> V row 1*5+1 = 6
    #   B1[1,2] -> V row 1*5+2 = 7
    #   B1[1,3] -> V row 1*5+3 = 8
    #   B1[1,4] -> V row 1*5+4 = 9
    
    # For B2 (rows j=2,3, cols k=0..4):
    #   B2[0,0] -> B[2,0] -> V row 2*5+0 = 10
    #   B2[0,1] -> B[2,1] -> V row 2*5+1 = 11
    #   ... etc
    
    # W rows: k*n + i
    #   C[0,0] -> W row 0*2+0 = 0
    #   C[0,1] -> W row 1*2+0 = 2
    #   C[0,2] -> W row 2*2+0 = 4
    #   C[0,3] -> W row 3*2+0 = 6
    #   C[0,4] -> W row 4*2+0 = 8
    #   C[1,0] -> W row 0*2+1 = 1
    #   C[1,1] -> W row 1*2+1 = 3
    #   ... etc
    
    # Now, for the 2x2 * 2x2 Strassen block:
    # X = [[x0, x1], [x2, x3]] corresponds to A subblock
    # Y = [[y0, y1], [y2, y3]] corresponds to B subblock
    # C = [[c0, c1], [c2, c3]] = X*Y
    
    # The U indices for X: [u_x0, u_x1, u_x2, u_x3]
    # The V indices for Y: [v_y0, v_y1, v_y2, v_y3]
    # The W indices for C: [w_c0, w_c1, w_c2, w_c3]
    
    # M1 = (x0+x3)(y0+y3): U contributes x0 and x3, V contributes y0 and y3
    # This is a sum of 4 terms: x0*y0, x0*y3, x3*y0, x3*y3
    # In the tensor decomposition, each bilinear product is one column.
    # For M1, we have 4 separate bilinear products? No, Strassen combines them.
    # Actually, in the tensor decomposition, each "M" term is a single bilinear product
    # where the U and V vectors have specific entries.
    
    # For M1 = (x0+x3)(y0+y3):
    #   U vector has 1 at positions x0 and x3
    #   V vector has 1 at positions y0 and y3
    #   This gives x0*y0 + x0*y3 + x3*y0 + x3*y3
    #   So it's a single rank-1 term in the tensor decomposition.
    
    # Similarly for other M terms.
    
    # So for each Strassen 2x2x2 block, we add 7 columns to U, V, W.
    
    # Let me define a function to add a Strassen 2x2x2 block.
    
    def add_strassen_222(u_idx, v_idx, w_idx, entries_list):
        """
        u_idx: list of 4 U row indices [u0, u1, u2, u3] for X = [[x0,x1],[x2,x3]]
        v_idx: list of 4 V row indices [v0, v1, v2, v3] for Y = [[y0,y1],[y2,y3]]
        w_idx: list of 4 W row indices [w0, w1, w2, w3] for C = [[c0,c1],[c2,c3]]
                 where c0=C[0,0], c1=C[0,1], c2=C[1,0], c3=C[1,1]
        """
        x0, x1, x2, x3 = u_idx
        y0, y1, y2, y3 = v_idx
        c0, c1, c2, c3 = w_idx
        
        # M1 = (x0+x3)(y0+y3) -> contributes to c0 and c3
        # M2 = (x2+x3)*y0 -> contributes to c2 and c3
        # M3 = x0*(y1-y3) -> contributes to c1 and c3
        # M4 = x3*(y2-y0) -> contributes to c0 and c2
        # M5 = (x0+x1)*y3 -> contributes to c0 and c1
        # M6 = (x2-x0)*(y0+y1) -> contributes to c3
        # M7 = (x1-x3)*(y2+y3) -> contributes to c0
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each M, we add one column.
        
        # M1: U has +1 at x0, +1 at x3; V has +1 at y0, +1 at y3; W has +1 at c0, +1 at c3
        entries_list.append({
            'U': {x0: 1, x3: 1},
            'V': {y0: 1, y3: 1},
            'W': {c0: 1, c3: 1}
        })
        
        # M2: U has +1 at x2, +1 at x3; V has +1 at y0; W has +1 at c2, +1 at c3
        entries_list.append({
            'U': {x2: 1, x3: 1},
            'V': {y0: 1},
            'W': {c2: 1, c3: 1}
        })
        
        # M3: U has +1 at x0; V has +1 at y1, -1 at y3; W has +1 at c1, +1 at c3
        entries_list.append({
            'U': {x0: 1},
            'V': {y1: 1, y3: -1},
            'W': {c1: 1, c3: 1}
        })
        
        # M4: U has +1 at x3; V has +1 at y2, -1 at y0; W has +1 at c0, +1 at c2
        entries_list.append({
            'U': {x3: 1},
            'V': {y2: 1, y0: -1},
            'W': {c0: 1, c2: 1}
        })
        
        # M5: U has +1 at x0, +1 at x1; V has +1 at y3; W has -1 at c0, +1 at c1
        entries_list.append({
            'U': {x0: 1, x1: 1},
            'V': {y3: 1},
            'W': {c0: -1, c1: 1}
        })
        
        # M6: U has +1 at x2, -1 at x0; V has +1 at y0, +1 at y1; W has +1 at c3
        entries_list.append({
            'U': {x2: 1, x0: -1},
            'V': {y0: 1, y1: 1},
            'W': {c3: 1}
        })
        
        # M7: U has +1 at x1, -1 at x3; V has +1 at y2, +1 at y3; W has +1 at c0
        entries_list.append({
            'U': {x1: 1, x3: -1},
            'V': {y2: 1, y3: 1},
            'W': {c0: 1}
        })
    
    def add_direct_221(u_idx, v_idx, w_idx, entries_list):
        """
        2x2 * 2x1 block. Direct multiplication, rank 4.
        u_idx: [u0, u1, u2, u3] for X = [[x0,x1],[x2,x3]]
        v_idx: [v0, v1] for Y = [[y0],[y1]]
        w_idx: [w0, w1] for C = [[c0],[c1]]
        """
        x0, x1, x2, x3 = u_idx
        y0, y1 = v_idx
        c0, c1 = w_idx
        
        # c0 = x0*y0 + x1*y1
        # c1 = x2*y0 + x3*y1
        
        # 4 bilinear products:
        # x0*y0 -> c0
        entries_list.append({'U': {x0: 1}, 'V': {y0: 1}, 'W': {c0: 1}})
        # x1*y1 -> c0
        entries_list.append({'U': {x1: 1}, 'V': {y1: 1}, 'W': {c0: 1}})
        # x2*y0 -> c1
        entries_list.append({'U': {x2: 1}, 'V': {y0: 1}, 'W': {c1: 1}})
        # x3*y1 -> c1
        entries_list.append({'U': {x3: 1}, 'V': {y1: 1}, 'W': {c1: 1}})
    
    entries = []
    
    # A1 * B1: A1 is rows i=0,1 cols j=0,1; B1 is rows j=0,1 cols k=0..4
    # Split B1 cols into: k=0,1 (2x2), k=2,3 (2x2), k=4 (2x1)
    
    # A1: x0=A[0,0]->U0, x1=A[0,1]->U1, x2=A[1,0]->U4, x3=A[1,1]->U5
    u_A1 = [0, 1, 4, 5]
    
    # B1 cols 0,1: y0=B[0,0]->V0, y1=B[0,1]->V1, y2=B[1,0]->V5, y3=B[1,1]->V6
    v_B1a = [0, 1, 5, 6]
    # C from A1*B1 cols 0,1: c0=C[0,0]->W0, c1=C[0,1]->W2, c2=C[1,0]->W1, c3=C[1,1]->W3
    w_C1a = [0, 2, 1, 3]
    
    add_strassen_222(u_A1, v_B1a, w_C1a, entries)
    
    # B1 cols 2,3: y0=B[0,2]->V2, y1=B[0,3]->V3, y2=B[1,2]->V7, y3=B[1,3]->V8
    v_B1b = [2, 3, 7, 8]
    # C from A1*B1 cols 2,3: c0=C[0,2]->W4, c1=C[0,3]->W6, c2=C[1,2]->W5, c3=C[1,3]->W7
    w_C1b = [4, 6, 5, 7]
    
    add_strassen_222(u_A1, v_B1b, w_C1b, entries)
    
    # B1 col 4: y0=B[0,4]->V4, y1=B[1,4]->V9
    v_B1c = [4, 9]
    # C from A1*B1 col 4: c0=C[0,4]->W8, c1=C[1,4]->W9
    w_C1c = [8, 9]
    
    add_direct_221(u_A1, v_B1c, w_C1c, entries)
    
    # A2 * B2: A2 is rows i=0,1 cols j=2,3; B2 is rows j=2,3 cols k=0..4
    # A2: x0=A[0,2]->U2, x1=A[0,3]->U3, x2=A[1,2]->U6, x3=A[1,3]->U7
    u_A2 = [2, 3, 6, 7]
    
    # B2 cols 0,1: y0=B[2,0]->V10, y1=B[2,1]->V11, y2=B[3,0]->V15, y3=B[3,1]->V16
    v_B2a = [10, 11, 15, 16]
    # C from A2*B2 cols 0,1: c0=C[0,0]->W0, c1=C[0,1]->W2, c2=C[1,0]->W1, c3=C[1,1]->W3
    w_C2a = [0, 2, 1, 3]
    
    add_strassen_222(u_A2, v_B2a, w_C2a, entries)
    
    # B2 cols 2,3: y0=B[2,2]->V12, y1=B[2,3]->V13, y2=B[3,2]->V17, y3=B[3,3]->V18
    v_B2b = [12, 13, 17, 18]
    # C from A2*B2 cols 2,3: c0=C[0,2]->W4, c1=C[0,3]->W6, c2=C[1,2]->W5, c3=C[1,3]->W7
    w_C2b = [4, 6, 5, 7]
    
    add_strassen_222(u_A2, v_B2b, w_C2b, entries)
    
    # B2 col 4: y0=B[2,4]->V14, y1=B[3,4]->V19
    v_B2c = [14, 19]
    # C from A2*B2 col 4: c0=C[0,4]->W8, c1=C[1,4]->W9
    w_C2c = [8, 9]
    
    add_direct_221(u_A2, v_B2c, w_C2c, entries)
    
    rank = len(entries)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, entry in enumerate(entries):
        for row, val in entry['U'].items():
            U[row][col] = val
        for row, val in entry['V'].items():
            V[row][col] = val
        for row, val in entry['W'].items():
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
