"""Matrix multiplication 4,4,7 decomposition."""
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    # We'll use Strassen for the 4x4*4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4*4x3 part (last 3 columns of B)
    
    # Strassen 2x2x2 terms: each term has (U, V, W) components
    # Strassen: 7 terms, each is a rank-1 tensor in 2x2x2 space
    # U[i,j], V[i,j], W[i,j] for 2x2 matrices
    
    # The 7 Strassen products for C = A*B where A,B,C are 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4 using Strassen:
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B, split into 2x2 blocks C11, C12, C21, C22
    #
    # Outer Strassen: 7 terms, each involving 2x2 block multiplications
    # Inner: each 2x2*2x2 uses Strassen (7 terms)
    # Total: 7*7 = 49 terms for 4x4*4x4
    
    # For 4x4*4x3 (last 3 cols of B): schoolbook
    # 4*4*3 = 48 terms
    
    # Total rank: 49 + 48 = 97
    
    rank = 97
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0.0] * rank for _ in range(p * n)]  # 28 x 97
    
    # --- Strassen part: 4x4 * 4x4 (first 4 cols of B) ---
    # Block indices: A row-block (br in {0,1}), col-block (bc in {0,1})
    # B row-block (br in {0,1}), col-block (bc in {0,1})
    # A indices: row = br*2 + i, col = bc*2 + j where i,j in {0,1}
    # B indices: row = br*2 + i, col = bc*2 + j where i,j in {0,1}
    # C indices: row = br*2 + i, col = bc*2 + j where i,j in {0,1}
    
    # Outer Strassen terms (7 terms):
    # Each outer term l has:
    #   - A blocks involved: linear combos of A11,A12,A21,A22
    #   - B blocks involved: linear combos of B11,B12,B21,B22
    #   - C blocks: linear combo result
    #
    # Outer Strassen for 2x2 blocks:
    # Let A_blocks = [[A11, A12], [A21, A22]], B_blocks = [[B11, B12], [B21, B22]]
    # Outer M1 = (A11+A22)*(B11+B22)
    # Outer M2 = (A21+A22)*B11
    # Outer M3 = A11*(B12-B22)
    # Outer M4 = A22*(B21-B11)
    # Outer M5 = (A11+A12)*B22
    # Outer M6 = (A21-A11)*(B11+B12)
    # Outer M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each outer term, we have specific A blocks and B blocks.
    # Each "block" is a 2x2 matrix. The inner Strassen decomposes the 2x2*2x2 product.
    
    # Inner Strassen terms for 2x2*2x2:
    # For matrices X = [[x11,x12],[x21,x22]], Y = [[y11,y12],[y21,y22]]
    # Z = X*Y:
    # Inner M1 = (x11+x22)*(y11+y22)
    # Inner M2 = (x21+x22)*y11
    # Inner M3 = x11*(y12-y22)
    # Inner M4 = x22*(y21-y11)
    # Inner M5 = (x11+x12)*y22
    # Inner M6 = (x21-x11)*(y11+y12)
    # Inner M7 = (x12-x22)*(y21+y22)
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Inner Strassen: for each of the 7 inner terms, we need:
    #   U: entries from X (the "A side" 2x2 block)
    #   V: entries from Y (the "B side" 2x2 block)
    #   W: entries from Z (the "C side" 2x2 block)
    
    # Let's define the inner Strassen terms as:
    # Each term t (0..6) has:
    #   U entries: which elements of X and with what coefficients
    #   V entries: which elements of Y and with what coefficients
    #   W entries: which elements of Z and with what coefficients
    
    # Inner Strassen term definitions:
    # Term 0: M1 = (x11+x22)*(y11+y22)
    #   U: x11:1, x22:1
    #   V: y11:1, y22:1
    #   W: z11:1, z22:1
    # Term 1: M2 = (x21+x22)*y11
    #   U: x21:1, x22:1
    #   V: y11:1
    #   W: z21:1, z22:-1
    # Term 2: M3 = x11*(y12-y22)
    #   U: x11:1
    #   V: y12:1, y22:-1
    #   W: z12:1, z22:1
    # Term 3: M4 = x22*(y21-y11)
    #   U: x22:1
    #   V: y21:1, y11:-1
    #   W: z11:1, z21:1
    # Term 4: M5 = (x11+x12)*y22
    #   U: x11:1, x12:1
    #   V: y22:1
    #   W: z11:-1, z12:1
    # Term 5: M6 = (x21-x11)*(y11+y12)
    #   U: x21:1, x11:-1
    #   V: y11:1, y12:1
    #   W: z22:1
    # Term 6: M7 = (x12-x22)*(y21+y22)
    #   U: x12:1, x22:-1
    #   V: y21:1, y22:1
    #   W: z11:1
    
    inner_terms = [
        # (U_entries, V_entries, W_entries)
        # Each is list of (position, coeff)
        # Position in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    # Outer Strassen terms:
    # Each outer term has:
    #   A_block_coeffs: list of (block_idx, coeff) where block_idx 0=A11,1=A12,2=A21,3=A22
    #   B_block_coeffs: list of (block_idx, coeff) where block_idx 0=B11,1=B12,2=B21,3=B22
    #   C_block_coeffs: list of (block_idx, coeff) where block_idx 0=C11,1=C12,2=C21,3=C22
    
    outer_terms = [
        # Outer M1 = (A11+A22)*(B11+B22) -> C11:1, C22:1
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        # Outer M2 = (A21+A22)*B11 -> C21:1, C22:-1
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        # Outer M3 = A11*(B12-B22) -> C12:1, C22:1
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        # Outer M4 = A22*(B21-B11) -> C11:1, C21:1
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        # Outer M5 = (A11+A12)*B22 -> C11:-1, C12:1
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        # Outer M6 = (A21-A11)*(B11+B12) -> C22:1
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        # Outer M7 = (A12-A22)*(B21+B22) -> C11:1
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    # Block to global index mapping for 2x2 blocks in 4x4:
    # Block 0 (11): rows 0,1 cols 0,1
    # Block 1 (12): rows 0,1 cols 2,3
    # Block 2 (21): rows 2,3 cols 0,1
    # Block 3 (22): rows 2,3 cols 2,3
    
    # For A (4x4): element (br*2+dr, bc*2+dc) where br,bc in {0,1}, dr,dc in {0,1}
    # Global row = br*2+dr, global col = bc*2+dc
    # A index in U: row * m + col = (br*2+dr)*4 + (bc*2+dc)
    
    # For B (4x4): element (br*2+dr, bc*2+dc)
    # B index in V: row * p + col. But B is 4x4, so p=4 for this part.
    # Wait, B is 4x7. For the first 4 cols, B[row][col] where col in 0..3
    # B index in V: row * 7 + col
    
    # For C (4x4): element (br*2+dr, bc*2+dc)
    # C index in W: col * n + row = (bc*2+dc)*4 + (br*2+dr)
    # W is p*n = 28 x rank, indexed as W[k*n+i, c] where k=col of C, i=row of C
    
    def block_to_indices(block_idx, is_a=True, is_b_first4=False):
        """Convert block index to list of (local_row, local_col) -> global indices"""
        if is_a:
            br = block_idx // 2
            bc = block_idx % 2
            # For A: global row = br*2+dr, global col = bc*2+dc
            # U index = global_row * m + global_col = (br*2+dr)*4 + (bc*2+dc)
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    u_idx = g_row * m + g_col
                    result.append((dr, dc, u_idx))
            return result
        elif is_b_first4:
            br = block_idx // 2
            bc = block_idx % 2
            # For B (first 4 cols): global row = br*2+dr, global col = bc*2+dc
            # V index = global_row * p + global_col = (br*2+dr)*7 + (bc*2+dc)
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    v_idx = g_row * p + g_col
                    result.append((dr, dc, v_idx))
            return result
        else:
            # For C
            br = block_idx // 2
            bc = block_idx % 2
            # For C: global row = br*2+dr, global col = bc*2+dc
            # W index = global_col * n + global_row = (bc*2+dc)*4 + (br*2+dr)
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    w_idx = g_col * n + g_row
                    result.append((dr, dc, w_idx))
            return result
    
    col = 0
    for o_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for i_idx, (u_entries, v_entries, w_entries) in enumerate(inner_terms):
            # For this combined term:
            # A side: sum over a_blocks of (coeff_a * block_a)
            #   Then inner Strassen U picks specific entries from this 2x2 block
            # B side: sum over b_blocks of (coeff_b * block_b)
            #   Then inner Strassen V picks specific entries from this 2x2 block
            # C side: sum over c_blocks of (coeff_c * block_c)
            #   Then inner Strassen W picks specific entries from this 2x2 block
            
            # Build U contributions
            for (a_bidx, a_coeff) in a_blocks:
                a_indices = block_to_indices(a_bidx, is_a=True)
                for (dr, dc, u_idx) in a_indices:
                    # Check if this (dr,dc) is in u_entries
                    for (pos, u_coeff) in u_entries:
                        if pos == dr*2+dc:
                            U[u_idx][col] += a_coeff * u_coeff
                            break
            
            # Build V contributions
            for (b_bidx, b_coeff) in b_blocks:
                b_indices = block_to_indices(b_bidx, is_a=False, is_b_first4=True)
                for (dr, dc, v_idx) in b_indices:
                    for (pos, v_coeff) in v_entries:
                        if pos == dr*2+dc:
                            V[v_idx][col] += b_coeff * v_coeff
                            break
            
            # Build W contributions
            for (c_bidx, c_coeff) in c_blocks:
                c_indices = block_to_indices(c_bidx, is_a=False, is_b_first4=False)
                for (dr, dc, w_idx) in c_indices:
                    for (pos, w_coeff) in w_entries:
                        if pos == dr*2+dc:
                            W[w_idx][col] += c_coeff * w_coeff
                            break
            
            col += 1
    
    # --- Schoolbook part: 4x4 * 4x3 (last 3 cols of B) ---
    # A is 4x4, B_last is 4x3 (cols 4,5,6 of B)
    # C_last is 4x3 (cols 4,5,6 of C)
    for i in range(n):  # row of A, row of C
        for j in range(m):  # col of A, row of B_last
            for k in range(3):  # col of B_last, col of C_last
                # A[i][j] * B[j][4+k] -> C[i][4+k]
                U[i * m + j][col] = 1.0
                V[j * p + (4+k)][col] = 1.0
                W[(4+k) * n + i][col] = 1.0
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Check time
    elapsed = time.time() - start
    if elapsed > deadline - start:
        pass  # Should not happen with fixed construction
    
    return {"U": U, "V": V, "W": W}
