import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Core: 4x4x4 Strassen (Rank 49)
    # Strip: 4x1x6 Schoolbook (Rank 24)
    # Total Rank: 73
    
    rank_core = 49
    rank_strip = 24
    total_rank = rank_core + rank_strip
    
    U = [[0] * total_rank for _ in range(n * m)] # 20 x 73
    V = [[0] * total_rank for _ in range(m * p)] # 30 x 73
    W = [[0] * total_rank for _ in range(p * n)] # 24 x 73
    
    # --- Core 4x4x4 Strassen ---
    # A is 4x4, B is 4x4 (first 4 cols of B)
    # A indices: 0..15
    # B indices: 0..15 (for the 4x4 part)
    # C indices: 0..15 (for the 4x4 part, mapped to k*n+i where k<4, i<4)
    
    # Strassen 2x2 factors
    # U_2x2: 4x7, V_2x2: 4x7, W_2x2: 4x7
    U_2 = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [1, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    V_2 = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [1, 0, 1, 0, 0, 1, 0], # f
        [1, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, -1, 1, 0, 1]# h
    ]
    W_2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Super products definitions
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # A blocks:
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B blocks (first 4 cols):
    # B00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C blocks (first 4 cols of output):
    # C00: rows 0-1, cols 0-1 -> indices 0,1,4,5 (k*n+i: k=0,1; i=0,1)
    # C01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # We need to map the 2x2 local indices (0,1,2,3) to global 4x4 indices.
    # Local 0: row 0, col 0
    # Local 1: row 0, col 1
    # Local 2: row 1, col 0
    # Local 3: row 1, col 1
    
    # Helper to get global index for a block
    def get_block_indices(block_row, block_col):
        # block_row, block_col in {0,1}
        # Returns list of 4 global indices for the 2x2 block
        # Row-major flattening of 4x4
        indices = []
        for r in range(2):
            for c in range(2):
                global_r = block_row * 2 + r
                global_c = block_col * 2 + c
                indices.append(global_r * 4 + global_c)
        return indices
    
    # Define the 7 super products
    # Each is a tuple of (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # A_block_coeffs: dict {block_idx: coeff}
    # B_block_coeffs: dict {block_idx: coeff}
    # C_block_coeffs: dict {block_idx: coeff}
    
    super_products = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col_idx = 0
    for sp_idx, (A_blocks, B_blocks, C_blocks) in enumerate(super_products):
        # For each of the 7 scalar multiplications in the 2x2 Strassen
        for k in range(7):
            # Extract coefficients for A and B from the 2x2 factors
            # U_2[local_idx][k] is coeff for A_local[local_idx]
            # V_2[local_idx][k] is coeff for B_local[local_idx]
            # W_2[local_idx][k] is coeff for C_local[local_idx]
            
            # We need to distribute these to the global 4x4 indices
            # based on the block mappings.
            
            # For A:
            # The 2x2 input to this super-product is formed by linear combination of A blocks.
            # A_local[0] = sum_{b} A_blocks[b] * A_global[block_b][0]
            # This is getting complicated to map directly.
            
            # Alternative: Just iterate over all 4x4 indices and determine the coefficient.
            # For a given global index ga in A (0..15):
            # It belongs to a specific block ba.
            # Within that block, it has a local index la.
            # The coefficient in U[ga, col_idx] is sum_{b} A_blocks[b] * (1 if b==ba else 0) * U_2[la][k]
            
            for ga in range(16):
                # Determine block and local index
                gr = ga // 4
                gc = ga % 4
                ba = (gr // 2) * 2 + (gc // 2)
                la = (gr % 2) * 2 + (gc % 2)
                
                coeff = 0
                if ba in A_blocks:
                    coeff = A_blocks[ba] * U_2[la][k]
                
                U[ga][col_idx] += coeff
            
            for gb in range(16):
                gr = gb // 4
                gc = gb % 4
                bb = (gr // 2) * 2 + (gc // 2)
                lb = (gr % 2) * 2 + (gc % 2)
                
                coeff = 0
                if bb in B_blocks:
                    coeff = B_blocks[bb] * V_2[lb][k]
                
                V[gb][col_idx] += coeff
            
            for gc_idx in range(16):
                gr = gc_idx // 4
                gc = gc_idx % 4
                bc = (gr // 2) * 2 + (gc // 2)
                lc = (gr % 2) * 2 + (gc % 2)
                
                coeff = 0
                if bc in C_blocks:
                    coeff = C_blocks[bc] * W_2[lc][k]
                
                W[gc_idx][col_idx] += coeff
            
            col_idx += 1
    
    # --- Strip 4x1x6 Schoolbook ---
    # A_strip: 4x1 (cols 4 of A) -> indices 16..19
    # B_strip: 1x6 (row 4 of B) -> indices 16..21
    # C_strip: 4x6 (cols 4-9 of C) -> indices 16..39
    
    # A_strip index i (0..3) corresponds to global A index 16+i
    # B_strip index k (0..5) corresponds to global B index 16+k
    # C_strip index (k, i) corresponds to global C index (4+4) + k*4 + i = 16 + k*4 + i
    
    for i in range(4):
        for k in range(6):
            U[16 + i][col_idx] = 1
            V[16 + k][col_idx] = 1
            W[16 + k * 4 + i][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
