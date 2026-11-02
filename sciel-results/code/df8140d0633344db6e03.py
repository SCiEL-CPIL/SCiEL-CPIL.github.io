"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Structure:
    - Two 4x4x4 blocks (columns 0-3 and 4-7 of B) using Strassen's algorithm (rank 49 each)
    - One 4x4x1 block (column 8 of B) using schoolbook (rank 16)
    Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll build U, V, W as lists of lists
    # U: (n*m) x rank = 16 x 114
    # V: (m*p) x rank = 36 x 114  
    # W: (p*n) x rank = 36 x 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_4x4_block(col_offset, rank_offset):
        """
        Adds a rank-49 Strassen decomposition for a 4x4x4 multiplication.
        col_offset: the starting column index in B (0 or 4).
        rank_offset: the starting index in the rank dimension.
        """
        # The 7 Strassen products at the block level for 4x4 matrices
        # Each product is X * Y where X, Y are 2x2 matrices formed by linear combinations of A and B blocks.
        # A blocks: A00 (0:2,0:2), A01 (0:2,2:4), A10 (2:4,0:2), A11 (2:4,2:4)
        # B blocks: B00 (0:2,co:co+2), B01 (0:2,co+2:co+4), B10 (2:4,co:co+2), B11 (2:4,co+2:co+4)
        
        # Define the 7 products M1..M7
        # Each product is defined by:
        # 1. The linear combination of A blocks to form X (2x2)
        # 2. The linear combination of B blocks to form Y (2x2)
        # 3. The contributions to C blocks (2x2 each) with coefficients
        
        # A block indices: 00, 01, 10, 11
        # Local indices within a 2x2 block: (0,0), (0,1), (1,0), (1,1)
        
        # Helper to get global A index from block id and local index
        def a_idx(block, lr, lc):
            # block: 00, 01, 10, 11
            # lr, lc: 0 or 1
            r = int(block[0]) * 2 + lr
            c = int(block[1]) * 2 + lc
            return r * m + c
        
        # Helper to get global B index from block id, col_offset, and local index
        def b_idx(block, lr, lc, co):
            r = int(block[0]) * 2 + lr
            c = int(block[1]) * 2 + lc + co
            return r * p + c
        
        # Helper to get global C index from block id and local index
        def c_idx(block, lr, lc):
            r = int(block[0]) * 2 + lr
            c = int(block[1]) * 2 + lc
            return c * n + r
        
        # The 7 Strassen block products:
        # M1 = (A00 + A11)(B00 + B11) -> C00 (+1), C11 (+1)
        # M2 = (A10 + A11)B00         -> C10 (+1)
        # M3 = A00(B01 - B11)         -> C01 (+1)
        # M4 = A11(B10 - B00)         -> C10 (+1), C00 (+1) -- Wait, C00 = M1 + M4 - M5 + M7. So M4 contributes +1 to C00.
        # M5 = (A00 + A01)B11         -> C01 (+1), C00 (-1)
        # M6 = (A10 - A00)(B00 + B01) -> C11 (+1)
        # M7 = (A01 - A11)(B10 + B11) -> C00 (+1)
        
        # Let's define the terms for each Mi.
        # Each Mi is a 2x2 matrix multiplication X*Y.
        # We use the standard 2x2 Strassen decomposition for X*Y.
        
        # 2x2 Strassen terms (u, v, w) for X*Y=Z:
        # T1: X: 00,11=1; Y: 00,11=1; Z: 00,11=1
        # T2: X: 10,11=1; Y: 00=1; Z: 10=1
        # T3: X: 00=1; Y: 01=1,11=-1; Z: 01=1
        # T4: X: 11=1; Y: 10=1,00=-1; Z: 10=1
        # T5: X: 00,01=1; Y: 11=1; Z: 01=1
        # T6: X: 10=1,00=-1; Y: 00,01=1; Z: 11=1
        # T7: X: 01=1,11=-1; Y: 10,11=1; Z: 00=1
        
        strassen_2x2_terms = [
            {
                'x': [(0,0,1), (1,1,1)],
                'y': [(0,0,1), (1,1,1)],
                'z': [(0,0,1), (1,1,1)]
            },
            {
                'x': [(1,0,1), (1,1,1)],
                'y': [(0,0,1)],
                'z': [(1,0,1)]
            },
            {
                'x': [(0,0,1)],
                'y': [(0,1,1), (1,1,-1)],
                'z': [(0,1,1)]
            },
            {
                'x': [(1,1,1)],
                'y': [(1,0,1), (0,0,-1)],
                'z': [(1,0,1)]
            },
            {
                'x': [(0,0,1), (0,1,1)],
                'y': [(1,1,1)],
                'z': [(0,1,1)]
            },
            {
                'x': [(1,0,1), (0,0,-1)],
                'y': [(0,0,1), (0,1,1)],
                'z': [(1,1,1)]
            },
            {
                'x': [(0,1,1), (1,1,-1)],
                'y': [(1,0,1), (1,1,1)],
                'z': [(0,0,1)]
            }
        ]
        
        # Define the 7 products M1..M7
        # Each product is a list of (coeff, block_id) for X and Y
        # And a list of (coeff, c_block_id) for Z contributions
        
        products = [
            # M1
            {
                'x': [(1, '00'), (1, '11')],
                'y': [(1, '00'), (1, '11')],
                'z_contrib': [(1, '00'), (1, '11')]
            },
            # M2
            {
                'x': [(1, '10'), (1, '11')],
                'y': [(1, '00')],
                'z_contrib': [(1, '10')]
            },
            # M3
            {
                'x': [(1, '00')],
                'y': [(1, '01'), (-1, '11')],
                'z_contrib': [(1, '01')]
            },
            # M4
            {
                'x': [(1, '11')],
                'y': [(1, '10'), (-1, '00')],
                'z_contrib': [(1, '10'), (1, '00')]
            },
            # M5
            {
                'x': [(1, '00'), (1, '01')],
                'y': [(1, '11')],
                'z_contrib': [(1, '01'), (-1, '00')]
            },
            # M6
            {
                'x': [(1, '10'), (-1, '00')],
                'y': [(1, '00'), (1, '01')],
                'z_contrib': [(1, '11')]
            },
            # M7
            {
                'x': [(1, '01'), (-1, '11')],
                'y': [(1, '10'), (1, '11')],
                'z_contrib': [(1, '00')]
            }
        ]
        
        current_rank = rank_offset
        
        for prod in products:
            # For each of the 7 inner Strassen terms
            for term in strassen_2x2_terms:
                # Build U, V, W vectors for this rank-1 term
                # U, V, W are already initialized to 0. We just add to the specific column.
                
                # Process X (A side)
                # term['x'] gives local indices and coefficients for the 2x2 matrix X
                # prod['x'] gives which A blocks and coefficients
                # We need to combine these.
                # X is a linear combination of A blocks.
                # X_local[r][c] = sum_{(coeff, block) in prod['x']} coeff * A_block[r][c]
                # Then the rank-1 term uses X_local[r][c] with coefficient term_x[r][c]
                
                # So for each local index (lr, lc) in term['x']:
                for lr, lc, coeff_x in term['x']:
                    for coeff_block, block_id in prod['x']:
                        global_idx = a_idx(block_id, lr, lc)
                        U[global_idx][current_rank] += coeff_x * coeff_block
                        
                # Process Y (B side)
                for lr, lc, coeff_y in term['y']:
                    for coeff_block, block_id in prod['y']:
                        global_idx = b_idx(block_id, lr, lc, col_offset)
                        V[global_idx][current_rank] += coeff_y * coeff_block
                        
                # Process Z (C side)
                # The result of the inner product is a 2x2 matrix Z_local.
                # This Z_local contributes to specific C blocks with specific coefficients.
                for lr, lc, coeff_z in term['z']:
                    for coeff_c, c_block_id in prod['z_contrib']:
                        global_idx = c_idx(c_block_id, lr, lc)
                        W[global_idx][current_rank] += coeff_z * coeff_c
                        
                current_rank += 1
                
    # Add first 4x4x4 block (columns 0-3 of B)
    add_strassen_4x4_block(0, 0)
    
    # Add second 4x4x4 block (columns 4-7 of B)
    add_strassen_4x4_block(4, 49)
    
    # Add 4x4x1 block (column 8 of B) using schoolbook
    # Rank 16
    # A is 4x4, B is 4x1, C is 4x1
    # C[i] = sum_j A[i,j] B[j,0]
    # Rank-1 terms: A[i,j] * B[j,0] -> C[i]
    for i in range(n):
        for j in range(m):
            col = 98 + (i * m + j) # 49+49=98
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1 # B[j, 8] since col 8 is the 9th column (index 8)
            W[8 * n + i][col] = 1 # C[8, i] -> index 8*n + i. Wait.
            # W index is k*n + i. Here k=8 (output column), i is row.
            # C is 4x9. C[i, k].
            # The task says W[k*n+i, r].
            # So for output column k=8, row i, index is 8*4 + i = 32 + i.
            W[8 * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
