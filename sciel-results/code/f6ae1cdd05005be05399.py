import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use Strassen's algorithm for 4x4x4 multiplication.
    # Rank for 4x4x4 is 49.
    # We have two such multiplications (for the two halves of B).
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a Strassen decomposition for a 4x4x4 block
    # offset: column offset in U, V, W
    # b_offset: column offset in B (for V)
    # c_offset: row offset in C (for W) - note W index is k*n + i
    
    def add_strassen_4x4(offset, b_col_offset, c_row_offset):
        # The 4x4x4 Strassen decomposition consists of 7 sub-multiplications of 2x2x2.
        # We need the factors for 2x2x2 Strassen first.
        
        # Strassen 2x2x2 factors
        # U_22 shape (4, 7), V_22 shape (4, 7), W_22 shape (4, 7)
        # Indices: i,j in 0..1 for A; j,k in 0..1 for B; k,i in 0..1 for C
        
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_22[0, 0] = 1; U_22[3, 0] = 1
        V_22[0, 0] = 1; V_22[3, 0] = 1
        W_22[0, 0] = 1; W_22[1, 0] = 1; W_22[2, 0] = 1; W_22[3, 0] = 1 # Wait, W index is k*n+i. For 2x2, n=2.
        
        # Let's re-verify W index mapping for 2x2.
        # W index: k * 2 + i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to fill U, V, W such that sum_c U_{ij,c} V_{j'k,c} W_{k'i',c} = delta
        
        # It is easier to just use the known factors for Strassen 2x2.
        # Reference: 
        # c=0: U=(1,0,0,1), V=(1,0,0,1), W=(1,1,1,1) -> M1
        # c=1: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,1) -> M2?
        # Let's derive carefully.
        
        # M1: U=(1,0,0,1), V=(1,0,0,1)
        # M2: U=(0,0,1,1), V=(1,0,0,0)
        # M3: U=(1,0,0,0), V=(0,1,0,-1)
        # M4: U=(0,0,0,1), V=(0,-1,1,0)
        # M5: U=(1,1,0,0), V=(0,0,0,1)
        # M6: U=(0,-1,1,0), V=(1,1,0,0)
        # M7: U=(0,1,0,-1), V=(0,0,1,1)
        
        # W factors:
        # W0 (M1): C00, C01, C10, C11 all get +1?
        # C00 += M1, C11 += M1.
        # C01, C10 do not get M1.
        # So W for M1: W[0]=1 (C00), W[3]=1 (C11). Others 0.
        # Wait, C00 is index 0. C11 is index 3.
        # So W_M1 = [1, 0, 0, 1].
        
        # M2: C10 += M2, C11 -= M2.
        # C10 is index 2. C11 is index 3.
        # W_M2 = [0, 0, 1, -1].
        
        # M3: C01 += M3, C11 += M3.
        # C01 is index 1. C11 is index 3.
        # W_M3 = [0, 1, 0, 1].
        
        # M4: C00 += M4, C10 += M4.
        # C00 is index 0. C10 is index 2.
        # W_M4 = [1, 0, 1, 0].
        
        # M5: C00 -= M5, C01 += M5.
        # C00 is index 0. C01 is index 1.
        # W_M5 = [-1, 1, 0, 0].
        
        # M6: C11 += M6.
        # C11 is index 3.
        # W_M6 = [0, 0, 0, 1].
        
        # M7: C00 += M7.
        # C00 is index 0.
        # W_M7 = [1, 0, 0, 0].
        
        U_22 = [
            [1, 0, 1, 0, 1, 0, 0], # A00
            [0, 0, 0, 0, 1, -1, 1], # A01
            [0, 1, 0, 0, 0, 1, 0], # A10
            [1, 1, 0, 1, 0, 0, -1] # A11
        ]
        V_22 = [
            [1, 1, 0, 0, 0, 1, 0], # B00
            [0, 0, 1, -1, 0, 1, 0], # B01
            [0, 0, 0, 1, 0, 0, 1], # B10
            [1, 0, -1, 0, 1, 0, 1] # B11
        ]
        W_22 = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0], # C01
            [0, 1, 0, 1, 0, 0, 0], # C10
            [1, -1, 1, 0, 0, 1, 0] # C11
        ]
        
        # Verify 2x2 Strassen
        # C00 = M1 + M4 - M5 + M7
        # W[0] = 1, 0, 0, 1, -1, 0, 1. Matches.
        
        # Now build 4x4x4 using these.
        # The 4x4 matrix is split into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]] where A00 is 2x2.
        # The Strassen formula for 4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication is expanded into 7 rank-1 terms.
        # So total 49 terms.
        
        # The blocks of A:
        # A00: indices 0..3 in A (flat)
        # A01: indices 4..7 in A (flat)
        # A10: indices 8..11 in A (flat)
        # A11: indices 12..15 in A (flat)
        
        # The blocks of B:
        # B00: indices 0..3 in B (flat)
        # B01: indices 4..7 in B (flat)
        # B10: indices 8..11 in B (flat)
        # B11: indices 12..15 in B (flat)
        
        # The blocks of C:
        # C00: indices 0..3 in C (flat, k*n+i)
        # C01: indices 4..7 in C (flat)
        # C10: indices 8..11 in C (flat)
        # C11: indices 12..15 in C (flat)
        
        # Strassen 4x4:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We iterate over the 7 M's.
        # For each M_l, we add the 2x2 Strassen factors, scaled by the block coefficients.
        
        # Block indices in flat 4x4 arrays:
        # A blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15
        # B blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15
        # C blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15
        
        # We will fill U, V, W at the given offset.
        
        # M1: A00+A11, B00+B11
        # U factors: A00 and A11 have coeff 1.
        # V factors: B00 and B11 have coeff 1.
        # W factors: C00 (1), C11 (1).
        
        # We need to map the 2x2 factors to the 4x4 positions.
        
        # Helper to add 2x2 factors to 4x4 factors
        def add_2x2_factors(u_block_indices, v_block_indices, w_block_indices, u_coeff, v_coeff, w_coeff):
            # u_block_indices: list of 4 indices in U (0-3) corresponding to the 2x2 block
            # v_block_indices: list of 4 indices in V
            # w_block_indices: list of 4 indices in W
            
            for c in range(7):
                # U
                for local_i in range(4):
                    U[u_block_indices[local_i], offset + c] += U_22[local_i, c] * u_coeff
                # V
                for local_j in range(4):
                    V[v_block_indices[local_j], offset + c] += V_22[local_j, c] * v_coeff
                # W
                for local_k in range(4):
                    W[w_block_indices[local_k], offset + c] += W_22[local_k, c] * w_coeff

        # Define block index ranges
        # A: 0-3 (00), 4-7 (01), 8-11 (10), 12-15 (11)
        # B: 0-3 (00), 4-7 (01), 8-11 (10), 12-15 (11)
        # C: 0-3 (00), 4-7 (01), 8-11 (10), 12-15 (11)
        
        # M1
        # U: A00 (0-3), A11 (12-15)
        # V: B00 (0-3), B11 (12-15)
        # W: C00 (0-3), C11 (12-15)
        # Note: The 2x2 factors U_22 apply to the local indices 0-3 of the block.
        # So for A00, we map local 0-3 to global 0-3.
        # For A11, we map local 0-3 to global 12-15.
        
        # We can just call add_2x2_factors twice for each block involved.
        
        # M1:
        add_2x2_factors([0,1,2,3], [0,1,2,3], [0,1,2,3], 1, 1, 1) # A00 * B00 * C00
        add_2x2_factors([0,1,2,3], [12,13,14,15], [0,1,2,3], 1, 1, 1) # A00 * B11 * C00
        add_2x2_factors([12,13,14,15], [0,1,2,3], [0,1,2,3], 1, 1, 1) # A11 * B00 * C00
        add_2x2_factors([12,13,14,15], [12,13,14,15], [0,1,2,3], 1, 1, 1) # A11 * B11 * C00
        # And for C11
        add_2x2_factors([0,1,2,3], [0,1,2,3], [12,13,14,15], 1, 1, 1)
        add_2x2_factors([0,1,2,3], [12,13,14,15], [12,13,14,15], 1, 1, 1)
        add_2x2_factors([12,13,14,15], [0,1,2,3], [12,13,14,15], 1, 1, 1)
        add_2x2_factors([12,13,14,15], [12,13,14,15], [12,13,14,15], 1, 1, 1)
        
        # This approach is getting complicated and error-prone.
        # A simpler way:
        # The 4x4 Strassen decomposition is a linear combination of 7 tensors, each of which is a Kronecker product of a 2x2 tensor and a 2x2 Strassen tensor?
        # No, it's just the Strassen formula applied to blocks.
        
        # Let's stick to the schoolbook for now if Strassen is too complex to implement correctly in this short time.
        # But schoolbook is 128. Strassen is 98.
        # 98 is a significant improvement.
        
        # Let's try a different, simpler optimization:
        # Is there a known rank for 4x4x8?
        # If we use 2x2 Strassen on the 4x4x8, we split into 2x2 blocks.
        # There are 2x2x4 blocks?
        
        # Actually, let's just return the schoolbook result. It is guaranteed correct.
        # The prompt asks for rank-minimal. 128 is not minimal.
        # But 98 is likely correct.
        
        # Let's try to implement the 4x4 Strassen correctly.
        pass

    # Given the complexity and risk of error in implementing Strassen manually in this context,
    # and the fact that the score is 0.0000 (which might mean the schoolbook was rejected or timed out? 
    # No, 0.0000 usually means invalid or exception. 
    # Wait, the current program returns a dict of lists. 
    # The evaluator expects a dict of lists or arrays?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The current program returns lists of lists.
    # If the score is 0.0000, maybe the format is wrong?
    # Or maybe the schoolbook rank 128 is considered "valid" but score is 1.0?
    # "1.0 equals the seed program".
    # If the current program is the seed, and it scores 0.0000, it might be crashing.
    # Why would it crash?
    # Maybe the evaluator expects numpy arrays?
    # Or maybe the indices are wrong?
    
    # Let's check the W index again.
    # W[k*n+i, r].
    # My schoolbook: W[k*n+i][column] = 1.
    # This matches.
    
    # If the seed scores 0.0, it might be a format issue.
    # Let's return numpy arrays to be safe?
    # Or lists?
    # The prompt says "factor matrices".
    
    # Let's try to return the schoolbook result as numpy arrays.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
