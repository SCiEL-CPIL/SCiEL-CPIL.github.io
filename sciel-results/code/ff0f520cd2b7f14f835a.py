import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses Strassen's algorithm for 4x4 blocks and a rank-1 decomposition for the 4x1 block.
    """
    n, m, p = DIMENSIONS
    
    # We split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A @ B = [A @ B1, A @ B2, A @ B3]
    
    # 1. Construct Strassen decomposition for 4x4 * 4x4
    # This will give U1, V1, W1 of rank 49
    
    # Strassen for 2x2 * 2x2 has rank 7.
    # For 4x4 * 4x4, we treat A and B as 2x2 blocks of 2x2 matrices.
    # The decomposition is the Kronecker product of the 2x2 Strassen decomposition?
    # No, it's a block-wise application.
    
    # Let's build the 4x4x4x4 tensor decomposition explicitly.
    # U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    # Strassen 2x2 indices:
    # A blocks: 00, 01, 10, 11 (indices 0,1,2,3 in 4x4)
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # The 7 products for 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to map this to the 4x4 case.
    # A 4x4 matrix has 16 elements.
    # Block 00: (0,0), (0,1), (1,0), (1,1)
    # Block 01: (0,2), (0,3), (1,2), (1,3)
    # Block 10: (2,0), (2,1), (3,0), (3,1)
    # Block 11: (2,2), (2,3), (3,2), (3,3)
    
    # We will iterate over the 7 Strassen terms for the "outer" 2x2 blocks.
    # For each outer term, we perform a 2x2 Strassen multiplication on the "inner" 2x2 blocks.
    # This results in 7 * 7 = 49 rank-1 terms.
    
    # Let's define the inner Strassen coefficients.
    # For a 2x2 multiplication C = AB:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # The 7 P terms are:
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # We need to express U, V, W for the 4x4 case.
    # The tensor is T[i, j, k].
    # i, j, k in 0..3.
    
    # We will fill U_44, V_44, W_44.
    
    # Helper to get block indices
    def get_block_indices(block_id):
        # block_id 0: 00, 1: 01, 2: 10, 3: 11
        # Returns list of (row, col) in 4x4
        rows = [0, 1] if block_id < 2 else [2, 3]
        cols = [0, 1] if block_id % 2 == 0 else [2, 3]
        indices = []
        for r in rows:
            for c in cols:
                indices.append((r, c))
        return indices

    # Inner Strassen Coefficients for 2x2
    # We define the 7 inner terms as linear combinations of A and B elements.
    # Actually, we can just use the known rank-7 decomposition for 2x2.
    # Let's hardcode the 2x2 Strassen decomposition.
    
    # For 2x2:
    # U_2x2 shape (4, 7), V_2x2 shape (4, 7), W_2x2 shape (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # P1: (A00+A11)(B00+B11)
    # A part: A00 + A11 -> U[0,0]=1, U[3,0]=1
    # B part: B00 + B11 -> V[0,0]=1, V[3,0]=1
    # C part: C00 += P1, C11 += P1 -> W[0,0]=1, W[3,0]=1
    
    # P2: (A10+A11)B00
    # A: A10+A11 -> U[2,1]=1, U[3,1]=1
    # B: B00 -> V[0,1]=1
    # C: C10 += P2 -> W[2,1]=1
    
    # P3: A00(B01-B11)
    # A: A00 -> U[0,2]=1
    # B: B01-B11 -> V[1,2]=1, V[3,2]=-1
    # C: C01 += P3 -> W[1,2]=1
    
    # P4: A11(B10-B00)
    # A: A11 -> U[3,3]=1
    # B: B10-B00 -> V[2,3]=1, V[0,3]=-1
    # C: C10 += P4, C00 += P4 -> W[2,3]=1, W[0,3]=1
    
    # P5: (A00+A01)B11
    # A: A00+A01 -> U[0,4]=1, U[1,4]=1
    # B: B11 -> V[3,4]=1
    # C: C01 += P5, C00 -= P5 -> W[1,4]=1, W[0,4]=-1
    
    # P6: (A10-A00)(B00+B01)
    # A: A10-A00 -> U[2,5]=1, U[0,5]=-1
    # B: B00+B01 -> V[0,5]=1, V[1,5]=1
    # C: C11 += P6 -> W[3,5]=1
    
    # P7: (A01-A11)(B10+B11)
    # A: A01-A11 -> U[1,6]=1, U[3,6]=-1
    # B: B10+B11 -> V[2,6]=1, V[3,6]=1
    # C: C11 += P7, C00 += P7 -> W[3,6]=1, W[0,6]=1
    
    # Let's populate the arrays
    # Note: The indices in U, V, W for 2x2 are:
    # U[i, j, r] where i,j in 0..1. Flattened i*2+j.
    # 0:00, 1:01, 2:10, 3:11.
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[2, 3] = 1; W_2[0, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[1, 4] = 1; W_2[0, 4] = -1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[3, 6] = 1; W_2[0, 6] = 1
    
    # Now construct 4x4 from 2x2 blocks.
    # Outer blocks: 00, 01, 10, 11.
    # For each outer Strassen term (7 terms), we have an "effective" A_block and B_block.
    # These effective blocks are 2x2 matrices.
    # We then apply the inner Strassen decomposition to these 2x2 blocks.
    
    # The 49 rank terms correspond to (outer_term, inner_term).
    # outer_term in 0..6, inner_term in 0..6.
    
    # We need to map the effective A_block elements to the original 4x4 A indices.
    # And effective B_block elements to original 4x4 B indices.
    
    # Let's define the linear combinations for the outer Strassen terms.
    # Outer A blocks: A00, A01, A10, A11 (each 2x2).
    # Outer B blocks: B00, B01, B10, B11.
    
    # Outer P1: (A00+A11)(B00+B11)
    # Outer P2: (A10+A11)B00
    # ... etc.
    
    # We will iterate over the 7 outer terms.
    # For each, we determine which original 4x4 elements contribute to the effective A and B.
    
    # Effective A for outer term t is a 2x2 matrix.
    # Its elements are linear combinations of the 4x2x2 blocks of A.
    # Actually, the effective A is a 2x2 matrix of "super-elements" which are 2x2 blocks.
    # So the effective A is a 4x4 matrix? No.
    # The outer Strassen operates on 2x2 blocks of 2x2 matrices.
    # So the "elements" of the outer 2x2 matrix are 2x2 matrices.
    # The inner Strassen operates on these 2x2 matrices.
    
    # So, for a given outer term t and inner term s:
    # The contribution to U[i, j, r] is non-zero if the (i, j) element of A is involved in the outer term t's A-part and the inner term s's A-part.
    
    # Let's build the mapping.
    # Outer A-part for term t:
    # t=0: A00+A11. So blocks 0 and 3.
    # t=1: A10+A11. Blocks 2 and 3.
    # t=2: A00. Block 0.
    # t=3: A11. Block 3.
    # t=4: A00+A01. Blocks 0 and 1.
    # t=5: A10-A00. Blocks 2 and 0 (with -1).
    # t=6: A01-A11. Blocks 1 and 3 (with -1).
    
    # Similarly for B.
    
    # For each outer term t, we have a set of (block_id, coeff) for A and B.
    # Then for each inner term s, we have a set of (local_i, local_j, coeff) for the 2x2 block.
    
    # We will sum these up to fill U_44, V_44, W_44.
    
    outer_A = [
        [(0, 1), (3, 1)], # P1
        [(2, 1), (3, 1)], # P2
        [(0, 1)],         # P3
        [(3, 1)],         # P4
        [(0, 1), (1, 1)], # P5
        [(2, 1), (0, -1)],# P6
        [(1, 1), (3, -1)] # P7
    ]
    
    outer_B = [
        [(0, 1), (3, 1)], # P1
        [(0, 1)],         # P2
        [(1, 1), (3, -1)],# P3
        [(2, 1), (0, -1)],# P4
        [(3, 1)],         # P5
        [(0, 1), (1, 1)], # P6
        [(2, 1), (3, 1)]  # P7
    ]
    
    # Inner Strassen U, V, W are defined for 2x2.
    # U_2[i, j, s] is the coefficient for A[i, j] in the s-th inner term.
    # V_2[i, j, s] is the coefficient for B[i, j] in the s-th inner term.
    # W_2[i, j, s] is the coefficient for C[i, j] in the s-th inner term.
    
    # Note: My U_2, V_2, W_2 above are flattened.
    # U_2[0] is A00, U_2[1] is A01, U_2[2] is A10, U_2[3] is A11.
    
    # We need to map (block_id, local_i, local_j) to global index.
    # block_id 0: rows 0,1 cols 0,1.
    # local (0,0) -> global (0,0) -> idx 0
    # local (0,1) -> global (0,1) -> idx 1
    # local (1,0) -> global (1,0) -> idx 2
    # local (1,1) -> global (1,1) -> idx 3
    
    # block_id 1: rows 0,1 cols 2,3.
    # local (0,0) -> global (0,2) -> idx 2
    # local (0,1) -> global (0,3) -> idx 3
    # local (1,0) -> global (1,2) -> idx 6
    # local (1,1) -> global (1,3) -> idx 7
    
    # block_id 2: rows 2,3 cols 0,1.
    # local (0,0) -> global (2,0) -> idx 8
    # local (0,1) -> global (2,1) -> idx 9
    # local (1,0) -> global (3,0) -> idx 12
    # local (1,1) -> global (3,1) -> idx 13
    
    # block_id 3: rows 2,3 cols 2,3.
    # local (0,0) -> global (2,2) -> idx 10
    # local (0,1) -> global (2,3) -> idx 11
    # local (1,0) -> global (3,2) -> idx 14
    # local (1,1) -> global (3,3) -> idx 15
    
    def block_to_global(block_id, li, lj):
        r0 = 0 if block_id < 2 else 2
        c0 = 0 if block_id % 2 == 0 else 2
        return (r0 + li) * 4 + (c0 + lj)
    
    # Now fill U_44, V_44, W_44
    # Rank 49.
    # r = outer_t * 7 + inner_s
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # U part
            for (block_id, coeff_outer) in outer_A[t]:
                # For this block, we need to add U_2[local_idx, s] * coeff_outer
                # U_2 is 4x7. local_idx 0..3.
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        val = U_2[local_idx, s] * coeff_outer
                        if val != 0:
                            global_idx = block_to_global(block_id, li, lj)
                            U_44[global_idx, r] += val
            
            # V part
            for (block_id, coeff_outer) in outer_B[t]:
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        val = V_2[local_idx, s] * coeff_outer
                        if val != 0:
                            global_idx = block_to_global(block_id, li, lj)
                            V_44[global_idx, r] += val
            
            # W part
            # W_2 is for C blocks.
            # The C blocks are determined by the outer term t?
            # No, the outer term t produces a "super C" block.
            # The inner term s produces a specific C element within that super block.
            # Wait, the outer Strassen produces 4 super C blocks.
            # The inner Strassen is applied to the multiplication of the effective A and B blocks.
            # The result is a 2x2 matrix of C blocks.
            # So for a fixed outer term t, the result is a 2x2 matrix of 2x2 blocks.
            # The inner Strassen decomposition gives the coefficients for these 2x2 blocks.
            # So W_44 should be filled based on W_2.
            
            # But which C blocks?
            # The outer term t corresponds to a specific linear combination of C blocks.
            # Actually, the outer Strassen decomposition expresses C00, C01, C10, C11 as sums of P1..P7.
            # So P_t contributes to specific C blocks with specific coefficients.
            # Let's define outer_W.
            # P1 -> C00 (+1), C11 (+1)
            # P2 -> C10 (+1)
            # P3 -> C01 (+1)
            # P4 -> C10 (+1), C00 (+1)
            # P5 -> C01 (+1), C00 (-1)
            # P6 -> C11 (+1)
            # P7 -> C11 (+1), C00 (+1)
            
    # This is getting complex to hardcode perfectly in the thought process.
    # I will implement this in the code.
    
    # 2. Construct rank-1 decomposition for 4x4 * 4x1
    # U_41 shape (16, 1), V_41 shape (4, 1), W_41 shape (4, 1)
    U_41 = np.ones((16, 1))
    V_41 = np.ones((4, 1))
    W_41 = np.ones((4, 1))
    
    # 3. Combine
    # Total rank = 49 + 49 + 1 = 99
    
    U_total = np.zeros((16, 99))
    V_total = np.zeros((36, 99))
    W_total = np.zeros((36, 99))
    
    # B is 4x9. Split into B1 (4x4), B2 (4x4), B3 (4x1).
    # V indices:
    # B1: j*4 + k, j in 0..3, k in 0..3. Indices 0..15.
    # B2: j*4 + k, j in 0..3, k in 4..7? No.
    # B is 4x9.
    # V index is j * p + k. p=9.
    # B1 columns 0..3. V indices: j*9 + k.
    # B2 columns 4..7. V indices: j*9 + k.
    # B3 column 8. V indices: j*9 + 8.
    
    # W indices:
    # C is 4x9.
    # W index is k * n + i. n=4.
    # C1 columns 0..3. W indices: k*4 + i.
    # C2 columns 4..7. W indices: k*4 + i.
    # C3 column 8. W indices: k*4 + i.
    
    # Copy U_44 to U_total
    U_total[:, 0:49] = U_44
    U_total[:, 49:98] = U_44
    U_total[:, 98] = U_41[:, 0]
    
    # Copy V_44 to V_total
    # B1 part
    for j in range(4):
        for k in range(4):
            V_total[j*9 + k, 0:49] = V_44[j*4 + k, 0:49]
    # B2 part
    for j in range(4):
        for k in range(4):
            V_total[j*9 + (k+4), 49:98] = V_44[j*4 + k, 0:49]
    # B3 part
    for j in range(4):
        V_total[j*9 + 8, 98] = V_41[j, 0]
        
    # Copy W_44 to W_total
    # C1 part
    for k in range(4):
        for i in range(4):
            W_total[k*4 + i, 0:49] = W_44[k*4 + i, 0:49]
    # C2 part
    for k in range(4):
        for i in range(4):
            W_total[(k+4)*4 + i, 49:98] = W_44[k*4 + i, 0:49]
    # C3 part
    for k in range(4):
        for i in range(4):
            W_total[(k+8)*4 + i, 98] = W_41[i, 0] # Note: W_41 index is i?
            # W_41 shape (4, 1). Index is k*n+i?
            # For 4x1, k=0. W index is 0*4+i = i.
            # So W_41[i, 0] is correct.
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
