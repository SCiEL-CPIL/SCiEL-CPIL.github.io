"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank = 49 + 16 + 20 = 85
    rank = 85
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition
    # u, v, w are vectors of length 4 (indices 00, 01, 10, 11)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Macro definitions for 4x4x4
    # (A_blocks, B_blocks, C_block, C_coeff)
    # A_blocks: list of (block_index, coeff)
    # B_blocks: list of (block_index, coeff)
    # C_block: index of the 2x2 block in C (0:00, 1:01, 2:10, 3:11)
    # C_coeff: scalar multiplier for the whole block result? 
    # Actually, the Strassen micro-terms already have coefficients in W.
    # We just need to map the micro-term W to the correct C block.
    
    macros = [
        # M1: (A00+A11)(B00+B11) -> C00
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], 0),
        # M2: (A10+A11)B00 -> C10, C11
        ([(2, 1), (3, 1)], [(0, 1)], 2), # Note: M2 contributes to C10 and C11 with different coeffs in micro-W
        # Wait, the micro-W vector has coefficients for the 4 entries of the 2x2 result.
        # M2 result is a 2x2 matrix.
        # C10 is the bottom-left block of C.
        # C11 is the bottom-right block of C.
        # The micro-W vector [0, 0, 1, -1] means:
        # C_res[1,0] += 1 * w2
        # C_res[1,1] += -1 * w2
        # So we add to C10 at pos (1,0) and C11 at pos (1,1).
        
        # Let's refine the macro definition to include the target C block and the specific mapping.
        # It's easier to just iterate and add.
    ]
    
    # Re-doing macro logic properly
    # For each macro m in 0..6:
    #   For each micro t in 0..6:
    #     u_vec = strassen_terms[t][0]
    #     v_vec = strassen_terms[t][1]
    #     w_vec = strassen_terms[t][2]
    #     
    #     Determine A blocks and coeffs for this macro.
    #     Determine B blocks and coeffs for this macro.
    #     Determine C block and coeffs for this macro.
    
    macro_defs = [
        # (A_blocks, B_blocks, C_block, C_coeffs)
        # C_coeffs is a 4-vector indicating how the 2x2 result of the macro maps to the C block.
        # Actually, the micro-W already gives the coefficients for the 2x2 result.
        # We just need to know WHICH C block the macro result goes into.
        # And if the macro result is summed into multiple C blocks?
        # No, each macro M_k produces a 2x2 matrix.
        # In Strassen, M_k is added to specific C blocks with specific coefficients.
        # The coefficients are embedded in the micro-W?
        # No. The micro-W gives the coefficients for the entries of the product matrix Z.
        # The macro definition says: Z is added to C_block with coefficient 1?
        # Let's check M2 again.
        # M2 = (A10+A11)B00.
        # Z = M2.
        # C10 += M2. (So C10[0,0] += Z[0,0], etc.)
        # C11 -= M2. (So C11[0,0] -= Z[0,0], etc.)
        # So the mapping is:
        # C_block 2 (C10) gets Z with coeff 1.
        # C_block 3 (C11) gets Z with coeff -1.
        
        # So macro_def should be: (A_blocks, B_blocks, [(C_block, C_coeff), ...])
        
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]), # M1 -> C00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2 -> C10, C11
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]), # M3 -> C01, C11
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]), # M4 -> C00, C10
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5 -> C00, C01
        ([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6 -> C11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])  # M7 -> C00
    ]
    
    col = 0
    for macro in macro_defs:
        a_blocks = macro[0]
        b_blocks = macro[1]
        c_mappings = macro[2]
        
        for t in range(7):
            u_vec, v_vec, w_vec = strassen_terms[t]
            
            # Construct U, V, W for this term
            for idx in range(4):
                if u_vec[idx] != 0:
                    coeff = u_vec[idx]
                    for block_idx, b_coeff in a_blocks:
                        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
                        # Global A index:
                        # A00: i in 0-1, j in 0-1
                        # A01: i in 0-1, j in 2-3
                        # A10: i in 2-3, j in 0-1
                        # A11: i in 2-3, j in 2-3
                        # idx: 0=00, 1=01, 2=10, 3=11 within the 2x2 block
                        i_local = idx // 2
                        j_local = idx % 2
                        
                        i_global = (block_idx // 2) * 2 + i_local
                        j_global = (block_idx % 2) * 2 + j_local
                        
                        U[i_global * m + j_global, col] += coeff * b_coeff
                        
                if v_vec[idx] != 0:
                    coeff = v_vec[idx]
                    for block_idx, b_coeff in b_blocks:
                        # block_idx: 0=B00, 1=B01, 2=B10, 3=B11
                        # Global B index:
                        # B00: i in 0-1, j in 0-1
                        # B01: i in 0-1, j in 2-3
                        # B10: i in 2-3, j in 0-1
                        # B11: i in 2-3, j in 2-3
                        i_local = idx // 2
                        j_local = idx % 2
                        
                        i_global = (block_idx // 2) * 2 + i_local
                        j_global = (block_idx % 2) * 2 + j_local
                        
                        V[i_global * p + j_global, col] += coeff * b_coeff
            
            for idx in range(4):
                if w_vec[idx] != 0:
                    coeff = w_vec[idx]
                    for c_block, c_coeff in c_mappings:
                        # c_block: 0=C00, 1=C01, 2=C10, 3=C11
                        # Global C index:
                        # C00: i in 0-1, j in 0-1
                        # C01: i in 0-1, j in 2-3
                        # C10: i in 2-3, j in 0-1
                        # C11: i in 2-3, j in 2-3
                        # idx: 0=00, 1=01, 2=10, 3=11 within the 2x2 block
                        i_local = idx // 2
                        j_local = idx % 2
                        
                        i_global = (c_block // 2) * 2 + i_local
                        j_global = (c_block % 2) * 2 + j_local
                        
                        # W index: k * n + i ? 
                        # Task says W[k*n+i, r].
                        # C is p x n. Here p=4, n=4 for the subproblem.
                        # Global C is 5x4.
                        # The subproblem C is 4x4.
                        # i_global is row (0..3), j_global is col (0..3).
                        # W index = j_global * n + i_global.
                        W[j_global * n + i_global, col] += coeff * c_coeff
                        
            col += 1
            
    # Add 4x4x1 part (A[0:4, 0:4] * B[0:4, 4])
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k=4
    # C indices: i in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    # Add 4x1x5 part (A[0:4, 4] * B[4, 0:5])
    # A indices: i in 0..3, j=4
    # B indices: j=4, k in 0..4
    # C indices: i in 0..3, k in 0..4
    for i in range(4):
        for k in range(5):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
