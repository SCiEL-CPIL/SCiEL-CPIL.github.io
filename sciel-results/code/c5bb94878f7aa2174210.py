import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 Strassen + 32 for 4x4x2 Schoolbook
    rank = 81
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Part 1: Strassen 4x4x4 (first 4 columns of B) ---
    # We generate the 49 terms for multiplying 4x4 A by 4x4 B_part1
    
    # Helper to add a term
    # u_coeffs: list of (index_in_A, coeff)
    # v_coeffs: list of (index_in_B, coeff)
    # w_coeffs: list of (index_in_C, coeff)
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs:
            U[idx, col] += val
        for idx, val in v_coeffs:
            V[idx, col] += val
        for idx, val in w_coeffs:
            W[idx, col] += val

    # Strassen 2x2x2 definitions
    # Each is a list of (u_indices, v_indices, w_indices)
    # Indices are 0,1,2,3 for 2x2 matrices
    # A: 0=00, 1=01, 2=10, 3=11
    # B: 0=00, 1=01, 2=10, 3=11
    # C: 0=00, 1=01, 2=10, 3=11
    
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
    
    # We need to map 2x2 indices to 4x4 indices
    # Block 00 (top-left): rows 0,1; cols 0,1 -> indices 0,1,4,5 in flattened 4x4?
    # Flattened index for A[i,j] is i*m + j = i*4 + j
    # Block 00: i in {0,1}, j in {0,1} -> 0,1,4,5
    # Block 01 (top-right): i in {0,1}, j in {2,3} -> 2,3,6,7
    # Block 10 (bottom-left): i in {2,3}, j in {0,1} -> 8,9,12,13
    # Block 11 (bottom-right): i in {2,3}, j in {2,3} -> 10,11,14,15
    
    # Let's define the mapping from 2x2 local indices to 4x4 global indices for each block
    # Local 00 -> 0, Local 01 -> 1, Local 10 -> 4, Local 11 -> 5 (for Block 00)
    # Local 00 -> 2, Local 01 -> 3, Local 10 -> 6, Local 11 -> 7 (for Block 01)
    # Local 00 -> 8, Local 01 -> 9, Local 10 -> 12, Local 11 -> 13 (for Block 10)
    # Local 00 -> 10, Local 01 -> 11, Local 10 -> 14, Local 11 -> 15 (for Block 11)
    
    def map_block_indices(block_id, local_indices):
        # block_id: 0=00, 1=01, 2=10, 3=11
        # local_indices: list of (local_idx, coeff)
        # Returns list of (global_idx, coeff)
        # Local indices: 0=00, 1=01, 2=10, 3=11
        base = [0, 2, 8, 10][block_id]
        # Local 0 -> base + 0
        # Local 1 -> base + 1
        # Local 2 -> base + 4
        # Local 3 -> base + 5
        offsets = [0, 1, 4, 5]
        result = []
        for li, c in local_indices:
            gi = base + offsets[li]
            result.append((gi, c))
        return result

    # Define the 7 Strassen terms for 2x2x2
    # Format: (u_locales, v_locales, w_locales)
    # w_locales will be used to determine which C blocks get the contribution
    # Actually, it's easier to iterate through the 7 macro terms, and for each, iterate through the 7 micro terms.
    
    # Macro Terms:
    # 1. (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
    # 2. (A10+A11)B00 -> C10 (+1), C11 (-1)
    # 3. A00(B01-B11) -> C01 (+1), C11 (+1)
    # 4. A11(B10-B00) -> C00 (+1), C10 (+1)
    # 5. (A00+A01)B11 -> C00 (-1), C01 (+1)
    # 6. (A10-A00)(B00+B01) -> C11 (+1)
    # 7. (A01-A11)(B10+B11) -> C00 (+1)
    
    # Micro Terms (Strassen 2x2x2):
    # 1. (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # 2. (A10+A11)B00 -> C10(+1), C11(-1)
    # 3. A00(B01-B11) -> C01(+1), C11(+1)
    # 4. A11(B10-B00) -> C00(+1), C10(+1)
    # 5. (A00+A01)B11 -> C00(-1), C01(+1)
    # 6. (A10-A00)(B00+B01) -> C11(+1)
    # 7. (A01-A11)(B10+B11) -> C00(+1)
    
    # Wait, the micro terms are the same structure, but applied to the sub-matrices.
    # Let's define the micro terms' local indices for U, V, W.
    
    micro_terms = [
        # 1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        # 2
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)]
        },
        # 3
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]
        },
        # 4
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]
        },
        # 5
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)]
        },
        # 6
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        # 7
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    # Macro Terms:
    # Each macro term specifies:
    # - A blocks involved and signs
    # - B blocks involved and signs
    # - C blocks involved and signs (how the macro term contributes to them)
    
    macro_terms = [
        # 1. (A00+A11)(B00+B11)
        {
            'a_blocks': [(0, 1), (3, 1)], # Block 00 coeff 1, Block 11 coeff 1
            'b_blocks': [(0, 1), (3, 1)],
            'c_blocks': [(0, 1), (3, 1)]  # Contributes to C00 and C11
        },
        # 2. (A10+A11)B00
        {
            'a_blocks': [(2, 1), (3, 1)],
            'b_blocks': [(0, 1)],
            'c_blocks': [(2, 1), (3, -1)]
        },
        # 3. A00(B01-B11)
        {
            'a_blocks': [(0, 1)],
            'b_blocks': [(1, 1), (3, -1)],
            'c_blocks': [(1, 1), (3, 1)]
        },
        # 4. A11(B10-B00)
        {
            'a_blocks': [(3, 1)],
            'b_blocks': [(2, 1), (0, -1)],
            'c_blocks': [(0, 1), (2, 1)]
        },
        # 5. (A00+A01)B11
        {
            'a_blocks': [(0, 1), (1, 1)],
            'b_blocks': [(3, 1)],
            'c_blocks': [(0, -1), (1, 1)]
        },
        # 6. (A10-A00)(B00+B01)
        {
            'a_blocks': [(2, 1), (0, -1)],
            'b_blocks': [(0, 1), (1, 1)],
            'c_blocks': [(3, 1)]
        },
        # 7. (A01-A11)(B10+B11)
        {
            'a_blocks': [(1, 1), (3, -1)],
            'b_blocks': [(2, 1), (3, 1)],
            'c_blocks': [(0, 1)]
        }
    ]
    
    col = 0
    for macro in macro_terms:
        for micro in micro_terms:
            # Combine macro and micro
            # U: For each micro u, for each macro a_block, add term
            u_coeffs = []
            for mu_idx, mu_coeff in micro['u']:
                for ma_block, ma_coeff in macro['a_blocks']:
                    # Map micro local index to macro block local index?
                    # No, micro local index is within the 2x2 sub-matrix.
                    # The macro block is a 2x2 sub-matrix of the 4x4.
                    # So we map the micro local index to the global index using the macro block id.
                    mapped = map_block_indices(ma_block, [(mu_idx, 1)])
                    for gi, g_coeff in mapped:
                        u_coeffs.append((gi, mu_coeff * ma_coeff))
                        
            v_coeffs = []
            for mv_idx, mv_coeff in micro['v']:
                for mb_block, mb_coeff in macro['b_blocks']:
                    mapped = map_block_indices(mb_block, [(mv_idx, 1)])
                    for gi, g_coeff in mapped:
                        v_coeffs.append((gi, mv_coeff * mb_coeff))
                        
            w_coeffs = []
            for mw_idx, mw_coeff in micro['w']:
                for mc_block, mc_coeff in macro['c_blocks']:
                    mapped = map_block_indices(mc_block, [(mw_idx, 1)])
                    for gi, g_coeff in mapped:
                        w_coeffs.append((gi, mw_coeff * mc_coeff))
                        
            add_term(u_coeffs, v_coeffs, w_coeffs, col)
            col += 1
            
    # --- Part 2: Schoolbook 4x4x2 (last 2 columns of B) ---
    # n=4, m=4, p=2 (for this part)
    # Global p=6. Columns 4 and 5 of B.
    # U indices: i*4+j
    # V indices: j*6 + (4+k)
    # W indices: (4+k)*4 + i
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_idx = i * 4 + j
                v_idx = j * 6 + (4 + k)
                w_idx = (4 + k) * 4 + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
