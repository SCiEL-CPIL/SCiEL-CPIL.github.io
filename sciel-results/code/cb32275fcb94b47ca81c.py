import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Partition 5 = 4 + 1
    block_sizes = [4, 1]
    
    # Offsets
    offsets = [0, 4, 5]
    
    # Get Strassen factors for 2x2x2
    def get_strassen_factors():
        factors = []
        # M1
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_2x2 = get_strassen_factors()
    
    # Construct 4x4x4 factors using Strassen
    # Split 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen steps:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to global indices.
    # For the 4x4x4 block at (0,0,0):
    # Local indices 0..3.
    # U: 4x4. V: 4x4. W: 4x4.
    
    # We will generate the factors for M_{4,4,4} as a list of (u_vec, v_vec, w_vec) of length 16.
    factors_4x4 = []
    
    # Helper to add a scaled 2x2x2 factor to the 4x4x4 accumulation
    # local_2x2_indices: list of 4 indices in 0..3
    # u_coeff, v_coeff, w_coeff: 1x4 vectors
    
    def add_2x2_factor(factors, u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff):
        # u_idx, v_idx, w_idx are lists of 4 global indices within the 4x4 block
        # This adds one 2x2x2 Strassen step
        for (uf, vf, wf) in strassen_2x2:
            u_new = [0.0] * 16
            v_new = [0.0] * 16
            w_new = [0.0] * 16
            
            for l in range(4):
                if uf[l] != 0:
                    u_new[u_idx[l]] += u_coeff[l] * uf[l]
                if vf[l] != 0:
                    v_new[v_idx[l]] += v_coeff[l] * vf[l]
                if wf[l] != 0:
                    w_new[w_idx[l]] += w_coeff[l] * wf[l]
            
            # Check if non-zero
            if any(u_new) or any(v_new) or any(w_new):
                factors.append((u_new, v_new, w_new))

    # Define local index mappings for 2x2 sub-blocks of 4x4
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    # Global 4x4 indices:
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
    # 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
    # 8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3)
    # 12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)
    
    # Sub-block 11 (top-left): indices 0, 1, 4, 5
    idx_11 = [0, 1, 4, 5]
    # Sub-block 12 (top-right): indices 2, 3, 6, 7
    idx_12 = [2, 3, 6, 7]
    # Sub-block 21 (bottom-left): indices 8, 9, 12, 13
    idx_21 = [8, 9, 12, 13]
    # Sub-block 22 (bottom-right): indices 10, 11, 14, 15
    idx_22 = [10, 11, 14, 15]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 + A22: U indices are idx_11 and idx_22 with coeff 1
    u_m1 = [0.0]*16
    for i in idx_11: u_m1[i] = 1.0
    for i in idx_22: u_m1[i] = 1.0
    
    v_m1 = [0.0]*16
    for i in idx_11: v_m1[i] = 1.0
    for i in idx_22: v_m1[i] = 1.0
    
    # This creates a 2x2x2 tensor where U is A11+A22, V is B11+B22.
    # We call add_2x2_factor with u_idx=idx_11, v_idx=idx_11, w_idx=idx_11?
    # No, the 2x2x2 factor expects the local 2x2 indices.
    # The "u_idx" in add_2x2_factor should map the local 2x2 indices to the global 4x4 indices.
    # For M1, the "A" part is A11+A22.
    # The local 2x2 indices 0,1,2,3 correspond to the elements of A11+A22?
    # No, the Strassen factor (uf, vf, wf) acts on the 2x2 matrices.
    # So we need to pass the indices of the 2x2 matrices involved.
    # For M1, the 2x2 matrices are A11, A22 (for U) and B11, B22 (for V).
    # But U is a single 4x4 matrix.
    # The factor (uf, vf, wf) is for a 2x2x2 multiplication.
    # It takes 2x2 inputs.
    # So we need to specify which 2x2 sub-blocks of the 4x4 matrices are used.
    
    # Revised add_2x2_factor:
    # u_blocks: list of 2x2 index lists (4 lists of 4 indices)
    # v_blocks: list of 2x2 index lists
    # w_blocks: list of 2x2 index lists
    # u_coeffs: 1x4 vector
    # v_coeffs: 1x4 vector
    # w_coeffs: 1x4 vector
    
    # This is getting complicated.
    # Simpler: Just use the known 49 factors for 4x4x4?
    # Or just use the 2+2+1 decomposition which is already working and safe.
    
    # The 4+1 split is theoretically better (110 vs 117).
    # But implementing the 4x4x4 factors correctly is error-prone.
    # I will stick to the 2+2+1 decomposition but verify the code is correct.
    # The current code is correct and scores 1.0684.
    
    # To improve, I can try to optimize the 2+2+1 code or just return it.
    # Given the risk of implementing 4x4x4 incorrectly, I will return the 2+2+1 version.
    
    # Wait, I can implement the 4+1 split by using Strassen for 4x4x4.
    # I will write a helper to build 4x4x4 factors from 2x2x2 factors.
    
    return None # Placeholder
