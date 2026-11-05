import numpy as np

def get_strassen_2x2_terms():
    # Returns list of (u, v, w) for 2x2x2 multiplication
    # u, v, w are lists of 4 elements (flattened 2x2)
    # w index: k*2 + i
    terms = []
    # M1
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1] # C00, C11
    terms.append((u, v, w))
    # M2
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 1, 0, -1] # C10, C11
    terms.append((u, v, w))
    # M3
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 0, 1, 1] # C01, C11
    terms.append((u, v, w))
    # M4
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 1, 0, 0] # C00, C10
    terms.append((u, v, w))
    # M5
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 0, 1, 0] # C00, C01
    terms.append((u, v, w))
    # M6
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1] # C11
    terms.append((u, v, w))
    # M7
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, -1, 0] # C00, C01
    terms.append((u, v, w))
    return terms

def get_strassen_4x4_terms():
    base_terms = get_strassen_2x2_terms()
    # For each of the 7 base terms, we have a "block" multiplication.
    # We need to expand each block multiplication into 7 scalar multiplications.
    # The block multiplication is of the form (sum u_i A_i) (sum v_j B_j) = sum u_i v_j A_i B_j
    # But A_i and B_j are 2x2 matrices.
    # So we use the 7-term decomposition for the 2x2 multiplication of A_i and B_j.
    
    all_terms = []
    for u_block, v_block, w_block in base_terms:
        # u_block, v_block are lists of 4 coefficients for the 4 2x2 blocks.
        # w_block is the contribution to the 4 2x2 output blocks.
        # We iterate over all pairs of non-zero coefficients in u_block and v_block.
        for i in range(4):
            if u_block[i] == 0: continue
            for j in range(4):
                if v_block[j] == 0: continue
                # We have a term u_block[i] * v_block[j] * (A_i * B_j)
                # A_i * B_j is a 2x2 multiplication.
                # We decompose A_i * B_j using the 7 base terms.
                for u_base, v_base, w_base in base_terms:
                    # u_base, v_base, w_base are vectors of length 4.
                    # We need to map these to the global 4x4 indices.
                    # A_i is at block i. If i=0 (top-left), A elements are at 0,1,2,3.
                    # If i=1 (top-right), A elements are at 4,5,6,7.
                    # A_i block index i corresponds to global A indices:
                    # i=0: 0..3, i=1: 4..7, i=2: 8..11, i=3: 12..15.
                    # Wait, A is 4x4. Blocks are 2x2.
                    # Block 0: rows 0-1, cols 0-1. Indices 0,1,2,3.
                    # Block 1: rows 0-1, cols 2-3. Indices 4,5,6,7.
                    # Block 2: rows 2-3, cols 0-1. Indices 8,9,10,11.
                    # Block 3: rows 2-3, cols 2-3. Indices 12,13,14,15.
                    
                    u_global = [0]*16
                    v_global = [0]*16
                    w_global = [0]*16
                    
                    # Fill u_global
                    for k in range(4):
                        if u_base[k] != 0:
                            # A_i block i, element k.
                            # Global index = i*4 + k.
                            u_global[i*4 + k] = u_block[i] * v_block[j] * u_base[k]
                    
                    # Fill v_global
                    # B_j block j.
                    # B is 4x4. Blocks are 2x2.
                    # Block 0: rows 0-1, cols 0-1. Indices 0,1,2,3.
                    # Block 1: rows 0-1, cols 2-3. Indices 4,5,6,7.
                    # Block 2: rows 2-3, cols 0-1. Indices 8,9,10,11.
                    # Block 3: rows 2-3, cols 2-3. Indices 12,13,14,15.
                    for k in range(4):
                        if v_base[k] != 0:
                            v_global[j*4 + k] = u_block[i] * v_block[j] * v_base[k]
                            
                    # Fill w_global
                    # w_block tells which output block gets this contribution.
                    # w_block has 4 elements, corresponding to output blocks 0,1,2,3.
                    # The contribution from A_i * B_j goes to ALL output blocks?
                    # No, the Strassen step M_S produces a single 2x2 matrix.
                    # This matrix is added to specific output blocks with coefficients from w_block.
                    # So for each output block l where w_block[l] != 0, we add the contribution.
                    for l in range(4):
                        if w_block[l] != 0:
                            # The 2x2 result A_i * B_j is added to output block l.
                            # w_base tells how the 4 elements of the 2x2 result map to C.
                            # Output block l has global C indices:
                            # l=0: 0..3, l=1: 4..7, l=2: 8..11, l=3: 12..15.
                            # Wait, C is 4x4. w index is k*4 + i.
                            # Block 0 (top-left): k=0,1; i=0,1. Indices 0,1,2,3.
                            # Block 1 (top-right): k=2,3; i=0,1. Indices 8,9,10,11.
                            # Block 2 (bottom-left): k=0,1; i=2,3. Indices 4,5,6,7.
                            # Block 3 (bottom-right): k=2,3; i=2,3. Indices 12,13,14,15.
                            
                            # Map w_base (length 4) to global indices in block l.
                            # w_base[k] corresponds to C element in block l.
                            # If l=0, w_base[0]->C00 (idx 0), w_base[1]->C10 (idx 1), w_base[2]->C01 (idx 2), w_base[3]->C11 (idx 3).
                            # If l=1, w_base[0]->C02 (idx 8), w_base[1]->C12 (idx 9), w_base[2]->C03 (idx 10), w_base[3]->C13 (idx 11).
                            # If l=2, w_base[0]->C20 (idx 4), w_base[1]->C30 (idx 5), w_base[2]->C21 (idx 6), w_base[3]->C31 (idx 7).
                            # If l=3, w_base[0]->C22 (idx 12), w_base[1]->C32 (idx 13), w_base[2]->C23 (idx 14), w_base[3]->C33 (idx 15).
                            
                            base_idx = [0, 8, 4, 12][l]
                            for k in range(4):
                                if w_base[k] != 0:
                                    w_global[base_idx + k] = u_block[i] * v_block[j] * w_block[l] * w_base[k]
                                    
                    all_terms.append((u_global, v_global, w_global))
    return all_terms
