import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # We decompose the 2x4x6 multiplication into a Kronecker product of:
    # 1. A 2x2x2 multiplication (Strassen, rank 7)
    # 2. A 1x2x3 multiplication (rank 5)
    # Total rank = 7 * 5 = 35
    
    rank = 35
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2x2
    # Indices: U (i,j), V (j,k), W (k,i)
    # Local indices 0..3 for 2x2 blocks
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]}, # M1
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]}, # M2
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]}, # M3
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]}, # M4
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]}, # M5
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]}, # M6
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]} # M7
    ]
    
    # Rank 5 decomposition for 1x2x3
    # Inputs: x1, x2 (from 1x2)
    # Inputs: y11, y12, y13, y21, y22, y23 (from 2x3)
    # Outputs: z1, z2, z3
    # z1 = x1 y11 + x2 y21
    # z2 = x1 y12 + x2 y22
    # z3 = x1 y13 + x2 y23
    #
    # M1 = x1 y11
    # M2 = x2 y21
    # M3 = x1 y12
    # M4 = x2 y22
    # M5 = x1 y13 + x2 y23
    #
    # z1 = M1 + M2
    # z2 = M3 + M4
    # z3 = M5
    
    # The 1x2x3 tensor corresponds to the "inner" dimension of the Kronecker product.
    # We are mapping the 4x6 matrix to 2x(2x3) blocks.
    # The 4x6 matrix is partitioned into 2x2 blocks along the row dimension (size 4 -> 2x2) 
    # and 2x3 blocks along the column dimension?
    # Wait, the Kronecker structure is:
    # (2x4) * (4x6)
    # Let A be 2x4, B be 4x6.
    # Partition A into 2x2 blocks? No, A is 2x4.
    # Partition B into 2x3 blocks?
    #
    # Let's define the mapping carefully.
    # The tensor is T_{ijk} for A_{ij} B_{jk}.
    # i in 0..1, j in 0..3, k in 0..5.
    #
    # We view j as j = 2*j1 + j0, where j1 in 0..1, j0 in 0..1.
    # We view k as k = 3*k1 + k0? No, 6 = 2*3.
    # Let k = 2*k1 + k0? No, the 1x2x3 part handles the 2x3 block.
    #
    # The decomposition is:
    # A_{i, 2*j1+j0} B_{2*j1+j0, 2*k1+k0} ?
    # This doesn't align with 1x2x3.
    #
    # Let's use the standard Kronecker product of tensors.
    # T_{2,4,6} = T_{2,2,2} \otimes T_{1,2,3}.
    # Indices:
    # i = i1 (from 2x2x2)
    # j = 2*j1 + j2 (j1 from 2x2x2, j2 from 1x2x3)
    # k = 2*k1 + k2 (k1 from 2x2x2, k2 from 1x2x3)
    #
    # Wait, 1x2x3 has dimensions 1, 2, 3.
    # So the second factor contributes 1 to n, 2 to m, 3 to p.
    # Total n = 2*1 = 2.
    # Total m = 2*2 = 4.
    # Total p = 2*3 = 6.
    #
    # So:
    # i = i1 (i1 in 0..1)
    # j = 2*j1 + j2 (j1 in 0..1, j2 in 0..1) -> j in 0..3
    # k = 2*k1 + k2 (k1 in 0..1, k2 in 0..2) -> k in 0..5
    #
    # The rank is rank(T1) * rank(T2) = 7 * 5 = 35.
    #
    # We iterate over the 7 Strassen terms and 5 1x2x3 terms.
    
    col = 0
    for t1 in range(7):
        term1 = strassen_terms[t1]
        for t2 in range(5):
            # Define 1x2x3 terms
            if t2 == 0:
                # M1 = x1 y11
                # x1 is j2=0, y11 is k2=0
                u2_indices = [(0, 1)] # j2=0
                v2_indices = [(0, 1)] # k2=0
                w2_indices = [(0, 1)] # output z1 coeff 1
            elif t2 == 1:
                # M2 = x2 y21
                # x2 is j2=1, y21 is k2=0
                u2_indices = [(1, 1)] # j2=1
                v2_indices = [(0, 1)] # k2=0
                w2_indices = [(0, 1)] # output z1 coeff 1
            elif t2 == 2:
                # M3 = x1 y12
                # x1 is j2=0, y12 is k2=1
                u2_indices = [(0, 1)] # j2=0
                v2_indices = [(1, 1)] # k2=1
                w2_indices = [(1, 1)] # output z2 coeff 1
            elif t2 == 3:
                # M4 = x2 y22
                # x2 is j2=1, y22 is k2=1
                u2_indices = [(1, 1)] # j2=1
                v2_indices = [(1, 1)] # k2=1
                w2_indices = [(1, 1)] # output z2 coeff 1
            else:
                # M5 = x1 y13 + x2 y23
                # x1 is j2=0, y13 is k2=2
                # x2 is j2=1, y23 is k2=2
                u2_indices = [(0, 1), (1, 1)] # j2=0, j2=1
                v2_indices = [(2, 1), (2, 1)] # k2=2, k2=2
                w2_indices = [(2, 1)] # output z3 coeff 1
            
            # Combine indices
            # U index: i * m + j
            # i = i1 (from term1)
            # j = 2*j1 + j2
            for i1, c1 in term1['U']:
                for j2, c2 in u2_indices:
                    # j1 is fixed by term1? No, term1 U indices are (i1, j1)
                    # Wait, term1['U'] contains (local_idx, coeff)
                    # local_idx = i1 * 2 + j1
                    i1_val = i1 // 2
                    j1_val = i1 % 2
                    j_val = 2 * j1_val + j2
                    u_idx = i1_val * m + j_val
                    U[u_idx, col] += c1 * c2
            
            # V index: j * p + k
            # j = 2*j1 + j2
            # k = 2*k1 + k2
            for j2, c2 in term1['V']: # Wait, term1['V'] is (j1, k1)
                j1_val = j2 // 2
                k1_val = j2 % 2
                for k2, c2_v in v2_indices:
                    k_val = 2 * k1_val + k2
                    v_idx = j_val * p + k_val
                    # Need j_val from term1['V']
                    # term1['V'] indices are (j1, k1)
                    # So j1_val, k1_val are from term1
                    # But we need j2 from v2_indices? No, v2_indices are for the 1x2x3 part.
                    # The 1x2x3 part has dimensions 1x2x3.
                    # The 'U' part of 1x2x3 is the 1x2 input (j2).
                    # The 'V' part of 1x2x3 is the 2x3 input (k2).
                    # So j2 comes from u2_indices, k2 comes from v2_indices.
                    # But j_val = 2*j1 + j2.
                    # We need j1 from term1['V'] and j2 from u2_indices.
                    # This coupling is tricky.
                    
                    # Let's restructure the loop.
                    pass
            
            col += 1
            
            if time.time() > deadline:
                break

    # The above logic is flawed because j2 is shared between U and V of the 1x2x3 part.
    # In the Kronecker product, the index j is split into j1 (from T1) and j2 (from T2).
    # U_T2 uses j2. V_T2 uses k2.
    # So for a fixed term in T1 and T2:
    # U_T1 uses (i1, j1). U_T2 uses j2.
    # So U_idx = (i1, 2*j1 + j2).
    # V_T1 uses (j1, k1). V_T2 uses k2.
    # So V_idx = (2*j1 + j2, 2*k1 + k2).
    # W_T1 uses (k1, i1). W_T2 uses output index o2.
    # So W_idx = (2*k1 + k2, i1).
    
    # Note: j2 must be consistent between U_T2 and V_T2?
    # In the 1x2x3 tensor, the input j2 is shared between the U and V parts?
    # No, the 1x2x3 tensor is T_{o, j, k}.
    # U part corresponds to j. V part corresponds to k.
    # So j2 is an index of the 1x2x3 tensor.
    # In the Kronecker product, the j2 index is the same for the U and V components of the 1x2x3 term.
    # So for a specific 1x2x3 term, j2 is fixed.
    
    # Let's rewrite the loop correctly.
    
    rank = 35
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # 1x2x3 terms: (j2, k2, o2, coeff)
    # o2 is the output index (0, 1, 2)
    rect_terms = [
        (0, 0, 0, 1), # M1: j2=0, k2=0 -> o2=0
        (1, 0, 0, 1), # M2: j2=1, k2=0 -> o2=0
        (0, 1, 1, 1), # M3: j2=0, k2=1 -> o2=1
        (1, 1, 1, 1), # M4: j2=1, k2=1 -> o2=1
        (0, 2, 2, 1), # M5a: j2=0, k2=2 -> o2=2
        (1, 2, 2, 1)  # M5b: j2=1, k2=2 -> o2=2
    ]
    # Wait, M5 is x1 y13 + x2 y23.
    # This is two terms in the tensor sum.
    # So the rank is 6?
    # If M5 is a single product, it would be (x1+x2)(y13+y23)? No.
    # M5 = x1 y13 + x2 y23 is a sum of two products.
    # So it contributes 2 to the rank.
    # Total rank = 7 * (4 + 2) = 42.
    
    # So the rank is 42.
    # This is the same as the baseline.
    
    # Can we do better?
    # If we use rank 4 for 1x2x3, we need M5 to be 1 term.
    # M5 = (x1+x2)(y13+y23) - x1 y23 - x2 y13?
    # This increases rank.
    
    # I will stick with rank 42 but ensure the code is clean and fast.
    # The current score is 1.1429.
    # I will return the rank 42 decomposition.

    col = 0
    for t1 in range(7):
        term1 = strassen_terms[t1]
        for t2 in range(6):
            j2, k2, o2, c2 = rect_terms[t2]
            
            # U
            for i1_idx, c1_u in term1['U']:
                i1_val = i1_idx // 2
                j1_val = i1_idx % 2
                j_val = 2 * j1_val + j2
                u_idx = i1_val * m + j_val
                U[u_idx, col] += c1_u * c2
            
            # V
            for j1_idx, c1_v in term1['V']:
                j1_val = j1_idx // 2
                k1_val = j1_idx % 2
                # j_val uses j2 from rect_term
                j_val = 2 * j1_val + j2
                k_val = 2 * k1_val + k2
                v_idx = j_val * p + k_val
                V[v_idx, col] += c1_v * c2
            
            # W
            for k1_idx, c1_w in term1['W']:
                k1_val = k1_idx // 2
                i1_val = k1_idx % 2
                k_val = 2 * k1_val + k2
                w_idx = k_val * n + i1_val
                W[w_idx, col] += c1_w * c2
            
            col += 1
            
            if time.time() > deadline:
                break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
