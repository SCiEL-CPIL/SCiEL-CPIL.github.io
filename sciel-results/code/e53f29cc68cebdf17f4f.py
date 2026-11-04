import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4 * 4x4 using Strassen (Rank 49)
    # Part 2: 4x4 * 4x3 using Schoolbook (Rank 48)
    
    # Helper to create a zero vector of given length
    def zero_vec(length):
        return [0] * length
    
    # Strassen 2x2x2 terms
    # u, v, w are lists of 4 elements
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # C01 gets M3
        # M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]), # C00 gets M4, C10 gets M4
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C00 gets -M5, C01 gets M5
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11 gets M6
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # C00 gets M7
    ]
    
    # Verify Strassen 2x2x2
    # C00 = M1 + M4 - M5 + M7
    # M1: W[0]=1. M4: W[0]=1. M5: W[0]=-1. M7: W[0]=1. Sum = 1+1-1+1 = 2? 
    # Wait, the W vector in the tuple above represents the coefficient of the scalar product for each C element.
    # Let's re-verify the W vectors.
    # C00 = M1 + M4 - M5 + M7
    # Term 1 (M1): W[0] should be 1.
    # Term 4 (M4): W[0] should be 1.
    # Term 5 (M5): W[0] should be -1.
    # Term 7 (M7): W[0] should be 1.
    # In my list:
    # T1: W=[1,0,0,1] -> C00 gets 1. Correct.
    # T4: W=[1,0,1,0] -> C00 gets 1. Correct.
    # T5: W=[-1,1,0,0] -> C00 gets -1. Correct.
    # T7: W=[1,0,0,0] -> C00 gets 1. Correct.
    # Sum for C00: 1+1-1+1 = 2? No, the sum of the products U*V*W must be 1.
    # The term is U*V*W.
    # For C00, we sum over all terms.
    # Term 1: U1*V1*W1[0] = (A00+A11)(B00+B11)*1
    # Term 4: U4*V4*W4[0] = A11(B01-B00)*1
    # Term 5: U5*V5*W5[0] = (A00+A01)B11*(-1)
    # Term 7: U7*V7*W7[0] = (A01-A11)(B10+B11)*1
    # This matches the Strassen formula.
    
    # Block coefficients for L_k and R_k
    # k: 0..6
    # b: 0..3 (block index in 4x4)
    L_coeffs = [
        [1, 0, 0, 1], # L0
        [0, 0, 1, 1], # L1
        [1, 0, 0, 0], # L2
        [0, 0, 0, 1], # L3
        [1, 1, 0, 0], # L4
        [-1, 0, 1, 0],# L5
        [0, 1, 0, -1] # L6
    ]
    R_coeffs = [
        [1, 0, 0, 1], # R0
        [1, 0, 0, 0], # R1
        [0, 0, 1, -1],# R2
        [-1, 1, 0, 0],# R3
        [0, 0, 0, 1], # R4
        [1, 1, 0, 0], # R5
        [0, 0, 1, 1]  # R6
    ]
    
    # Coefficients of P_k in C_b
    # C0 = P0 + P3 - P4 + P6
    # C1 = P2 + P4
    # C2 = P1 + P3
    # C3 = P0 - P1 + P2 + P5
    C_coeffs = [
        # k=0
        [1, 0, 0, 1],
        # k=1
        [0, 0, 0, -1],
        # k=2
        [0, 1, 0, 1],
        # k=3
        [1, 0, 1, 0],
        # k=4
        [-1, 1, 0, 0],
        # k=5
        [0, 0, 0, 1],
        # k=6
        [1, 0, 0, 0]
    ]
    
    terms = []
    
    # Generate 49 terms for 4x4x4
    for k in range(7):
        for s in range(7):
            u_2x2, v_2x2, w_2x2 = strassen_2x2[s]
            
            U_global = zero_vec(16)
            V_global = zero_vec(16)
            W_global = zero_vec(16)
            
            for b in range(4):
                # Expand U
                coeff_u = L_coeffs[k][b]
                if coeff_u != 0:
                    # Block b is at offset b*4 in the 16-length vector?
                    # A is 4x4. Flattened row-major:
                    # Block 0 (0,0): indices 0,1,4,5
                    # Block 1 (0,1): indices 2,3,6,7
                    # Block 2 (1,0): indices 8,9,12,13
                    # Block 3 (1,1): indices 10,11,14,15
                    # u_2x2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # Global index for block b, local index l:
                    # b=0: base=0. l=0->0, l=1->1, l=2->4, l=3->5
                    # b=1: base=2. l=0->2, l=1->3, l=2->6, l=3->7
                    # b=2: base=8. l=0->8, l=1->9, l=2->12, l=3->13
                    # b=3: base=10. l=0->10, l=1->11, l=2->14, l=3->15
                    
                    base = [0, 2, 8, 10][b]
                    offsets = [0, 1, 4, 5]
                    for l in range(4):
                        idx = base + offsets[l]
                        U_global[idx] += coeff_u * u_2x2[l]
                        
                # Expand V
                coeff_v = R_coeffs[k][b]
                if coeff_v != 0:
                    base = [0, 2, 8, 10][b]
                    offsets = [0, 1, 4, 5]
                    for l in range(4):
                        idx = base + offsets[l]
                        V_global[idx] += coeff_v * v_2x2[l]
                        
                # Expand W
                coeff_w = C_coeffs[k][b]
                if coeff_w != 0:
                    base = [0, 2, 8, 10][b]
                    offsets = [0, 1, 4, 5]
                    for l in range(4):
                        idx = base + offsets[l]
                        W_global[idx] += coeff_w * w_2x2[l]
                        
            terms.append((U_global, V_global, W_global))
            
    # Generate 48 terms for 4x4x3 (Schoolbook)
    # A is 4x4 (indices 0..15)
    # B is 4x3 (indices 0..11). B[j, k] -> index j*3 + k
    # C is 4x3 (indices 0..11). C[i, k] -> index i*3 + k
    # Wait, the interface requires V to be (m*p, rank) and W to be (p*n, rank).
    # Here we are doing 4x4 * 4x3.
    # U shape: (4*4, rank) = (16, rank)
    # V shape: (4*3, rank) = (12, rank)
    # W shape: (3*4, rank) = (12, rank)
    
    # But the final output W is (p*n, rank) = (7*4, rank) = (28, rank).
    # The first part (4x4x4) contributes to C[:, 0:4].
    # The second part (4x4x3) contributes to C[:, 4:7].
    
    # So for the second part, the W vector should be placed in the correct columns of the 28-length W vector.
    # C is 4x7. Flattened row-major:
    # C[i, k] -> index i*7 + k.
    # For k in 0..3, index is i*7 + k.
    # For k in 4..6, index is i*7 + k.
    
    # In the first part, I constructed W_global of length 16.
    # I need to expand this to length 28.
    
    # Let's re-do the storage.
    final_U = []
    final_V = []
    final_W = []
    
    # Process Part 1 (4x4x4)
    for u, v, w in terms:
        # u is length 16 (4x4)
        # v is length 16 (4x4) -> needs to be mapped to 4x7 (28)
        # w is length 16 (4x4) -> needs to be mapped to 7x4 (28)
        
        V_ext = zero_vec(28)
        # v corresponds to B (4x4). B is the first 4 columns of the full B (4x7).
        # B[j, k] for k in 0..3.
        # V index: j*7 + k.
        # v index: j*4 + k.
        for j in range(4):
            for k in range(4):
                V_ext[j*7 + k] = v[j*4 + k]
                
        W_ext = zero_vec(28)
        # w corresponds to C (4x4). C is the first 4 columns of the full C (4x7).
        # W index: k*4 + i (column-major for W? No, W[k*n+i]).
        # Task says: W[k*n+i, r].
        # My W_global was constructed with indices i*4 + k?
        # Let's check my W_global construction.
        # I used base + offsets.
        # offsets = [0, 1, 4, 5].
        # For block 0 (i=0,1; k=0,1):
        # l=0 (0,0) -> idx 0.
        # l=1 (0,1) -> idx 1.
        # l=2 (1,0) -> idx 4.
        # l=3 (1,1) -> idx 5.
        # This is row-major for C (4x4). C[i, k] -> i*4 + k.
        # But the task requires W[k*n + i].
        # So I need to transpose the W part.
        
        for i in range(4):
            for k in range(4):
                W_ext[k*4 + i] = w[i*4 + k]
                
        final_U.append(u)
        final_V.append(V_ext)
        final_W.append(W_ext)
        
    # Process Part 2 (4x4x3)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = zero_vec(16)
                u[i*4 + j] = 1
                
                v = zero_vec(28)
                # B is 4x7. We are using columns 4,5,6.
                # B[j, 4+k].
                # V index: j*7 + (4+k).
                v[j*7 + (4+k)] = 1
                
                w = zero_vec(28)
                # C is 4x7.
                # C[i, 4+k].
                # W index: (4+k)*4 + i.
                w[(4+k)*4 + i] = 1
                
                final_U.append(u)
                final_V.append(v)
                final_W.append(w)
                
    rank = len(final_U)
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    for c in range(rank):
        for r in range(16):
            U[r][c] = final_U[c][r]
        for r in range(28):
            V[r][c] = final_V[c][r]
        for r in range(28):
            W[r][c] = final_W[c][r]
            
    return {"U": U, "V": V, "W": W}

