import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are constructing a rank-49 decomposition using recursive Strassen.
    # The tensor is T[i, j, k] where C[i, k] = sum_j A[i, j] B[j, k].
    # Indices:
    # A: i (0..3), j (0..3) -> flat i*4 + j
    # B: j (0..3), k (0..3) -> flat j*4 + k
    # C: k (0..3), i (0..3) -> flat k*4 + i (Note: W index is k*n + i)
    
    # Define Strassen linear forms for 2x2 matrices
    # Inputs X (2x2), Y (2x2)
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Output Z (2x2)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Level 1: 4x4 multiplication as 2x2 block multiplication
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    #
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B01 - B11
    # T4 = B10 - B00
    # T5 = B11
    # T6 = B00 + B01
    # T7 = B10 + B11
    
    # C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to map these to scalar indices.
    # A flat index: i*4 + j
    # B flat index: j*4 + k
    # C flat index (W): k*4 + i
    
    # Helper to get flat index for A[i, j]
    def idx_A(i, j): return i * 4 + j
    # Helper to get flat index for B[j, k]
    def idx_B(j, k): return j * 4 + k
    # Helper to get flat index for C[i, k] (stored as W[k, i])
    def idx_C(i, k): return k * 4 + i

    # Define the 7 block-level linear combinations for A and B
    # Each is a 2x2 matrix of linear forms (represented as dicts of {flat_idx: coeff})
    
    def add_dict(d1, d2):
        d = {}
        for k, v in d1.items(): d[k] = v
        for k, v in d2.items(): d[k] = d.get(k, 0) + v
        return d

    def sub_dict(d1, d2):
        d = {}
        for k, v in d1.items(): d[k] = v
        for k, v in d2.items(): d[k] = d.get(k, 0) - v
        return d

    def scale_dict(d, s):
        return {k: v * s for k, v in d.items()}

    # A entries
    A00 = {idx_A(0,0):1, idx_A(0,1):1, idx_A(1,0):1, idx_A(1,1):1} # Wait, A00 is the top-left 2x2 block
    # A00 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    # Let's be precise.
    # Block A00 corresponds to i in 0,1 and j in 0,1.
    # Block A01 corresponds to i in 0,1 and j in 2,3.
    # Block A10 corresponds to i in 2,3 and j in 0,1.
    # Block A11 corresponds to i in 2,3 and j in 2,3.
    
    A00 = {idx_A(0,0):1, idx_A(0,1):1, idx_A(1,0):1, idx_A(1,1):1}
    A01 = {idx_A(0,2):1, idx_A(0,3):1, idx_A(1,2):1, idx_A(1,3):1}
    A10 = {idx_A(2,0):1, idx_A(2,1):1, idx_A(3,0):1, idx_A(3,1):1}
    A11 = {idx_A(2,2):1, idx_A(2,3):1, idx_A(3,2):1, idx_A(3,3):1}
    
    B00 = {idx_B(0,0):1, idx_B(0,1):1, idx_B(1,0):1, idx_B(1,1):1}
    B01 = {idx_B(0,2):1, idx_B(0,3):1, idx_B(1,2):1, idx_B(1,3):1}
    B10 = {idx_B(2,0):1, idx_B(2,1):1, idx_B(3,0):1, idx_B(3,1):1}
    B11 = {idx_B(2,2):1, idx_B(2,3):1, idx_B(3,2):1, idx_B(3,3):1}

    # Block-level S and T matrices (2x2 matrices of linear forms)
    # S[k] is a 2x2 matrix, S[k][r][c] is a dict
    S = []
    T = []
    
    # S1 = A00 + A11
    S1 = [ [add_dict(A00, A11), add_dict(A00, A11)], # This is wrong. S1 is a matrix.
            # S1 = A00 + A11 means the whole block A00 plus the whole block A11.
            # But A00 and A11 are 2x2 blocks.
            # So S1 is a 2x2 matrix where each entry is a linear form.
            # S1[0,0] = A[0,0] + A[2,2]
            # S1[0,1] = A[0,1] + A[2,3]
            # S1[1,0] = A[1,0] + A[3,2]
            # S1[1,1] = A[1,1] + A[3,3]
            ]
            
    # Let's construct S1 properly
    S1 = [
        [ {idx_A(0,0):1, idx_A(2,2):1}, {idx_A(0,1):1, idx_A(2,3):1} ],
        [ {idx_A(1,0):1, idx_A(3,2):1}, {idx_A(1,1):1, idx_A(3,3):1} ]
    ]
    
    S2 = [ # A10 + A11
        [ {idx_A(2,0):1, idx_A(2,2):1}, {idx_A(2,1):1, idx_A(2,3):1} ],
        [ {idx_A(3,0):1, idx_A(3,2):1}, {idx_A(3,1):1, idx_A(3,3):1} ]
    ]
    
    S3 = [ # A00
        [ {idx_A(0,0):1}, {idx_A(0,1):1} ],
        [ {idx_A(1,0):1}, {idx_A(1,1):1} ]
    ]
    
    S4 = [ # A11
        [ {idx_A(2,2):1}, {idx_A(2,3):1} ],
        [ {idx_A(3,2):1}, {idx_A(3,3):1} ]
    ]
    
    S5 = [ # A00 + A01
        [ {idx_A(0,0):1, idx_A(0,2):1}, {idx_A(0,1):1, idx_A(0,3):1} ],
        [ {idx_A(1,0):1, idx_A(1,2):1}, {idx_A(1,1):1, idx_A(1,3):1} ]
    ]
    
    S6 = [ # A10 - A00
        [ {idx_A(2,0):1, idx_A(0,0):-1}, {idx_A(2,1):1, idx_A(0,1):-1} ],
        [ {idx_A(3,0):1, idx_A(1,0):-1}, {idx_A(3,1):1, idx_A(1,1):-1} ]
    ]
    
    S7 = [ # A01 - A11
        [ {idx_A(0,2):1, idx_A(2,2):-1}, {idx_A(0,3):1, idx_A(2,3):-1} ],
        [ {idx_A(1,2):1, idx_A(3,2):-1}, {idx_A(1,3):1, idx_A(3,3):-1} ]
    ]
    
    S = [S1, S2, S3, S4, S5, S6, S7]
    
    T1 = [ # B00 + B11
        [ {idx_B(0,0):1, idx_B(2,2):1}, {idx_B(0,1):1, idx_B(2,3):1} ],
        [ {idx_B(1,0):1, idx_B(3,2):1}, {idx_B(1,1):1, idx_B(3,3):1} ]
    ]
    
    T2 = [ # B00
        [ {idx_B(0,0):1}, {idx_B(0,1):1} ],
        [ {idx_B(1,0):1}, {idx_B(1,1):1} ]
    ]
    
    T3 = [ # B01 - B11
        [ {idx_B(0,2):1, idx_B(2,2):-1}, {idx_B(0,3):1, idx_B(2,3):-1} ],
        [ {idx_B(1,2):1, idx_B(3,2):-1}, {idx_B(1,3):1, idx_B(3,3):-1} ]
    ]
    
    T4 = [ # B10 - B00
        [ {idx_B(2,0):1, idx_B(0,0):-1}, {idx_B(2,1):1, idx_B(0,1):-1} ],
        [ {idx_B(3,0):1, idx_B(1,0):-1}, {idx_B(3,1):1, idx_B(1,1):-1} ]
    ]
    
    T5 = [ # B11
        [ {idx_B(2,2):1}, {idx_B(2,3):1} ],
        [ {idx_B(3,2):1}, {idx_B(3,3):1} ]
    ]
    
    T6 = [ # B00 + B01
        [ {idx_B(0,0):1, idx_B(0,2):1}, {idx_B(0,1):1, idx_B(0,3):1} ],
        [ {idx_B(1,0):1, idx_B(1,2):1}, {idx_B(1,1):1, idx_B(1,3):1} ]
    ]
    
    T7 = [ # B10 + B11
        [ {idx_B(2,0):1, idx_B(2,2):1}, {idx_B(2,1):1, idx_B(2,3):1} ],
        [ {idx_B(3,0):1, idx_B(3,2):1}, {idx_B(3,1):1, idx_B(3,3):1} ]
    ]
    
    T = [T1, T2, T3, T4, T5, T6, T7]

    # Now, for each k in 0..6, decompose S[k] * T[k] using Strassen (7 terms)
    # And accumulate the contributions to C.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for k in range(7):
        Sk = S[k]
        Tk = T[k]
        
        # Strassen terms for 2x2 matrices Sk, Tk
        # M1 = (Sk00 + Sk11)(Tk00 + Tk11)
        # ...
        
        # We need the linear forms for A and B for each of the 7 terms.
        # Term 1:
        # U1 = Sk00 + Sk11
        # V1 = Tk00 + Tk11
        # U1 is a 2x2 matrix of linear forms? No, Sk00 is a linear form (scalar).
        # Wait, Sk is a 2x2 matrix of linear forms.
        # Sk[0][0] is a dict. Sk[1][1] is a dict.
        # U1_dict = add_dict(Sk[0][0], Sk[1][1])
        # V1_dict = add_dict(Tk[0][0], Tk[1][1])
        
        # The product M1 = U1 * V1 is a scalar (linear form in A * linear form in B).
        # This corresponds to one rank-1 term in the tensor decomposition.
        # The coefficients for U are in U1_dict.
        # The coefficients for V are in V1_dict.
        
        # Now, where does M1 go in the final C?
        # M1 contributes to Mk[0][0], Mk[1][1], etc.
        # Mk is the result of Sk * Tk.
        # Mk[0][0] = M1 + M4 - M5 + M7
        # Mk[0][1] = M3 + M5
        # Mk[1][0] = M2 + M4
        # Mk[1][1] = M1 - M2 + M3 + M6
        
        # And Mk contributes to C blocks.
        # C00 = M1 + M4 - M5 + M7 (Block level)
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Note: The block-level Strassen uses the same structure.
        # Let's denote the block-level products as P1..P7.
        # P_k = S_k T_k.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So for a specific scalar term m_{k,l} (which is part of P_k):
        # It contributes to C[i,j] with coefficient:
        # coeff_C_block[k][block_idx] * coeff_Pk[k][l][local_idx]
        
        # Let's precompute the coefficients for P_k entries in terms of m_{k,l}
        # P_k[0][0] = m1 + m4 - m5 + m7
        # P_k[0][1] = m3 + m5
        # P_k[1][0] = m2 + m4
        # P_k[1][1] = m1 - m2 + m3 + m6
        
        coeffs_P = [
            [ [1, 0, 0, 1, -1, 0, 1],  # m1..m7 coeffs for P00
              [0, 0, 1, 0, 1, 0, 0] ], # for P01
            [ [0, 1, 0, 1, 0, 0, 0],   # for P10
              [1, -1, 1, 0, 0, 1, 0] ] # for P11
        ]
        
        # Coeffs for C blocks in terms of P_k
        # C00 = P1 + P4 - P5 + P7 -> coeff for P1 is 1, P4 is 1, P5 is -1, P7 is 1
        # C01 = P3 + P5 -> coeff for P3 is 1, P5 is 1
        # C10 = P2 + P4 -> coeff for P2 is 1, P4 is 1
        # C11 = P1 - P2 + P3 + P6 -> coeff for P1 is 1, P2 is -1, P3 is 1, P6 is 1
        
        # C_block[0][0] (C00) depends on P1, P4, P5, P7
        # C_block[0][1] (C01) depends on P3, P5
        # C_block[1][0] (C10) depends on P2, P4
        # C_block[1][1] (C11) depends on P1, P2, P3, P6
        
        # We need to map C_block[r][c] to C[i][j].
        # C_block[0][0] -> i in 0,1, j in 0,1
        # C_block[0][1] -> i in 0,1, j in 2,3
        # C_block[1][0] -> i in 2,3, j in 0,1
        # C_block[1][1] -> i in 2,3, j in 2,3
        
        # For a given k and l (term index 0..6):
        # Get U_dict and V_dict for m_{k,l}.
        # Determine the coefficient for each C[i,j].
        
        # First, get the linear forms for m_{k,l}
        # l=0 (m1): U = Sk00+Sk11, V = Tk00+Tk11
        # l=1 (m2): U = Sk10+Sk11, V = Tk00
        # l=2 (m3): U = Sk00, V = Tk01-Tk11
        # l=3 (m4): U = Sk11, V = Tk10-Tk00
        # l=4 (m5): U = Sk00+Sk01, V = Tk11
        # l=5 (m6): U = Sk10-Sk00, V = Tk00+Tk01
        # l=6 (m7): U = Sk01-Sk11, V = Tk10+Tk11
        
        U_forms = []
        V_forms = []
        
        U_forms.append(add_dict(Sk[0][0], Sk[1][1]))
        V_forms.append(add_dict(Tk[0][0], Tk[1][1]))
        
        U_forms.append(add_dict(Sk[1][0], Sk[1][1]))
        V_forms.append(Tk[0][0])
        
        U_forms.append(Sk[0][0])
        V_forms.append(sub_dict(Tk[0][1], Tk[1][1]))
        
        U_forms.append(Sk[1][1])
        V_forms.append(sub_dict(Tk[1][0], Tk[0][0]))
        
        U_forms.append(add_dict(Sk[0][0], Sk[0][1]))
        V_forms.append(Tk[1][1])
        
        U_forms.append(sub_dict(Sk[1][0], Sk[0][0]))
        V_forms.append(add_dict(Tk[0][0], Tk[0][1]))
        
        U_forms.append(sub_dict(Sk[0][1], Sk[1][1]))
        V_forms.append(add_dict(Tk[1][0], Tk[1][1]))
        
        for l in range(7):
            u_dict = U_forms[l]
            v_dict = V_forms[l]
            
            # Fill U and V
            for idx, coeff in u_dict.items():
                U[idx][col] = coeff
            for idx, coeff in v_dict.items():
                V[idx][col] = coeff
            
            # Fill W
            # Determine contribution of m_{k,l} to C[i,j]
            # m_{k,l} contributes to P_k[0][0], P_k[0][1], P_k[1][0], P_k[1][1]
            # coeffs_P[k] is not needed, coeffs_P is the same for all k.
            
            p00_coeff = coeffs_P[0][0][l]
            p01_coeff = coeffs_P[0][1][l]
            p10_coeff = coeffs_P[1][0][l]
            p11_coeff = coeffs_P[1][1][l]
            
            # Now map P_k contributions to C blocks
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # k is 0-indexed. P_k is index k.
            # We need the coefficient of P_k in each C block.
            
            c00_coeff = 0
            c01_coeff = 0
            c10_coeff = 0
            c11_coeff = 0
            
            if k == 0: # P1
                c00_coeff = 1
                c11_coeff = 1
            elif k == 1: # P2
                c10_coeff = 1
                c11_coeff = -1
            elif k == 2: # P3
                c01_coeff = 1
                c11_coeff = 1
            elif k == 3: # P4
                c00_coeff = 1
                c10_coeff = 1
            elif k == 4: # P5
                c00_coeff = -1
                c01_coeff = 1
            elif k == 5: # P6
                c11_coeff = 1
            elif k == 6: # P7
                c00_coeff = 1
            
            # Total coefficient for C[i,j]
            # C[i,j] is in block (br, bc) where br = i//2, bc = j//2
            # Local index in block is (lr, lc) = (i%2, j%2)
            
            for i in range(4):
                for j in range(4):
                    br = i // 2
                    bc = j // 2
                    lr = i % 2
                    lc = j % 2
                    
                    block_coeff = 0
                    if br == 0 and bc == 0:
                        block_coeff = c00_coeff
                    elif br == 0 and bc == 1:
                        block_coeff = c01_coeff
                    elif br == 1 and bc == 0:
                        block_coeff = c10_coeff
                    elif br == 1 and bc == 1:
                        block_coeff = c11_coeff
                    
                    if block_coeff == 0:
                        continue
                    
                    # P_k[br][bc] is the block entry?
                    # No, P_k is a 2x2 matrix of 2x2 blocks?
                    # No, P_k = S_k T_k is a 2x2 matrix of scalars (if S_k, T_k were 1x1).
                    # Here S_k, T_k are 2x2 matrices of linear forms.
                    # So P_k is a 2x2 matrix of linear forms (products).
                    # P_k[0][0] is a linear form (scalar).
                    # This linear form corresponds to the top-left entry of the block product.
                    # The block product P_k is a 2x2 matrix.
                    # The entry P_k[br][bc] contributes to the C block at (br, bc) at local position (lr, lc)?
                    # No.
                    # S_k is a 2x2 matrix. T_k is a 2x2 matrix.
                    # P_k = S_k T_k is a 2x2 matrix.
                    # The entry P_k[0][0] is a scalar (linear form).
                    # This scalar is the (0,0) entry of the matrix P_k.
                    # The matrix P_k corresponds to the block of C?
                    # No.
                    # S_k and T_k are formed from A and B blocks.
                    # P_k = S_k T_k is a 2x2 matrix.
                    # This 2x2 matrix P_k is one of the 7 terms in the Strassen combination for C.
                    # C00 = P1 + ...
                    # So P1 is a 2x2 matrix.
                    # C00 is a 2x2 block.
                    # So C00[0][0] = P1[0][0] + ...
                    # C00[0][1] = P1[0][1] + ...
                    
                    # So, for a specific local position (lr, lc) in the block (br, bc):
                    # The value is sum over k of (coeff of P_k in C_block[br][bc]) * P_k[lr][lc]
                    
                    # We have coeffs_P[br][lc][l] which is the coeff of m_{k,l} in P_k[br][lc].
                    # Wait, coeffs_P was defined for P_k[0][0] etc.
                    # coeffs_P[0][0] is coeffs for P[0][0].
                    # So coeff of m_{k,l} in P_k[lr][lc] is coeffs_P[lr][lc][l].
                    
                    local_p_coeff = coeffs_P[lr][lc][l]
                    
                    total_coeff = block_coeff * local_p_coeff
                    
                    if total_coeff != 0:
                        W[idx_C(i, j)][col] += total_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
