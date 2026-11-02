import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # We will build the decomposition by summing smaller decompositions.
    # The tensor is T = sum_{i,k,j} A_{ik} B_{kj} C_{ij}
    # We split k into k1 (0,1) and k2 (2,3).
    # T = T1 + T2
    # T1 involves A_{i, k1}, B_{k1, j}
    # T2 involves A_{i, k2}, B_{k2, j}
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    # block_A_indices: list of (i, k) pairs for the 2x2 block of A
    # block_B_indices: list of (k, j) pairs for the 2x2 or 2x1 block of B
    # type: 'strassen' or 'schoolbook'
    
    def add_strassen_block(a_idx, b_idx, c_idx):
        # a_idx: [(i,k), (i,k), (i,k), (i,k)] corresponding to a11, a12, a21, a22
        # b_idx: [(k,j), (k,j), (k,j), (k,j)] corresponding to b11, b12, b21, b22
        # c_idx: [(i,j), (i,j), (i,j), (i,j)] corresponding to c11, c12, c21, c22
        
        # Strassen formulas
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We add 7 rank-1 terms
        
        # M1
        U_list.append([1, 1, 0, 0]) # a11, a12, a21, a22 -> 1, 0, 0, 1
        V_list.append([1, 0, 0, 1]) # b11, b12, b21, b22 -> 1, 0, 0, 1
        W_list.append([1, 0, 0, 0]) # c11, c12, c21, c22 -> 1, 0, 0, 0
        
        # M4
        U_list.append([0, 0, 0, 1])
        V_list.append([0, 0, 1, 0]) # b21 - b11 -> 0, 0, 1, -1? No, V is for B.
        # Wait, V index corresponds to B entries.
        # b11 is index 0, b12 is 1, b21 is 2, b22 is 3.
        # M4 = a22 * (b21 - b11).
        # U: a22 is 1 at pos 3. [0,0,0,1]
        # V: b21 is 1 at pos 2, b11 is -1 at pos 0. [-1, 0, 1, 0]
        # W: c21 is 1 at pos 2. [0,0,1,0]
        
        # Let's redo carefully.
        
        # M1: (a11+a22)(b11+b22) -> C11, C22
        # U: [1, 0, 0, 1]
        # V: [1, 0, 0, 1]
        # W: [1, 0, 0, 1] (Contributes to C11 and C22)
        
        # M2: (a21+a22)b11 -> C21
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0]
        
        # M3: a11(b12-b22) -> C12
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 0]
        
        # M4: a22(b21-b11) -> C11, C21
        # U: [0, 0, 0, 1]
        # V: [-1, 0, 1, 0]
        # W: [1, 0, 1, 0]
        
        # M5: (a11+a12)b22 -> C11, C12
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [1, 1, 0, 0]
        
        # M6: (a21-a11)(b11+b12) -> C22
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 0, 1]
        
        # M7: (a12-a22)(b21+b22) -> C11, C22
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [1, 0, 0, 1]
        
        # Wait, the W vectors must sum to the identity for the block.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's check W contributions:
        # C11: M1(1) + M4(1) - M5(1) + M7(1) -> 1+1-1+1 = 2? No, coefficients in W.
        # The W vector for each term indicates the coefficient in C.
        
        # Term 1 (M1): W=[1, 0, 0, 1]
        # Term 2 (M2): W=[0, 0, 1, 0]
        # Term 3 (M3): W=[0, 1, 0, 0]
        # Term 4 (M4): W=[1, 0, 1, 0]
        # Term 5 (M5): W=[-1, -1, 0, 0]  <-- Note negative sign
        # Term 6 (M6): W=[0, 0, 0, 1]
        # Term 7 (M7): W=[1, 0, 0, 1]
        
        # Sum for C11: 1 + 0 + 0 + 1 - 1 + 0 + 1 = 2? No.
        # Standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My W vectors:
        # 1: [1, 0, 0, 1]
        # 2: [0, 0, 1, 0]
        # 3: [0, 1, 0, 0]
        # 4: [1, 0, 1, 0]
        # 5: [-1, -1, 0, 0]  (Because C11 has -M5, C12 has +M5)
        # 6: [0, 0, 0, 1]
        # 7: [1, 0, 0, 1]
        
        # C11: 1 + 0 + 0 + 1 - 1 + 0 + 1 = 2. Should be 1.
        # Ah, M1 contributes to C11 and C22.
        # M4 contributes to C11 and C21.
        # M5 contributes to C11 and C12.
        # M7 contributes to C11 and C22.
        
        # Let's re-verify Strassen formulas.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My W vectors for each term:
        # T1 (M1): C11=1, C22=1 -> [1, 0, 0, 1]
        # T2 (M2): C21=1 -> [0, 0, 1, 0]
        # T3 (M3): C12=1 -> [0, 1, 0, 0]
        # T4 (M4): C11=1, C21=1 -> [1, 0, 1, 0]
        # T5 (M5): C11=-1, C12=1 -> [-1, 1, 0, 0]
        # T6 (M6): C22=1 -> [0, 0, 0, 1]
        # T7 (M7): C11=1, C22=1 -> [1, 0, 0, 1]
        
        # Sum C11: 1 + 0 + 0 + 1 - 1 + 0 + 1 = 2. Still 2.
        # Why?
        # C11 = M1 + M4 - M5 + M7.
        # Coeff of M1 in C11 is 1.
        # Coeff of M4 in C11 is 1.
        # Coeff of M5 in C11 is -1.
        # Coeff of M7 in C11 is 1.
        # Sum = 1 + 1 - 1 + 1 = 2.
        # Is the formula wrong?
        # Standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Let's check C11 for A=I, B=I.
        # M1 = 2*2=4
        # M4 = 1*(1-1)=0
        # M5 = 2*1=2
        # M7 = 1*(1+1)=2
        # C11 = 4 + 0 - 2 + 2 = 4. But C11 should be 1.
        # So the formula is scaled?
        # Actually, for I*I, C=I.
        # M1=4, M4=0, M5=2, M7=2.
        # 4 - 2 + 2 = 4.
        # It seems the standard Strassen gives 2*C?
        # No, Strassen is exact.
        # Let's re-read Strassen.
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # If A=I, B=I:
        # a11=1, a12=0, a21=0, a22=1
        # b11=1, b12=0, b21=0, b22=1
        # M1 = 2*2=4
        # M2 = 1*1=1
        # M3 = 1*(0-1)=-1
        # M4 = 1*(0-1)=-1
        # M5 = 1*1=1
        # M6 = (-1)*(1+0)=-1
        # M7 = (-1)*(0+1)=-1
        # C11 = 4 + (-1) - 1 + (-1) = 1. Correct.
        # C12 = -1 + 1 = 0. Correct.
        # C21 = 1 + (-1) = 0. Correct.
        # C22 = 4 - 1 + (-1) + (-1) = 1. Correct.
        
        # So the W vectors are correct.
        
        # Add these 7 terms to the global lists.
        # We need to map the local indices (0,1,2,3) to global indices.
        
        # U local: [a11, a12, a21, a22]
        # V local: [b11, b12, b21, b22]
        # W local: [c11, c12, c21, c22]
        
        u_vecs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        
        v_vecs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [-1, 0, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        w_vecs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 0], # M2
            [0, 1, 0, 0], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 1]  # M7
        ]
        
        for r in range(7):
            u_g = [0.0] * (n*m)
            v_g = [0.0] * (m*p)
            w_g = [0.0] * (p*n)
            
            for local_idx, val in enumerate(u_vecs[r]):
                if val != 0:
                    global_idx = a_idx[local_idx]
                    u_g[global_idx] += val
            
            for local_idx, val in enumerate(v_vecs[r]):
                if val != 0:
                    global_idx = b_idx[local_idx]
                    v_g[global_idx] += val
            
            for local_idx, val in enumerate(w_vecs[r]):
                if val != 0:
                    global_idx = c_idx[local_idx]
                    w_g[global_idx] += val
            
            U_list.append(u_g)
            V_list.append(v_g)
            W_list.append(w_g)

    def add_schoolbook_block_2x2x1(a_idx, b_idx, c_idx):
        # a_idx: [a11, a12, a21, a22] (only 2 used? No, 2x2 A, 2x1 B)
        # B is 2x1: b1, b2.
        # C is 2x1: c1, c2.
        # c1 = a11 b1 + a12 b2
        # c2 = a21 b1 + a22 b2
        # Rank 4? Or 2?
        # c1 = a11 b1 + a12 b2
        # c2 = a21 b1 + a22 b2
        # We can do:
        # M1 = a11 b1
        # M2 = a12 b2
        # M3 = a21 b1
        # M4 = a22 b2
        # C1 = M1 + M2
        # C2 = M3 + M4
        # Rank 4.
        
        # But we can share b1, b2?
        # No, A is different.
        
        # Actually, for 2x2x1, rank is 2?
        # C = A b.
        # This is linear in b.
        # Bilinear rank of 2x2x1 is 2?
        # T = a11 b1 c1 + a12 b2 c1 + a21 b1 c2 + a22 b2 c2
        # = b1 (a11 c1 + a21 c2) + b2 (a12 c1 + a22 c2)
        # This is not rank 1.
        # It is rank 4 in general?
        # No, 2x2x1 tensor is rank 2?
        # Let's check.
        # If we fix b, C is linear in A.
        # The tensor is sum of 4 terms.
        # Can we do it in 2?
        # M1 = (a11+a21) b1
        # M2 = (a12+a22) b2
        # C1 = M1 + M2? No.
        
        # Actually, 2x2x1 rank is 4.
        
        # So for the last block (2x2x1), we use rank 4.
        
        u_vecs = [
            [1, 0, 0, 0], # a11
            [0, 1, 0, 0], # a12
            [0, 0, 1, 0], # a21
            [0, 0, 0, 1]  # a22
        ]
        
        # B is 2x1. Indices: b1, b2.
        # We need to map b_idx[0] to b1, b_idx[1] to b2.
        # But a_idx has 4 entries.
        # We assume a_idx is [a11, a12, a21, a22]
        # b_idx is [b1, b2]
        # c_idx is [c1, c2]
        
        # M1: a11 b1 -> c1
        # M2: a12 b2 -> c1
        # M3: a21 b1 -> c2
        # M4: a22 b2 -> c2
        
        v_vecs = [
            [1, 0], # b1
            [0, 1], # b2
            [1, 0], # b1
            [0, 1]  # b2
        ]
        
        w_vecs = [
            [1, 0], # c1
            [1, 0], # c1
            [0, 1], # c2
            [0, 1]  # c2
        ]
        
        for r in range(4):
            u_g = [0.0] * (n*m)
            v_g = [0.0] * (m*p)
            w_g = [0.0] * (p*n)
            
            for local_idx, val in enumerate(u_vecs[r]):
                if val != 0:
                    global_idx = a_idx[local_idx]
                    u_g[global_idx] += val
            
            for local_idx, val in enumerate(v_vecs[r]):
                if val != 0:
                    global_idx = b_idx[local_idx]
                    v_g[global_idx] += val
            
            for local_idx, val in enumerate(w_vecs[r]):
                if val != 0:
                    global_idx = c_idx[local_idx]
                    w_g[global_idx] += val
            
            U_list.append(u_g)
            V_list.append(v_g)
            W_list.append(w_g)

    # Global indices
    # A: 2x4. i in 0,1. k in 0,1,2,3.
    # A[i, k] index = i*4 + k
    # B: 4x7. k in 0,1,2,3. j in 0..6.
    # B[k, j] index = k*7 + j
    # C: 2x7. i in 0,1. j in 0..6.
    # C[i, j] index = i*7 + j
    
    # Split k into L (0,1) and R (2,3)
    
    # Block L: k in {0,1}
    # A_L indices:
    # a11: A[0,0] = 0
    # a12: A[0,1] = 1
    # a21: A[1,0] = 4
    # a22: A[1,1] = 5
    a_L = [0, 1, 4, 5]
    
    # B_L is 2x7. k in {0,1}.
    # We split j into 4 groups: {0,1}, {2,3}, {4,5}, {6}
    
    # Group 1: j in {0,1}
    b_L_1 = [
        0*7+0, # b11 (k=0, j=0)
        0*7+1, # b12 (k=0, j=1)
        1*7+0, # b21 (k=1, j=0)
        1*7+1  # b22 (k=1, j=1)
    ]
    c_1 = [
        0*7+0, # c11 (i=0, j=0)
        0*7+1, # c12 (i=0, j=1)
        1*7+0, # c21 (i=1, j=0)
        1*7+1  # c22 (i=1, j=1)
    ]
    add_strassen_block(a_L, b_L_1, c_1)
    
    # Group 2: j in {2,3}
    b_L_2 = [
        0*7+2,
        0*7+3,
        1*7+2,
        1*7+3
    ]
    c_2 = [
        0*7+2,
        0*7+3,
        1*7+2,
        1*7+3
    ]
    add_strassen_block(a_L, b_L_2, c_2)
    
    # Group 3: j in {4,5}
    b_L_3 = [
        0*7+4,
        0*7+5,
        1*7+4,
        1*7+5
    ]
    c_3 = [
        0*7+4,
        0*7+5,
        1*7+4,
        1*7+5
    ]
    add_strassen_block(a_L, b_L_3, c_3)
    
    # Group 4: j in {6}
    # B is 2x1.
    # b1: k=0, j=6 -> 6
    # b2: k=1, j=6 -> 13
    b_L_4 = [6, 13]
    c_4 = [
        0*7+6, # c1 (i=0, j=6)
        1*7+6  # c2 (i=1, j=6)
    ]
    add_schoolbook_block_2x2x1(a_L, b_L_4, c_4)
    
    # Block R: k in {2,3}
    # A_R indices:
    # a11: A[0,2] = 2
    # a12: A[0,3] = 3
    # a21: A[1,2] = 6
    # a22: A[1,3] = 7
    a_R = [2, 3, 6, 7]
    
    # Group 1: j in {0,1}
    b_R_1 = [
        2*7+0,
        2*7+1,
        3*7+0,
        3*7+1
    ]
    c_1_R = [
        0*7+0,
        0*7+1,
        1*7+0,
        1*7+1
    ]
    add_strassen_block(a_R, b_R_1, c_1_R)
    
    # Group 2: j in {2,3}
    b_R_2 = [
        2*7+2,
        2*7+3,
        3*7+2,
        3*7+3
    ]
    c_2_R = [
        0*7+2,
        0*7+3,
        1*7+2,
        1*7+3
    ]
    add_strassen_block(a_R, b_R_2, c_2_R)
    
    # Group 3: j in {4,5}
    b_R_3 = [
        2*7+4,
        2*7+5,
        3*7+4,
        3*7+5
    ]
    c_3_R = [
        0*7+4,
        0*7+5,
        1*7+4,
        1*7+5
    ]
    add_strassen_block(a_R, b_R_3, c_3_R)
    
    # Group 4: j in {6}
    b_R_4 = [
        2*7+6,
        3*7+6
    ]
    c_4_R = [
        0*7+6,
        1*7+6
    ]
    add_schoolbook_block_2x2x1(a_R, b_R_4, c_4_R)
    
    # Convert to arrays
    U = np.array(U_list).T # Shape (n*m, rank)
    V = np.array(V_list).T # Shape (m*p, rank)
    W = np.array(W_list).T # Shape (p*n, rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
