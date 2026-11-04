import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # --- Part 1: 4x4x4 Rank 49 via Strassen ---
    
    # Base 2x2x2 Strassen Rank 7
    # U, V, W are 4x7 matrices
    # Indices: i*2+j
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (a+d)(e+h) -> C00 +=, C11 +=
    # a=0, d=2, e=0, h=2
    U2[0, 0] = 1; U2[2, 0] = 1
    V2[0, 0] = 1; V2[2, 0] = 1
    W2[0, 0] = 1; W2[2, 0] = 1 # This W2 is for the 2x2 output of the block product.
    
    # M2: (c+d)e -> C10 +=
    # c=1, d=2, e=0
    U2[1, 1] = 1; U2[2, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = 1 # Wait, W2 indices for 2x2 output: 0,1,2,3.
    # C10 is bottom-left. In 2x2, C10 is index 2?
    # 2x2 Matrix:
    # 0 1
    # 2 3
    # C10 is row 1, col 0 -> index 2.
    # C11 is row 1, col 1 -> index 3.
    # C00 is 0, C01 is 1.
    # M2 contributes to C10?
    # C10 = M2 + M4.
    # So W2 for M2 should be at index 2.
    # But M2 is a 2x2 matrix product? No, M2 is a scalar product of linear combinations.
    # In Strassen, M2 is a scalar.
    # The "W2" I am defining here is the vector for the 2x2 matrix product tensor.
    # The tensor for 2x2x2 multiplication is T_{(i,j), (j,k), (k,i)}.
    # The rank 7 decomposition expresses T as sum of 7 rank 1 tensors.
    # Each rank 1 tensor corresponds to one of the 7 multiplications.
    # So for M2, the "multiplication" is between linear combo of A and linear combo of B.
    # The result is added to C.
    # So W2[r] should indicate which part of the 2x2 result C this multiplication contributes to.
    # M2 contributes to C10.
    # So W2[2, 1] = 1?
    # Let's check the standard Strassen decomposition for 2x2.
    # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    # u_r is 2x2 matrix, v_r is 2x2 matrix, w_r is 2x2 matrix.
    # M1: u = [[1,0],[0,1]], v = [[1,0],[0,1]], w = [[1,0],[0,1]]? No.
    # M1 = (a+d)(e+h).
    # u_1 = [[1,0],[0,1]] (a and d)
    # v_1 = [[1,0],[0,1]] (e and h)
    # w_1 = [[1,0],[0,1]]?
    # C00 += M1, C11 += M1.
    # So w_1 = [[1,0],[0,1]].
    # M2 = (c+d)e.
    # u_2 = [[0,0],[1,1]] (c and d)
    # v_2 = [[1,0],[0,0]] (e)
    # C10 += M2.
    # w_2 = [[0,0],[1,0]].
    # M3 = a(f-h).
    # u_3 = [[1,0],[0,0]] (a)
    # v_3 = [[0,1],[0,-1]] (f and -h)
    # C01 += M3.
    # w_3 = [[0,1],[0,0]].
    # M4 = d(g-e).
    # u_4 = [[0,0],[0,1]] (d)
    # v_4 = [[-1,0],[1,0]] (-e and g)
    # C10 += M4, C00 += M4.
    # w_4 = [[1,0],[1,0]].
    # M5 = (a+b)h.
    # u_5 = [[1,1],[0,0]] (a and b)
    # v_5 = [[0,0],[0,1]] (h)
    # C01 += M5, C00 -= M5.
    # w_5 = [[-1,1],[0,0]].
    # M6 = (c-a)(e+f).
    # u_6 = [[-1,0],[1,0]] (-a and c)
    # v_6 = [[1,1],[0,0]] (e and f)
    # C11 += M6.
    # w_6 = [[0,0],[0,1]].
    # M7 = (b-d)(g+h).
    # u_7 = [[0,1],[-1,0]] (b and -d)
    # v_7 = [[0,1],[1,1]] (g and h)
    # C00 += M7, C11 += M7.
    # w_7 = [[1,0],[0,1]].
    
    # Let's populate U2, V2, W2 (4x7)
    # Flatten 2x2 matrices to 4 vectors.
    # Order: 00, 01, 10, 11.
    
    # M1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1
    
    # M3
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1
    
    # M4
    U2[3,3]=1
    V2[0,3]=-1; V2[2,3]=1
    W2[0,3]=1; W2[2,3]=1
    
    # M5
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # M6
    U2[0,5]=-1; U2[2,5]=1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7
    U2[1,6]=1; U2[2,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1; W2[3,6]=1

    # Now build 4x4x4 Rank 49
    U4 = []
    V4 = []
    W4 = []
    
    # Block indices: 00, 01, 10, 11
    # 4x4 indices:
    # 00: 0,1,4,5
    # 01: 2,3,6,7
    # 10: 8,9,12,13
    # 11: 10,11,14,15
    
    def get_block_indices(block):
        if block == 0: return [0, 1, 4, 5]
        if block == 1: return [2, 3, 6, 7]
        if block == 2: return [8, 9, 12, 13]
        if block == 3: return [10, 11, 14, 15]

    # Strassen 4x4 steps
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    steps = [
        # (A_blocks, signs, B_blocks, signs, C_blocks, coeffs)
        # P1
        ([0, 3], [1, 1], [0, 3], [1, 1], [(0, 1), (3, 1)]),
        # P2
        ([2, 3], [1, 1], [0], [1], [(2, 1)]),
        # P3
        ([0], [1], [1, 3], [1, -1], [(1, 1)]),
        # P4
        ([3], [1], [2, 0], [1, -1], [(2, 1), (0, 1)]),
        # P5
        ([0, 1], [1, 1], [3], [1], [(1, 1), (0, -1)]),
        # P6
        ([2, 0], [1, -1], [0, 1], [1, 1], [(3, 1)]),
        # P7
        ([1, 3], [1, -1], [2, 3], [1, 1], [(0, 1), (3, 1)])
    ]
    
    # Note: C_coeffs in steps above are (block, coeff)
    # C00: P1+P4-P5+P7
    # C01: P3+P5
    # C10: P2+P4
    # C11: P1-P2+P3+P6
    
    # Wait, my steps list C_coeffs are wrong.
    # I should define the contribution of each Pk to each Cij.
    
    contrib = {
        0: {0: 1, 3: 1}, # P1 -> C00, C11
        1: {2: 1},       # P2 -> C10
        2: {1: 1},       # P3 -> C01
        3: {2: 1, 0: 1}, # P4 -> C10, C00
        4: {1: 1, 0: -1},# P5 -> C01, C00
        5: {3: 1},       # P6 -> C11
        6: {0: 1, 3: 1}  # P7 -> C00, C11
    }
    
    for k in range(7):
        A_blocks, A_signs, B_blocks, B_signs = steps[k][0], steps[k][1], steps[k][2], steps[k][3]
        for r in range(7):
            u_vec = np.zeros(16)
            v_vec = np.zeros(16) # Will be padded to 20 later
            w_vec = np.zeros(16) # Will be padded to 20 later
            
            for b_idx, sign in zip(A_blocks, A_signs):
                indices = get_block_indices(b_idx)
                for i, idx in enumerate(indices):
                    u_vec[idx] += sign * U2[i, r]
            
            for b_idx, sign in zip(B_blocks, B_signs):
                indices = get_block_indices(b_idx)
                for i, idx in enumerate(indices):
                    v_vec[idx] += sign * V2[i, r]
            
            for c_block, coeff in contrib[k].items():
                indices = get_block_indices(c_block)
                for i, idx in enumerate(indices):
                    w_vec[idx] += coeff * W2[i, r]
            
            U4.append(u_vec)
            V4.append(v_vec)
            W4.append(w_vec)
            
    U4 = np.array(U4).T # 16x49
    V4 = np.array(V4).T # 16x49
    W4 = np.array(W4).T # 16x49
    
    # --- Part 2: 4x4x1 Rank 16 ---
    # A (4x4) * b (4x1) = c (4x1)
    # c_i = sum_j A_ij b_j
    # Rank 16.
    U5 = []
    V5 = []
    W5 = []
    
    for i in range(4):
        for j in range(4):
            u_vec = np.zeros(16)
            v_vec = np.zeros(4)
            w_vec = np.zeros(4)
            
            u_vec[i*4+j] = 1
            v_vec[j] = 1
            w_vec[i] = 1
            
            U5.append(u_vec)
            V5.append(v_vec)
            W5.append(w_vec)
            
    U5 = np.array(U5).T # 16x16
    V5 = np.array(V5).T # 4x16
    W5 = np.array(W5).T # 4x16
    
    # --- Combine ---
    # Total Rank 65
    U_final = np.zeros((16, 65))
    V_final = np.zeros((20, 65))
    W_final = np.zeros((20, 65))
    
    # Copy 4x4x4 part
    U_final[:, :49] = U4
    V_final[:16, :49] = V4
    W_final[:16, :49] = W4
    
    # Copy 4x4x1 part
    U_final[:, 49:] = U5
    V_final[16:, 49:] = V5
    W_final[16:, 49:] = W5
    
    # Convert to list of lists of integers
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
