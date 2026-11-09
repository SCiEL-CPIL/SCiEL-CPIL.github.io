import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen 4x4 decomposition
    # Blocks:
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    # C11, C12, C21, C22
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mq is 2x2. We apply Strassen 2x2 to each Mq.
    # This yields 7 * 7 = 49 scalar multiplications.
    
    # Helper to add a term
    # U_idx: index in A (0-15)
    # V_idx: index in B (0-15)
    # W_idx: index in C (0-15)
    # coeff: coefficient
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff == 0: return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Indices for A: i*4 + k
    # A11: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A12: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # A21: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # A22: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    # Indices for B: k*4 + j
    # B11: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # B12: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # B21: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # B22: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    # Indices for C: j*4 + i (based on W[k*n+i] convention, assuming k=j, i=i)
    # C11: (0,0)=0, (0,1)=4, (1,0)=1, (1,1)=5
    # C12: (0,2)=2, (0,3)=6, (1,2)=3, (1,3)=7
    # C21: (2,0)=8, (2,1)=12, (3,0)=9, (3,1)=13
    # C22: (2,2)=10, (2,3)=14, (3,2)=11, (3,3)=15
    
    # Let's verify C index mapping:
    # C_ij is at index 4*j + i.
    # C11 (i=0,j=0) -> 0.
    # C12 (i=0,j=1) -> 4.
    # C21 (i=1,j=0) -> 1.
    # C22 (i=1,j=1) -> 5.
    # This is for 2x2 blocks.
    # For 4x4, C is 4x4.
    # Block C11 corresponds to rows 0,1 and cols 0,1 of C.
    # C[0,0] (i=0,j=0) -> 4*0+0=0.
    # C[0,1] (i=0,j=1) -> 4*1+0=4.
    # C[1,0] (i=1,j=0) -> 4*0+1=1.
    # C[1,1] (i=1,j=1) -> 4*1+1=5.
    
    # Block C12 corresponds to rows 0,1 and cols 2,3 of C.
    # C[0,2] (i=0,j=2) -> 4*2+0=8.
    # C[0,3] (i=0,j=3) -> 4*3+0=12.
    # C[1,2] (i=1,j=2) -> 4*2+1=9.
    # C[1,3] (i=1,j=3) -> 4*3+1=13.
    
    # Block C21 corresponds to rows 2,3 and cols 0,1 of C.
    # C[2,0] (i=2,j=0) -> 4*0+2=2.
    # C[2,1] (i=2,j=1) -> 4*1+2=6.
    # C[3,0] (i=3,j=0) -> 4*0+3=3.
    # C[3,1] (i=3,j=1) -> 4*1+3=7.
    
    # Block C22 corresponds to rows 2,3 and cols 2,3 of C.
    # C[2,2] (i=2,j=2) -> 4*2+2=10.
    # C[2,3] (i=2,j=3) -> 4*3+2=14.
    # C[3,2] (i=3,j=2) -> 4*2+3=11.
    # C[3,3] (i=3,j=3) -> 4*3+3=15.

    # Define the 7 Mq matrices in terms of A and B blocks.
    # Each Mq is Xq * Yq.
    # Xq is 2x2, Yq is 2x2.
    # Xq entries are linear combos of A blocks.
    # Yq entries are linear combos of B blocks.
    
    # M1: X1 = A11+A22, Y1 = B11+B22
    # M2: X2 = A21+A22, Y2 = B11
    # M3: X3 = A11, Y3 = B12-B22
    # M4: X4 = A22, Y4 = B21-B11
    # M5: X5 = A11+A12, Y5 = B22
    # M6: X6 = A21-A11, Y6 = B11+B12
    # M7: X7 = A12-A22, Y7 = B21+B22
    
    # For each Mq, we apply Strassen 2x2.
    # Let Xq = [[x11, x12], [x21, x22]] and Yq = [[y11, y12], [y21, y22]].
    # Note: In M2, Y2 is just B11, so y11=B11, y12=0, y21=0, y22=0?
    # No, Y2 is a 2x2 matrix. In M2 = (A21+A22)B11, B11 is a 2x2 block.
    # So Y2 = [[B11, 0], [0, 0]]? No.
    # M2 is a 2x2 matrix product.
    # X2 is 2x2. Y2 is 2x2.
    # In M2, the second factor is just B11.
    # So Y2 = [[B11, 0], [0, 0]]?
    # Wait, M2 = X2 * Y2.
    # If Y2 = [[B11, 0], [0, 0]], then M2 = [[X2_11 B11 + X2_12 0, ...], ...].
    # This is not correct.
    # M2 = (A21+A22) * B11.
    # This is a product of a 2x2 matrix and a 2x2 matrix.
    # So X2 = A21+A22 (2x2), Y2 = B11 (2x2).
    # So Y2 is just the block B11.
    # In the 2x2 Strassen decomposition, we treat X2 and Y2 as 2x2 matrices.
    
    # Let's define the blocks as lists of 4 indices.
    # A11: [0, 1, 4, 5]
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    # B11: [0, 1, 4, 5]
    # B12: [2, 3, 6, 7]
    # B21: [8, 9, 12, 13]
    # B22: [10, 11, 14, 15]
    
    # C11: [0, 4, 1, 5] -> indices for C[0,0], C[0,1], C[1,0], C[1,1]
    # C12: [8, 12, 9, 13] -> indices for C[0,2], C[0,3], C[1,2], C[1,3]
    # C21: [2, 6, 3, 7] -> indices for C[2,0], C[2,1], C[3,0], C[3,1]
    # C22: [10, 14, 11, 15] -> indices for C[2,2], C[2,3], C[3,2], C[3,3]
    
    # Strassen 2x2 formulas for Mq = Xq Yq:
    # P1 = (x11+x22)(y11+y22)
    # P2 = (x21+x22)y11
    # P3 = x11(y12-y22)
    # P4 = x22(y21-y11)
    # P5 = (x11+x12)y22
    # P6 = (x21-x11)(y11+y12)
    # P7 = (x12-x22)(y21+y22)
    
    # Contributions to Mq:
    # m11 = P1 + P4 - P5 + P7
    # m12 = P3 + P5
    # m21 = P2 + P4
    # m22 = P1 - P2 + P3 + P6
    
    # We need to map each P_s to U, V, W.
    # P_s is a product of a linear form in A and a linear form in B.
    # The result is a 2x2 matrix Mq.
    # Mq contributes to C blocks.
    
    # For each q (1-7), we define Xq and Yq as 2x2 matrices of linear forms.
    # Each entry is a list of (index, coeff).
    
    def get_block_indices(block_name, is_A):
        if is_A:
            if block_name == '11': return [0, 1, 4, 5]
            if block_name == '12': return [2, 3, 6, 7]
            if block_name == '21': return [8, 9, 12, 13]
            if block_name == '22': return [10, 11, 14, 15]
        else:
            if block_name == '11': return [0, 1, 4, 5]
            if block_name == '12': return [2, 3, 6, 7]
            if block_name == '21': return [8, 9, 12, 13]
            if block_name == '22': return [10, 11, 14, 15]

    # Define Xq and Yq for each q
    # Xq, Yq are 2x2 matrices. Each entry is a dict {index: coeff}
    
    def make_entry(indices, coeff):
        return {idx: coeff for idx in indices}

    def add_entries(e1, e2, sign=1):
        res = dict(e1)
        for k, v in e2.items():
            res[k] = res.get(k, 0) + sign * v
        return res

    # M1: X1 = A11+A22, Y1 = B11+B22
    X1 = [
        [add_entries(make_entry(get_block_indices('11', True), 1), make_entry(get_block_indices('22', True), 1))],
        [add_entries(make_entry(get_block_indices('11', True), 1), make_entry(get_block_indices('22', True), 1))]
    ]
    # Wait, X1 is 2x2.
    # X1 = [[A11+A22, 0], [0, A11+A22]]? No.
    # M1 = (A11+A22)(B11+B22).
    # This is a product of two 2x2 matrices.
    # X1 = A11+A22 (2x2). Y1 = B11+B22 (2x2).
    # So X1 is a single 2x2 matrix, not a 2x2 matrix of 2x2 matrices.
    # My previous block logic was for the 4x4 split into 2x2 blocks.
    # M1 is one of the 7 terms in the 4x4 Strassen.
    # M1 is a 2x2 matrix product.
    # So X1 is 2x2, Y1 is 2x2.
    # X1 = A11+A22.
    # Y1 = B11+B22.
    
    # So for M1:
    # X1_11 = A11_11 + A22_11
    # X1_12 = A11_12 + A22_12
    # X1_21 = A11_21 + A22_21
    # X1_22 = A11_22 + A22_22
    
    # This is getting complex to hardcode manually.
    # I will write a helper to construct the 49 terms programmatically.
    
    pass
