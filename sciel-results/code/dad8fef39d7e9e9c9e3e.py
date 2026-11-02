import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition (Rank 56) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Construct M_{2,4,7} with rank 28 using Kronecker product
    # M_{2,4,7} = M_{2,2,4} ⊗ M_{1,2,1}
    # where M_{2,2,4} has rank 14, M_{1,2,1} has rank 2
    
    # First, construct M_{2,2,4} (2x2 * 2x4) with rank 14
    # Split into two M_{2,2,2} blocks (each rank 7 via Strassen)
    
    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a)(e+h), M2 = (a+c)(h), M3 = (c+d)(e),
    # M4 = (b-d)(g+h), M5 = (a+b)(e), M6 = (c-a)(f+h),
    # M7 = (b-c)(e+f)
    # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    
    # U vectors for Strassen (A side, flattened [a,b,c,d]):
    strassen_U = [
        [1, 0, 0, 0],  # M1: a
        [1, 0, 1, 0],  # M2: a+c
        [0, 0, 1, 1],  # M3: c+d
        [0, 1, 0, -1], # M4: b-d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, -1, 0], # M7: b-c
    ]
    
    # V vectors for Strassen (B side, flattened [e,f,g,h]):
    strassen_V = [
        [1, 0, 0, 1],  # M1: e+h
        [0, 0, 0, 1],  # M2: h
        [1, 0, 0, 0],  # M3: e
        [0, 0, 1, 1],  # M4: g+h
        [1, 0, 0, 0],  # M5: e
        [0, 1, 0, 1],  # M6: f+h
        [1, 1, 0, 0],  # M7: e+f
    ]
    
    # W vectors for Strassen (C side, flattened [C11,C12,C21,C22]):
    strassen_W = [
        [1, 0, 0, 1],  # M1: C11 += 1, C22 += 1
        [0, 0, 1, -1], # M2: C21 += 1, C22 -= 1
        [0, 0, 1, 1],  # M3: C21 += 1, C22 += 1
        [1, 0, 1, 0],  # M4: C11 += 1, C21 += 1
        [-1, 1, 0, 0], # M5: C11 -= 1, C12 += 1
        [0, 0, 0, 1],  # M6: C22 += 1
        [1, 0, 0, 0],  # M7: C11 += 1
    ]
    
    # Construct M_{2,2,4} by concatenating two Strassen blocks
    # First block: columns 0,1 of the 2x4 matrix
    # Second block: columns 2,3 of the 2x4 matrix
    
    U_224 = []
    V_224 = []
    W_224 = []
    
    # First Strassen block (columns 0,1)
    for r in range(7):
        U_224.append(strassen_U[r])  # A side: [a,b,c,d]
        # B side for first block: [e,f,g,h] corresponds to B[0,0],B[0,1],B[1,0],B[1,1]
        # In the 2x4 matrix B, this is B[0,0],B[0,1],B[1,0],B[1,1]
        V_224.append(strassen_V[r])  # [e,f,g,h] -> indices 0,1,4,5 in flattened 2x4
        W_224.append(strassen_W[r])  # C side: [C11,C12,C21,C22]
    
    # Second Strassen block (columns 2,3)
    for r in range(7):
        U_224.append(strassen_U[r])  # A side: [a,b,c,d]
        # B side for second block: [e,f,g,h] corresponds to B[0,2],B[0,3],B[1,2],B[1,3]
        # In the 2x4 matrix B, this is B[0,2],B[0,3],B[1,2],B[1,3]
        # Flattened 2x4: [B00,B01,B02,B03,B10,B11,B12,B13]
        # So e=B02(idx2), f=B03(idx3), g=B12(idx6), h=B13(idx7)
        v_second = [0, 0, 0, 0, 0, 0, 0, 0]
        v_second[2] = strassen_V[r][0]  # e -> idx 2
        v_second[3] = strassen_V[r][1]  # f -> idx 3
        v_second[6] = strassen_V[r][2]  # g -> idx 6
        v_second[7] = strassen_V[r][3]  # h -> idx 7
        V_224.append(v_second)
        # C side for second block: [C13,C14,C23,C24]
        # In flattened C (2x4): [C11,C12,C13,C14,C21,C22,C23,C24]
        # So C13=idx2, C14=idx3, C23=idx6, C24=idx7
        w_second = [0, 0, 0, 0, 0, 0, 0, 0]
        w_second[2] = strassen_W[r][0]  # C11 -> C13
        w_second[3] = strassen_W[r][1]  # C12 -> C14
        w_second[6] = strassen_W[r][2]  # C21 -> C23
        w_second[7] = strassen_W[r][3]  # C22 -> C24
        W_224.append(w_second)
    
    # Now M_{2,2,4} has rank 14
    # U_224: shape (4, 14)
    # V_224: shape (8, 14)
    # W_224: shape (8, 14)
    
    # Now construct M_{1,2,1} (1x2 * 2x1) with rank 2
    # A = [a, b] (1x2), B = [c, d]^T (2x1)
    # C = a*c + b*d
    # M1 = a*c, M2 = b*d
    # C = M1 + M2
    
    U_121 = [[1, 0], [0, 1]]  # a, b
    V_121 = [[1, 0], [0, 1]]  # c, d
    W_121 = [[1, 1]]           # C = M1 + M2
    
    # Now form M_{2,4,7} = M_{2,2,4} ⊗ M_{1,2,1}
    # The Kronecker product of two tensor decompositions:
    # If T1 = sum_r u1_r ⊗ v1_r ⊗ w1_r and T2 = sum_s u2_s ⊗ v2_s ⊗ w2_s
    # Then T1 ⊗ T2 = sum_{r,s} (u1_r ⊗ u2_s) ⊗ (v1_r ⊗ v2_s) ⊗ (w1_r ⊗ w2_s)
    
    # M_{2,2,4}: A is 2x2 (4 elements), B is 2x4 (8 elements), C is 2x4 (8 elements)
    # M_{1,2,1}: A is 1x2 (2 elements), B is 2x1 (2 elements), C is 1x1 (1 element)
    # M_{2,4,7}: A is 2x4 (8 elements), B is 4x7 (28 elements), C is 2x7 (14 elements)
    
    # Wait, the dimensions don't match up for a simple Kronecker product.
    # M_{2,2,4} ⊗ M_{1,2,1} would give:
    # A: 2*1 x 2*2 = 2x4 ✓
    # B: 2*2 x 4*1 = 4x4 ✗ (we need 4x7)
    # C: 2*1 x 4*1 = 2x4 ✗ (we need 2x7)
    
    # This doesn't work. Let me reconsider.
    
    # Actually, I think the correct decomposition is different.
    # Let me just return schoolbook for now.
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
