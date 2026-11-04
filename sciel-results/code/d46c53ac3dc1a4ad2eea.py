import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # 1. Get 2x2x2 Strassen decomposition
    # U, V, W are 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M1 contributes to C00, C11
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M2 contributes to C10, C11
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    # M3 contributes to C01, C11
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    # M4 contributes to C00, C10
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M5 contributes to C00, C01
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    # M6 contributes to C11
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B11+B01)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[3, 6] = 1; V2[1, 6] = 1
    # M7 contributes to C00
    W2[0, 6] = 1

    # 2. Build 4x4x4 decomposition (Rank 49)
    # We use the recursive structure.
    # U4, V4, W4 are 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 7 macro-terms of Strassen for 4x4
    # Each macro-term k corresponds to a product of two 2x2 blocks.
    # We need to map the 2x2 Strassen indices to the 4x4 indices.
    
    # Macro-term definitions (X, Y, C_blocks)
    # X is a linear comb of A blocks, Y of B blocks.
    # C_blocks is where M_k goes in C.
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B10-B11) -> C01, C11
    # M4 = A11(B01-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B10) -> C11
    # M7 = (A01-A11)(B11+B01) -> C00
    
    # We iterate over the 7 macro-terms.
    # For each, we iterate over the 7 micro-terms of 2x2 Strassen.
    
    macro_terms = [
        # (A_blocks, B_blocks, C_blocks, coeffs)
        # A_blocks: list of (block_idx, coeff)
        # B_blocks: list of (block_idx, coeff)
        # C_blocks: list of (block_idx, coeff)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]), # M2
        ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (1, 1), (3, 1) ]), # M3
        ([ (3, 1) ], [ (1, 1), (0, -1) ], [ (0, 1), (2, 1) ]), # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]), # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (3, 1) ]), # M6
        ([ (1, 1), (3, -1) ], [ (3, 1), (1, 1) ], [ (0, 1) ])  # M7
    ]
    
    for k in range(7):
        A_defs, B_defs, C_defs = macro_terms[k]
        
        # For each micro-term r in 2x2 Strassen
        for r in range(7):
            col = k * 7 + r
            
            # Fill U4
            # U2[:, r] has non-zeros at 4 indices (0..3)
            # These correspond to 2x2 block indices.
            # We need to map 2x2 block index b to 4x4 block index.
            # The 2x2 matrix X is formed by A_defs.
            # X is a 2x2 matrix. Its elements are linear combs of A blocks.
            # U2[:, r] is the vector for X.
            # So if U2[idx, r] is non-zero, it contributes to the X[idx] position.
            # X[idx] is a linear comb of A blocks defined by A_defs?
            # No. X is the input to the 2x2 multiplication.
            # X is constructed from A.
            # For example, for M1, X = A00 + A11.
            # This X is a 2x2 matrix.
            # The 2x2 Strassen decomposition of X*Y uses U2, V2, W2.
            # U2[:, r] is the vector such that <U2[:, r], X> is the scalar.
            # So if U2[i, r] = c, it means X[i] contributes c.
            # X[i] is the i-th element of the 2x2 matrix X.
            # But X itself is built from A blocks.
            
            # This logic is slightly tricky.
            # Let's re-evaluate.
            # M_k = X_k Y_k.
            # X_k is a 2x2 matrix.
            # X_k is defined by A_defs.
            # A_defs is a list of (block_idx, coeff).
            # So X_k = sum_{(b, c) in A_defs} c * A_b.
            # Here A_b is the b-th 2x2 block of A.
            
            # The 2x2 Strassen decomposition computes M_k = X_k Y_k.
            # It uses U2, V2, W2.
            # The scalar term for micro-r is <U2[:, r], X_k> * <V2[:, r], Y_k> * W2[:, r].
            # <U2[:, r], X_k> = sum_{i=0}^3 U2[i, r] * X_k[i].
            # X_k[i] is the i-th element of the 2x2 matrix X_k.
            # But X_k is a matrix of linear forms in A.
            # X_k[i] = sum_{(b, c) in A_defs} c * A_{b, i}.
            # So <U2[:, r], X_k> = sum_{i} U2[i, r] sum_{(b, c) in A_defs} c A_{b, i}
            # = sum_{(b, c) in A_defs} c sum_{i} U2[i, r] A_{b, i}.
            
            # So the coefficient for A_{b, i} in the U vector is U2[i, r] * c.
            
            # Similarly for V.
            
            # And for W:
            # The result M_k is a 2x2 matrix.
            # It contributes to C blocks according to C_defs.
            # C_block_j = sum_{(b, c) in C_defs} c * M_k.
            # M_k = sum_{i=0}^3 (term_i) * E_i ? No.
            # M_k is a matrix.
            # The 2x2 Strassen W2[:, r] is a vector such that M_k = sum_r term_r * W2[:, r] (as matrices).
            # So the contribution to C_block_j is c * sum_r term_r * W2[:, r].
            # So the coefficient for C_{j, i} in the W vector is W2[i, r] * c.
            
            # Implementation:
            # For each (b, c) in A_defs:
            #   For i in 0..3:
            #     U4[4*b + i, col] += c * U2[i, r]
            
            # For each (b, c) in B_defs:
            #   For i in 0..3:
            #     V4[4*b + i, col] += c * V2[i, r]
            
            # For each (b, c) in C_defs:
            #   For i in 0..3:
            #     W4[4*b + i, col] += c * W2[i, r]
            
            for b, c in A_defs:
                for i in range(4):
                    U4[4*b + i, col] += c * U2[i, r]
            
            for b, c in B_defs:
                for i in range(4):
                    V4[4*b + i, col] += c * V2[i, r]
            
            for b, c in C_defs:
                for i in range(4):
                    W4[4*b + i, col] += c * W2[i, r]

    # 3. Construct final matrices
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Part 1: Cols 0-3
    U[:, 0:49] = U4
    # V mapping: V4 (16x49) -> V (36x49)
    # V4 index j*4+k -> V index j*9+k
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V4[j*4 + k, 0:49]
    # W mapping: W4 (16x49) -> W (36x49)
    # W4 index k*4+i -> W index k*4+i (since k<4)
    W[:, 0:49] = W4
    
    # Part 2: Cols 4-7
    U[:, 49:98] = U4
    # V mapping: V4 index j*4+k -> V index j*9+(k+4)
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), 49:98] = V4[j*4 + k, 0:49]
    # W mapping: W4 index k*4+i -> W index (k+4)*4+i
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, 0:49]
            
    # Part 3: Col 8
    for i in range(4):
        for j in range(4):
            col = 98 + i*4 + j
            U[4*i + j, col] = 1
            V[4*j + 8, col] = 1
            W[4*8 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
