# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145

def construct():
    n, m, p = 4, 4, 8
    
    # Get 2x2 Strassen decomposition
    U2, V2, W2 = _get_strassen_2x2()
    
    # Build 4x4x4 decomposition
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # We will create lists of vectors for U, V, W for the 49 terms.
    U4_terms = []
    V4_terms = []
    W4_terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U4_terms.append(u_vec)
        V4_terms.append(v_vec)
        W4_terms.append(w_vec)
        
    # Define block indices
    # A: 0-3, 4-7, 8-11, 12-15
    # B: 0-3, 4-7, 8-11, 12-15
    
    # Strassen steps for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For each step, we have a 2x2 multiplication X * Y = Z.
    # X and Y are 2x2 matrices formed by linear combinations of A and B blocks.
    # We use the 2x2 Strassen decomposition (7 terms) for each step.
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    # X = A00 + A11. Y = B00 + B11.
    # For each of the 7 terms in 2x2 Strassen:
    # u_term is a linear combo of X's elements.
    # v_term is a linear combo of Y's elements.
    # w_term is a linear combo of Z's elements (which contribute to C blocks).
    
    # Let's map the 2x2 indices to 4x4 block indices.
    # In 2x2, A indices are 0,1,2,3.
    # In 4x4 step 1, X = A00 + A11.
    # X[0] = A[0] + A[12]
    # X[1] = A[1] + A[13]
    # X[2] = A[2] + A[14]
    # X[3] = A[3] + A[15]
    
    # Y = B00 + B11.
    # Y[0] = B[0] + B[12]
    # ...
    
    # Z contributes to C00 and C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 and C11.
    # In the 2x2 decomposition, W vectors point to C indices 0,1,2,3.
    # Here, C index 0 corresponds to C00 block? No.
    # The 2x2 multiplication result Z is a 2x2 matrix.
    # Z[0] is the top-left element of M1.
    # M1 is a 2x2 matrix.
    # M1[0,0] is an element of C00.
    # M1[0,1] is an element of C00.
    # M1[1,0] is an element of C00.
    # M1[1,1] is an element of C00.
    # Wait, M1 is the product of two 2x2 matrices, so it is a 2x2 matrix.
    # And C00 is a 2x2 matrix.
    # So M1 adds directly to C00.
    # And M1 adds to C11.
    
    # So for Step 1:
    # The "C" space for the 2x2 decomposition is actually two copies: one for C00 and one for C11.
    # But the W vectors in the 2x2 decomposition are length 4.
    # We need to map these 4 outputs to the 16 outputs of 4x4.
    # For Step 1, the output Z (2x2) is added to C00 and C11.
    # So for each term in the 2x2 decomposition, the W vector (length 4) should be mapped to:
    # C00 indices (0-3) and C11 indices (12-15).
    # Specifically, if W_2x2 has 1 at index i, then W_4x4 should have 1 at C00[i] and 1 at C11[i].
    
    # Let's implement a generic step function.
    
    def process_step(A_coeffs, B_coeffs, C_target_blocks, C_signs):
        # A_coeffs: list of 4 lists, each of length 16, representing the 4 elements of the 2x2 matrix X
        # B_coeffs: list of 4 lists, each of length 16, representing the 4 elements of the 2x2 matrix Y
        # C_target_blocks: list of block indices (0,1,2,3) corresponding to C00, C01, C10, C11
        # C_signs: list of signs corresponding to the blocks
        
        # For each of the 7 terms in U2, V2, W2:
        for t in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # Calculate u_vec (linear combo of A)
            for i in range(4): # 4 elements of X
                # U2[i][t] is the coefficient for A_element_i in the 2x2 decomposition
                coeff = U2[i][t]
                if coeff != 0:
                    # X[i] is defined by A_coeffs[i]
                    for j in range(16):
                        if A_coeffs[i][j] != 0:
                            u_vec[j] += coeff * A_coeffs[i][j]
                            
            # Calculate v_vec (linear combo of B)
            for i in range(4):
                coeff = V2[i][t]
                if coeff != 0:
                    for j in range(16):
                        if B_coeffs[i][j] != 0:
                            v_vec[j] += coeff * B_coeffs[i][j]
                            
            # Calculate w_vec (linear combo of C)
            # W2[i][t] is the coefficient for C_element_i in the 2x2 decomposition
            # C_element_i corresponds to the i-th element of the 2x2 result Z.
            # Z is added to the target blocks.
            for i in range(4):
                coeff = W2[i][t]
                if coeff != 0:
                    for b_idx, sign in zip(C_target_blocks, C_signs):
                        # Block b_idx starts at b_idx * 4
                        start = b_idx * 4
                        w_vec[start + i] += coeff * sign
                        
            add_term(u_vec, v_vec, w_vec)

    # Define coefficients for A and B blocks
    # A00: 0-3, A01: 4-7, A10: 8-11, A11: 12-15
    def block_coeff(block_idx):
        coeffs = []
        for i in range(4):
            vec = [0] * 16
            vec[block_idx * 4 + i] = 1
            coeffs.append(vec)
        return coeffs
        
    A00 = block_coeff(0)
    A01 = block_coeff(1)
    A10 = block_coeff(2)
    A11 = block_coeff(3)
    
    B00 = block_coeff(0)
    B01 = block_coeff(1)
    B10 = block_coeff(2)
    B11 = block_coeff(3)
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    X1 = [ [a+b for a,b in zip(A00[i], A11[i])] for i in range(4) ]
    Y1 = [ [a+b for a,b in zip(B00[i], B11[i])] for i in range(4) ]
    # M1 contributes to C00 (sign 1) and C11 (sign 1)
    process_step(X1, Y1, [0, 3], [1, 1])
    
    # Step 2: M2 = (A10 + A11)B00
    X2 = [ [a+b for a,b in zip(A10[i], A11[i])] for i in range(4) ]
    Y2 = B00
    # M2 contributes to C10 (sign 1) and C11 (sign -1)
    process_step(X2, Y2, [2, 3], [1, -1])
    
    # Step 3: M3 = A00(B01 - B11)
    X3 = A00
    Y3 = [ [a-b for a,b in zip(B01[i], B11[i])] for i in range(4) ]
    # M3 contributes to C01 (sign 1) and C11 (sign 1)
    process_step(X3, Y3, [1, 3], [1, 1])
    
    # Step 4: M4 = A11(B10 - B00)
    X4 = A11
    Y4 = [ [a-b for a,b in zip(B10[i], B00[i])] for i in range(4) ]
    # M4 contributes to C00 (sign 1) and C10 (sign 1)
    process_step(X4, Y4, [0, 2], [1, 1])
    
    # Step 5: M5 = (A00 + A01)B11
    X5 = [ [a+b for a,b in zip(A00[i], A01[i])] for i in range(4) ]
    Y5 = B11
    # M5 contributes to C00 (sign -1) and C01 (sign 1)
    process_step(X5, Y5, [0, 1], [-1, 1])
    
    # Step 6: M6 = (A10 - A00)(B00 + B01)
    X6 = [ [a-b for a,b in zip(A10[i], A00[i])] for i in range(4) ]
    Y6 = [ [a+b for a,b in zip(B00[i], B01[i])] for i in range(4) ]
    # M6 contributes to C11 (sign 1)
    process_step(X6, Y6, [3], [1])
    
    # Step 7: M7 = (A01 - A11)(B10 + B11)
    X7 = [ [a-b for a,b in zip(A01[i], A11[i])] for i in range(4) ]
    Y7 = [ [a+b for a,b in zip(B10[i], B11[i])] for i in range(4) ]
    # M7 contributes to C00 (sign 1)
    process_step(X7, Y7, [0], [1])
    
    # U4_terms, V4_terms, W4_terms now contain 49 terms.
    # Each term is a vector of length 16.
    
    # Now, build 4x4x8 decomposition.
    # We have two independent 4x4x4 multiplications:
    # C1 = A * B1 (B1 is first 4 cols of B, C1 is first 4 cols of C)
    # C2 = A * B2 (B2 is second 4 cols of B, C2 is second 4 cols of C)
    
    # B indices: 0-15 for B1, 16-31 for B2
    # C indices: 0-15 for C1, 16-31 for C2
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    col = 0
    # First 49 terms for B1/C1
    for t in range(49):
        for i in range(16):
            U[i][col] = U4_terms[t][i]
            # V for B1: indices 0-15
            V[i][col] = V4_terms[t][i]
            # W for C1: indices 0-15
            W[i][col] = W4_terms[t][i]
        col += 1
        
    # Next 49 terms for B2/C2
    for t in range(49):
        for i in range(16):
            U[i][col] = U4_terms[t][i]
            # V for B2: indices 16-31
            V[i + 16][col] = V4_terms[t][i]
            # W for C2: indices 16-31
            W[i + 16][col] = W4_terms[t][i]
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W
# EVOLVE-BLOCK-END
