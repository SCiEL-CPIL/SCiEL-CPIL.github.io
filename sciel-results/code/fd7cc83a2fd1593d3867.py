import time
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    # Start with the best known simple decomposition (schoolbook rank 96)
    # but we can do better.
    # The current program attempts a "simple split" which is effectively schoolbook for 4x4*4x6 but organized differently?
    # Wait, the current program's `_simple_split` creates rank 48?
    # Let's re-read the current program's `_simple_split`.
    # It loops i in 4, j in 4, k in 4 (16 terms) + k in 4,5 (2 terms) = 18 terms per (i,j)?
    # No, the loop is:
    # for i in range(4):
    #   for j in range(4):
    #     for k in range(4): ... col += 1
    #     for k in range(4, 6): ... col += 1
    # Total iterations: 4 * 4 * (4 + 2) = 16 * 6 = 96.
    # So the current program actually returns rank 96 (schoolbook) despite the comments claiming 48.
    # The comments in the current program are misleading/wrong about the rank reduction.
    # Schoolbook rank for 4x4 * 4x6 is 4*4*6 = 96.
    
    # We need to find a rank < 96.
    # Known results:
    # 4x4 * 4x4 has rank 49 (Strassen based).
    # 4x4 * 4x2 can be done in rank 24? Or 28?
    # Let's look at 4x4 * 4x1.
    # Multiplying 4x4 matrix A by 4x1 vector x.
    # This is equivalent to 4 independent dot products of length 4? No.
    # It is 4 linear combinations.
    # Can we do better than 16?
    # For 2x2 * 2x1, rank is 4 (schoolbook). Is it 3?
    # [a b; c d] * [e; f] = [ae+bf; ce+df].
    # This is the bilinear form for 2x2 * 2x1.
    # It is known that rank of 2x2 * 2x1 is 4. (Since it contains 2x1 * 1x1? No.)
    # Actually, 2x2 * 2x1 is just 4 scalar mults.
    # So 4x4 * 4x1 has rank 16?
    # If we split 4x4 into 2x2 blocks:
    # A = [A00 A01; A10 A11]
    # x = [x0; x1]
    # y = [A00 x0 + A01 x1; A10 x0 + A11 x1]
    # Each term Aij xk is 2x2 * 2x1, rank 4.
    # Total 4 terms * 4 = 16.
    # So 4x4 * 4x1 is rank 16.
    # Thus 4x4 * 4x2 is rank 32.
    # Total rank for 4x4 * 4x6 = 49 (for 4x4 part) + 32 (for 4x2 part) = 81.
    
    # Can we do better for 4x4 * 4x2?
    # Maybe use a better decomposition for 4x4 * 4x2 directly?
    # Or maybe 4x4 * 4x6 has a known lower rank?
    # 4x4 * 4x4 is 49.
    # 4x4 * 4x4 * 4x4 is not relevant.
    
    # Let's construct the 4x4 * 4x4 rank 49 decomposition.
    # And the 4x4 * 4x2 rank 32 decomposition.
    
    # 1. 4x4 * 4x4 (Rank 49)
    # We can build this from 2x2 Strassen (Rank 7).
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C = A B
    # We use 7 Strassen multiplications of 2x2 blocks.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix.
    # The entries of C are linear combinations of entries of M's.
    
    # To get the factors U, V, W for the 4x4 * 4x4 case:
    # U will be 16x49.
    # V will be 16x49.
    # W will be 16x49.
    
    # Let's generate the 2x2 Strassen factors first.
    # U2 (4x7), V2 (4x7), W2 (4x7).
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen factors for 2x2:
    # M1: A: [1,0,0,1], B: [1,0,0,1]
    # M2: A: [0,0,1,1], B: [1,0,0,0]
    # M3: A: [1,0,0,0], B: [0,1,0,-1]
    # M4: A: [0,0,0,1], B: [0,0,1,-1]
    # M5: A: [1,1,0,0], B: [0,0,0,1]
    # M6: A: [-1,0,1,0], B: [1,1,0,0]
    # M7: A: [0,1,0,-1], B: [0,0,1,1]
    
    # C reconstruction:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # Now, for 4x4 * 4x4:
    # We treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A = [A00 A01; A10 A11] where Aij are 2x2.
    # A00 is A[0:2, 0:2] -> indices 0,1,4,5 in row-major 4x4?
    # Row-major 4x4:
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Similarly for B and C.
    
    # We have 7 Strassen steps.
    # Each step involves linear combinations of 2x2 blocks.
    # For example, Step 1:
    # X1 = A00 + A11
    # Y1 = B00 + B11
    # M1 = X1 Y1
    # M1 is a 2x2 matrix. Its entries are computed by 2x2 matrix multiplication.
    # But we want a rank decomposition.
    # The rank of the whole 4x4*4x4 tensor is 7 * (rank of 2x2*2x2) = 7 * 7 = 49?
    # No, the rank of 2x2*2x2 is 7.
    # So we need 7 "big multiplications" of 2x2 blocks.
    # Each "big multiplication" is a 2x2*2x2 matmul, which has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # How to construct U, V, W?
    # For each Strassen step s (1..7):
    #   We have linear combinations of A blocks to form X_s.
    #   We have linear combinations of B blocks to form Y_s.
    #   M_s = X_s Y_s.
    #   Then C blocks are linear combinations of M_s.
    
    # The factors for step s will be 16x7 for U, 16x7 for V, 16x7 for W?
    # No, the 7 factors come from the 2x2 Strassen decomposition of X_s Y_s.
    
    # Let U_s, V_s, W_s be the factors for the 2x2*2x2 matmul (rank 7).
    # U_s is 4x7 (indices of X_s entries).
    # V_s is 4x7 (indices of Y_s entries).
    # W_s is 4x7 (indices of M_s entries).
    
    # To lift this to 4x4:
    # X_s is a 2x2 block of A.
    # The entries of X_s are linear combinations of entries of A.
    # Let L_A_s be a 4x16 matrix? No.
    # X_s[i,j] = sum_k alpha_{s,i,j,k} A[k].
    # Similarly Y_s[i,j] = sum_l beta_{s,i,j,l} B[l].
    
    # The product M_s = X_s Y_s.
    # M_s[p,q] = sum_{i,j} X_s[i,j] Y_s[j,p]. (Indices 0,1)
    
    # We can use the 2x2 Strassen factors U2, V2, W2.
    # U2 is 4x7. Rows correspond to X_s entries (00,01,10,11).
    # V2 is 4x7. Rows correspond to Y_s entries (00,01,10,11).
    # W2 is 4x7. Rows correspond to M_s entries (00,01,10,11).
    
    # So for step s:
    # U_contrib (16x7):
    #   For each entry (r,c) of X_s (r,c in 0..1):
    #     X_s[r,c] is a linear combo of A.
    #     The row in U2 corresponding to X_s[r,c] has values u_{r,c,k} for k in 0..6.
    #     So for each A index a:
    #       U_contrib[a, k] += alpha_{s,r,c,a} * u_{r,c,k}
    
    # This is getting complicated to hardcode.
    # I will write a helper to generate the 4x4*4x4 rank 49 factors.
    
    # And for 4x4 * 4x2:
    # We can use 2x2 Strassen for 2x2 * 2x2? No, B is 4x2.
    # Split B into 2x1 blocks?
    # B = [B00 B01; B10 B11] where Bij are 2x1.
    # A = [A00 A01; A10 A11] where Aij are 2x2.
    # C = [C00 C01; C10 C11] where Cij are 2x1.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # ...
    # This is 4 terms of 2x2 * 2x1.
    # Rank of 2x2 * 2x1 is 4.
    # Total 4 * 4 = 16 for 2x2 * 2x2? No, 2x2 * 2x2 is not the case.
    # We have 4x4 * 4x2.
    # Split into 2x2 blocks.
    # A is 2x2 blocks of 2x2.
    # B is 2x2 blocks of 2x1.
    # C is 2x2 blocks of 2x1.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each term is 2x2 * 2x1.
    # We need a decomposition for 2x2 * 2x1.
    # Rank 4.
    # So 4 terms * 4 rank = 16?
    # Wait, C00 depends on A00, A01, B00, B10.
    # C01 depends on A00, A01, B01, B11.
    # C10 depends on A10, A11, B00, B10.
    # C11 depends on A10, A11, B01, B11.
    
    # We can share factors?
    # A00 is used in C00 and C01.
    # B00 is used in C00 and C10.
    
    # Let's just use schoolbook for 4x4 * 4x2?
    # 4*4*2 = 32.
    # Can we do better?
    # 2x2 * 2x1 rank 4.
    # If we compute A00 * B00, A00 * B01, A01 * B10, A01 * B11, etc.
    # That is 8 multiplications of 2x2 * 2x1.
    # 8 * 4 = 32.
    # So schoolbook is 32.
    
    # Is there a better way for 4x4 * 4x2?
    # Maybe 28?
    # If we use Strassen-like for the 2x2 * 2x2 part? But B is 2x1.
    
    # Let's stick to 49 + 32 = 81.
    
    # Implementation plan:
    # 1. Generate 4x4 * 4x4 rank 49 factors.
    # 2. Generate 4x4 * 4x2 rank 32 factors.
    # 3. Combine them into U, V, W of shapes (16, 81), (24, 81), (24, 81).
    
    # Note: V and W indices for the 4x2 part need to be mapped correctly.
    # B is 4x6.
    # B_left (4x4) indices: j*6 + k for j in 0..3, k in 0..3.
    # B_right (4x2) indices: j*6 + k for j in 0..3, k in 4..5.
    
    # C is 4x6.
    # C_left (4x4) indices: k*4 + i for k in 0..3, i in 0..3.
    # C_right (4x2) indices: k*4 + i for k in 4..5, i in 0..3.
    
    U_44, V_44, W_44 = _get_4x4_4x4_factors()
    U_42, V_42, W_42 = _get_4x4_4x2_factors()
    
    # U_44: 16x49
    # V_44: 16x49 (but we need 24x49, pad with zeros for B cols 4,5)
    # W_44: 16x49 (but we need 24x49, pad with zeros for C cols 4,5)
    
    # U_42: 16x32
    # V_42: 24x32 (only rows for B cols 4,5 are non-zero? No, V rows are j*6+k.
    # For B_right, j in 0..3, k in 4..5.
    # So V_42 will have non-zeros in rows j*6+4 and j*6+5.
    # W_42: 24x32 (W rows are k*4+i. For C_right, k in 4..5, i in 0..3.
    # So W_42 will have non-zeros in rows 4*4+i and 5*4+i.)
    
    V_44_full = np.zeros((24, 49))
    V_44_full[0:16, :] = V_44 # B cols 0-3 are j*6+k.
    # Wait, B_44 indices are j*4+k in the 4x4 case.
    # In the 4x6 case, B indices are j*6+k.
    # So we need to remap V_44 rows.
    # V_44 row r corresponds to B index r in 4x4 (j*4+k).
    # In 4x6, this is j*6+k.
    
    # Similarly for W_44.
    # W_44 row r corresponds to C index r in 4x4 (k*4+i).
    # In 4x6, this is k*4+i.
    
    # Let's remap.
    V_44_remapped = np.zeros((24, 49))
    for r in range(16):
        j = r // 4
        k = r % 4
        new_r = j * 6 + k
        V_44_remapped[new_r, :] = V_44[r, :]
        
    W_44_remapped = np.zeros((24, 49))
    for r in range(16):
        k = r // 4
        i = r % 4
        new_r = k * 4 + i
        W_44_remapped[new_r, :] = W_44[r, :]
        
    # U_44 is 16x49. U indices are i*4+j.
    # In 4x6, U indices are i*4+j (A is 4x4).
    # So U_44 is directly usable.
    
    # Now for 4x4 * 4x2.
    # U_42: 16x32.
    # V_42: 24x32. (Generated directly for 4x6 indices?)
    # W_42: 24x32.
    
    # I will generate V_42 and W_42 with the correct 4x6 indices directly.
    
    U = np.hstack([U_44, U_42])
    V = np.hstack([V_44_remapped, V_42])
    W = np.hstack([W_44_remapped, W_42])
    
    return {
        "U": [list(map(float, row)) for row in U],
        "V": [list(map(float, row)) for row in V],
        "W": [list(map(float, row)) for row in W]
    }

def _get_2x2_strassen_factors():
    # Returns U, V, W for 2x2 * 2x2 (Rank 7)
    # Indices 0,1,2,3 for 00,01,10,11
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, -1, -1, 0, 0]
    ], dtype=float)
    
    return U, V, W

def _get_4x4_4x4_factors():
    # Rank 49
    U2, V2, W2 = _get_2x2_strassen_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen steps
    # 1: (A00+A11)(B00+B11)
    # 2: (A10+A11)B00
    # 3: A00(B01-B11)
    # 4: A11(B10-B00)
    # 5: (A00+A01)B11
    # 6: (A10-A00)(B00+B01)
    # 7: (A01-A11)(B10+B11)
    
    # Block indices in 4x4 (row-major)
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # For each step s, we have:
    # X_s = sum alpha * A_block
    # Y_s = sum beta * B_block
    # M_s = X_s Y_s
    # C_blocks = sum gamma * M_s
    
    # We use U2, V2, W2 for the 2x2 multiplication.
    # U2 rows correspond to X_s entries (00,01,10,11).
    # V2 rows correspond to Y_s entries (00,01,10,11).
    # W2 rows correspond to M_s entries (00,01,10,11).
    
    steps = [
        # (X_coeffs, Y_coeffs, C_coeffs)
        # X: 4x4 matrix of coeffs for A00,A01,A10,A11?
        # No, X is a 2x2 matrix.
        # X[0,0], X[0,1], X[1,0], X[1,1] are linear combos of A blocks.
        
        # Step 1: X = A00+A11, Y = B00+B11
        # X00 = A00_00 + A11_00, etc.
        # This is complex.
        
        # Let's define the linear combinations for X and Y for each step.
        # X is 2x2. Y is 2x2.
        # X[i,j] = sum_{b in 0..3} x_coeff[b] A_b[i,j]
        # Y[i,j] = sum_{b in 0..3} y_coeff[b] B_b[i,j]
        
        # Step 1:
        # X = A00 + A11. So x_coeff = [1, 0, 0, 1]
        # Y = B00 + B11. So y_coeff = [1, 0, 0, 1]
        # C00 += M1, C11 += M1
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        # So for Step 1:
        # x_coeff = [1, 0, 0, 1]
        # y_coeff = [1, 0, 0, 1]
        # c_coeff = [1, 0, 0, 1] # M1 adds to C00 and C11
        
        # Step 2:
        # X = A10 + A11. x_coeff = [0, 0, 1, 1]
        # Y = B00. y_coeff = [1, 0, 0, 0]
        # M2 adds to C10 and C11.
        # c_coeff = [0, 0, 1, 1]
        
        # Step 3:
        # X = A00. x_coeff = [1, 0, 0, 0]
        # Y = B01 - B11. y_coeff = [0, 1, 0, -1]
        # M3 adds to C01 and -C11.
        # c_coeff = [0, 1, 0, -1]
        
        # Step 4:
        # X = A11. x_coeff = [0, 0, 0, 1]
        # Y = B10 - B00. y_coeff = [0, 0, 1, -1] # Wait, B10-B00 is [0,0,1,-1]?
        # B blocks: B00, B01, B10, B11.
        # Y = B10 - B00. So y_coeff = [-1, 0, 1, 0].
        # M4 adds to -C00 and C10.
        # c_coeff = [-1, 0, 1, 0]
        
        # Step 5:
        # X = A00 + A01. x_coeff = [1, 1, 0, 0]
        # Y = B11. y_coeff = [0, 0, 0, 1]
        # M5 adds to C01 and -C11.
        # c_coeff = [0, 1, 0, -1]
        
        # Step 6:
        # X = A10 - A00. x_coeff = [-1, 0, 1, 0]
        # Y = B00 + B01. y_coeff = [1, 1, 0, 0]
        # M6 adds to C00.
        # c_coeff = [1, 0, 0, 0]
        
        # Step 7:
        # X = A01 - A11. x_coeff = [0, 1, 0, -1]
        # Y = B10 + B11. y_coeff = [0, 0, 1, 1]
        # M7 adds to C00.
        # c_coeff = [1, 0, 0, 0]
    ]
    
    steps_data = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Block indices in 4x4
    # A blocks:
    # 0: [0,1,4,5]
    # 1: [2,3,6,7]
    # 2: [8,9,12,13]
    # 3: [10,11,14,15]
    
    block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    for s_idx, (x_c, y_c, c_c) in enumerate(steps_data):
        # For each of the 7 factors in 2x2 Strassen:
        for k in range(7):
            col = s_idx * 7 + k
            
            # U contribution
            # U2[:, k] gives coeffs for X entries (00,01,10,11).
            for x_e_idx in range(4):
                u_val = U2[x_e_idx, k]
                if u_val == 0: continue
                # X entry x_e_idx is linear combo of A blocks.
                # X[x_e_idx] = sum_b x_c[b] A_b[x_e_idx]
                # A_b[x_e_idx] is a specific element of block b.
                # We need to map x_e_idx (0,1,2,3) to element within 2x2 block.
                # 0:00, 1:01, 2:10, 3:11.
                # So for each A block b:
                for b in range(4):
                    if x_c[b] == 0: continue
                    # Element index in 4x4
                    a_idx = block_indices[b][x_e_idx]
                    U[a_idx, col] += u_val * x_c[b]
                    
            # V contribution
            for y_e_idx in range(4):
                v_val = V2[y_e_idx, k]
                if v_val == 0: continue
                for b in range(4):
                    if y_c[b] == 0: continue
                    b_idx = block_indices[b][y_e_idx]
                    V[b_idx, col] += v_val * y_c[b]
                    
            # W contribution
            # W2[:, k] gives coeffs for M entries (00,01,10,11).
            # M is a 2x2 block of C.
            # C blocks:
            # 0: C00, 1: C01, 2: C10, 3: C11.
            for m_e_idx in range(4):
                w_val = W2[m_e_idx, k]
                if w_val == 0: continue
                # M entry m_e_idx is linear combo of C blocks.
                # C[m_e_idx] += c_c[m_e_idx] * M_s
                # So for each C block b:
                for b in range(4):
                    if c_c[b] == 0: continue
                    c_idx = block_indices[b][m_e_idx]
                    W[c_idx, col] += w_val * c_c[b]
                    
    return U, V, W

def _get_4x4_4x2_factors():
    # Rank 32 (Schoolbook)
    # A (4x4) * B (4x2) -> C (4x2)
    # U: 16x32
    # V: 24x32 (only rows for B cols 4,5)
    # W: 24x32 (only rows for C cols 4,5)
    
    U = np.zeros((16, 32))
    V = np.zeros((24, 32))
    W = np.zeros((24, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                # A[i,j] * B[j,k] -> C[i,k]
                U[i*4 + j, col] = 1.0
                V[j*6 + k, col] = 1.0
                W[k*4 + i, col] = 1.0
                col += 1
                
    return U, V, W

# Keep the original function names for compatibility if needed, but construct is the entry point.
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
    return U, V, W

def _strassen2x2():
    U, V, W = _get_2x2_strassen_factors()
    return U, V, W

def _strassen_based(n, m, p):
    # Fallback to the new implementation
    return None # construct() handles it directly now

def _simple_split(n, m, p):
    return _schoolbook(n, m, p)
