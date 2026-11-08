import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 5
    # Rank 65 = 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank_total = 65
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # We define the 2x2 Strassen factors first.
    # U2, V2, W2 of shape (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now build 4x4x4 decomposition.
    # 7 top-level terms s=0..6.
    # Each term s involves L_s and R_s which are 4x4 matrices.
    # L_s and R_s are formed by linear combinations of 2x2 blocks of A and B.
    # We represent L_s and R_s as 4x4 matrices of coefficients (0, 1, -1).
    
    # Define the 7 top-level combinations.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will generate 49 terms.
    col = 0
    for s in range(7):
        # Define L_s and R_s as 4x4 matrices of coefficients.
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        
        if s == 0:
            # L = A00+A11, R = B00+B11
            L_s[0:2, 0:2] = np.eye(2) # A00
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[0:2, 0:2] = np.eye(2) # B00
            R_s[2:4, 2:4] = np.eye(2) # B11
        elif s == 1:
            # L = A10+A11, R = B00
            L_s[2:4, 0:2] = np.eye(2) # A10
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[0:2, 0:2] = np.eye(2) # B00
        elif s == 2:
            # L = A00, R = B01-B11
            L_s[0:2, 0:2] = np.eye(2) # A00
            R_s[0:2, 2:4] = np.eye(2) # B01
            R_s[2:4, 2:4] = -np.eye(2) # -B11
        elif s == 3:
            # L = A11, R = B10-B00
            L_s[2:4, 2:4] = np.eye(2) # A11
            R_s[2:4, 0:2] = np.eye(2) # B10
            R_s[0:2, 0:2] = -np.eye(2) # -B00
        elif s == 4:
            # L = A00+A01, R = B11
            L_s[0:2, 0:2] = np.eye(2) # A00
            L_s[0:2, 2:4] = np.eye(2) # A01
            R_s[2:4, 2:4] = np.eye(2) # B11
        elif s == 5:
            # L = A10-A00, R = B00+B01
            L_s[2:4, 0:2] = np.eye(2) # A10
            L_s[0:2, 0:2] = -np.eye(2) # -A00
            R_s[0:2, 0:2] = np.eye(2) # B00
            R_s[0:2, 2:4] = np.eye(2) # B01
        elif s == 6:
            # L = A01-A11, R = B10+B11
            L_s[0:2, 2:4] = np.eye(2) # A01
            L_s[2:4, 2:4] = -np.eye(2) # -A11
            R_s[2:4, 0:2] = np.eye(2) # B10
            R_s[2:4, 2:4] = np.eye(2) # B11
            
        # Now decompose L_s * R_s using Strassen 2x2 on blocks.
        # L_s is a 2x2 matrix of 2x2 blocks.
        # We apply the 7-term Strassen decomposition to the block matrices.
        # This generates 7 new block products.
        
        # Get blocks of L_s
        L00 = L_s[0:2, 0:2]
        L01 = L_s[0:2, 2:4]
        L10 = L_s[2:4, 0:2]
        L11 = L_s[2:4, 2:4]
        
        R00 = R_s[0:2, 0:2]
        R01 = R_s[0:2, 2:4]
        R10 = R_s[2:4, 0:2]
        R11 = R_s[2:4, 2:4]
        
        # Inner Strassen terms t=0..6
        # t=0: (L00+L11)(R00+R11)
        # t=1: (L10+L11)R00
        # t=2: L00(R01-R11)
        # t=3: L11(R10-R00)
        # t=4: (L00+L01)R11
        # t=5: (L10-L00)(R00+R01)
        # t=6: (L01-L11)(R10+R11)
        
        for t in range(7):
            # Determine X and Y (2x2 matrices)
            if t == 0:
                X = L00 + L11
                Y = R00 + R11
            elif t == 1:
                X = L10 + L11
                Y = R00
            elif t == 2:
                X = L00
                Y = R01 - R11
            elif t == 3:
                X = L11
                Y = R10 - R00
            elif t == 4:
                X = L00 + L01
                Y = R11
            elif t == 5:
                X = L10 - L00
                Y = R00 + R01
            elif t == 6:
                X = L01 - L11
                Y = R10 + R11
            
            # Now we have X, Y as 2x2 matrices.
            # We want to compute X*Y.
            # We use the 2x2 Strassen decomposition U2, V2, W2.
            # The contribution to the global U, V, W for this term (s, t) and inner term u is:
            # U[global_idx, r] += X[i,j] * U2[i*2+j, u]
            # V[global_idx, r] += Y[i,j] * V2[i*2+j, u]
            # W[global_idx, r] += W2[i*2+j, u]
            
            # Wait, the indices for U, V, W in the 4x4 case are:
            # U: row is (I, J) where I, J are 0..3.
            # V: row is (J, K) where J, K are 0..3.
            # W: row is (K, I) where K, I are 0..3.
            
            # X is a 2x2 matrix. Its entries X[i,j] are coefficients for the global A entries.
            # But X is formed from L_s blocks.
            # L_s is a 4x4 matrix of coefficients.
            # L_s[I, J] is the coefficient of A[I, J].
            # So X[i,j] is actually a sum of L_s entries?
            # No, X is a 2x2 matrix.
            # The blocks of L_s are 2x2.
            # L00 is a 2x2 matrix of coefficients.
            # So X is a 2x2 matrix of coefficients?
            # No, X is a 2x2 matrix.
            # The entries of X are scalars (0, 1, -1).
            # These scalars are the coefficients of the specific A entries in that block.
            
            # Let's trace carefully.
            # L_s is 4x4.
            # L00 = L_s[0:2, 0:2]. This is a 2x2 matrix of coefficients.
            # So L00[0,0] is coeff of A[0,0].
            # X is a 2x2 matrix.
            # X[0,0] is a scalar.
            # But X[0,0] comes from L00[0,0] + L11[0,0]?
            # No, L00 and L11 are 2x2 matrices.
            # X = L00 + L11.
            # So X[0,0] = L00[0,0] + L11[0,0].
            # L00[0,0] is coeff of A[0,0].
            # L11[0,0] is coeff of A[2,2].
            # So X[0,0] is the sum of coefficients for A[0,0] and A[2,2]?
            # No, that's not right.
            # The Strassen algorithm operates on the blocks.
            # The "scalars" in the 2x2 Strassen are the blocks themselves.
            # So when we say X = L00 + L11, X is a 2x2 matrix of blocks?
            # No, L00 is a 2x2 matrix of scalars (coefficients).
            # So X is a 2x2 matrix of scalars.
            # But this loses the information of WHICH global index it corresponds to.
            
            # We need to map the 2x2 Strassen factors to the 4x4 indices.
            # The 2x2 Strassen decomposes the product of two 2x2 matrices.
            # Let the 2x2 matrices be P and Q.
            # P = [[P00, P01], [P10, P11]].
            # Q = [[Q00, Q01], [Q10, Q11]].
            # The product R = P Q.
            # The factors U2, V2, W2 give:
            # R[i,k] = sum_j sum_u P[i,j] Q[j,k] U2[i*2+j, u] V2[j*2+k, u] W2[k*2+i, u]?
            # No, the decomposition is:
            # sum_u U2[i*2+j, u] V2[j*2+k, u] W2[k*2+i, u] = 1 for all i,j,k.
            # So if we set:
            # U_444[I, J, r] = P[i,j] U2[i*2+j, u]
            # V_444[J, K, r] = Q[j,k] V2[j*2+k, u]
            # W_444[K, I, r] = W2[k*2+i, u]
            # Then sum_r U_444[I, J, r] V_444[J, K, r] W_444[K, I, r]
            # = sum_u sum_{i,j,k} P[i,j] Q[j,k] U2[i*2+j, u] V2[j*2+k, u] W2[k*2+i, u]
            # This doesn't look right. The indices I, J, K are global.
            
            # Let's use the explicit mapping.
            # For a specific inner term (s, t), we have X and Y.
            # X is a 2x2 matrix.
            # The entries of X are coefficients for the global A entries.
            # Specifically, X[i,j] is the coefficient for the entry in the (i,j) position of the 2x2 block.
            # But which block?
            # The inner Strassen operates on the 2x2 blocks of L_s.
            # So the "2x2 matrix" X is formed from the 2x2 blocks of L_s.
            # So X[0,0] is the sum of the coefficients of the top-left entries of the blocks involved.
            
            # Let's just iterate over the 16 global indices for U, V, W.
            # For each global index (I, J) in U:
            # Find which block (bI, bJ) it belongs to. bI = I//2, bJ = J//2.
            # Find local index (i, j). i = I%2, j = J%2.
            # The coefficient of A[I, J] in X is X[i,j] if the block (bI, bJ) is part of X?
            # No, X is a 2x2 matrix of blocks.
            # So X[0,0] is a 2x2 block.
            # So X[0,0][i,j] is the coefficient.
            
            # So we need to treat X as a 4x4 matrix?
            # No, X is a 2x2 matrix of 2x2 blocks.
            # So X is effectively a 4x4 matrix.
            # Let's call it X_4x4.
            # X_4x4[I, J] = X[bI, bJ][I%2, J%2].
            
            # Similarly for Y.
            
            # Then:
            # U[I, J, r] = X_4x4[I, J] * U2[i*2+j, u] ?
            # No, U2 is for the 2x2 multiplication.
            # The indices i,j in U2 are local to the 2x2 block.
            # So U[I, J, r] = X_4x4[I, J] * U2[(I%2)*2 + (J%2), u].
            
            # Let's verify.
            # sum_r U[I, J, r] V[J, K, r] W[K, I, r]
            # = sum_u sum_{i,j,k} X_4x4[I, J] U2[i*2+j, u] Y_4x4[J, K] V2[j*2+k, u] W2[k*2+i, u]
            # where i = I%2, j = J%2, k = K%2?
            # No, i,j,k are the indices in the 2x2 Strassen.
            # The 2x2 Strassen sums over j (the inner index of the 2x2 product).
            # So for a fixed I, K, we sum over J.
            # The term is X[I, J] Y[J, K].
            # The factor is U2[i*2+j, u] V2[j*2+k, u] W2[k*2+i, u].
            # Here i = I%2, k = K%2.
            # j is the local column index of I and row index of K.
            # j = J%2.
            
            # So yes, the formula holds.
            
            # Construct X_4x4 and Y_4x4.
            X_4x4 = np.zeros((4, 4))
            Y_4x4 = np.zeros((4, 4))
            
            # Fill X_4x4 from X (2x2 matrix of 2x2 blocks)
            for bI in range(2):
                for bJ in range(2):
                    # X[bI, bJ] is a 2x2 matrix
                    X_4x4[bI*2:bI*2+2, bJ*2:bJ*2+2] = X[bI, bJ]
            
            # Fill Y_4x4 from Y
            for bI in range(2):
                for bJ in range(2):
                    Y_4x4[bI*2:bI*2+2, bJ*2:bJ*2+2] = Y[bI, bJ]
            
            # Now fill U, V, W for this (s, t) and all u.
            for u in range(7):
                r = col + u
                # U
                for I in range(4):
                    for J in range(4):
                        i = I % 2
                        j = J % 2
                        U[I*4+J, r] = X_4x4[I, J] * U2[i*2+j, u]
                
                # V
                for J in range(4):
                    for K in range(4):
                        j = J % 2
                        k = K % 2
                        V[J*4+K, r] = Y_4x4[J, K] * V2[j*2+k, u]
                        # Note: V is (m*p, R). m=4, p=4 for this part.
                        # But V is (20, R). K goes 0..3.
                        # So V[J*4+K, r] is correct for K<4.
                        # For K=4, it is 0.
                
                # W
                for K in range(4):
                    for I in range(4):
                        k = K % 2
                        i = I % 2
                        W[K*4+I, r] = W2[k*2+i, u]
                        # W is (20, R). K goes 0..3.
                        # So W[K*4+I, r] is correct for K<4.
                        
                # Check time
                if time.time() - start_time > 100:
                    # Fallback to schoolbook if too slow?
                    # But this loop is small (7*7*7*16*2 + ...).
                    pass
            col += 7
            
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # Compute C[:, 4] = A * B[:, 4]
    for i in range(4):
        for j in range(4):
            r = col
            U[i*4+j, r] = 1
            # V index: j*5 + 4
            V[j*5+4, r] = 1
            # W index: 4*4 + i = 16 + i
            W[16+i, r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
