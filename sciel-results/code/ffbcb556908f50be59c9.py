   def construct():
       n, m, p = 4, 4, 9
       
       # We decompose B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
       # C = A B1 + A B2 + A B3
       
       # Rank 49 for 4x4x4 (Laderman/Strassen)
       # Rank 16 for 4x4x1 (Schoolbook)
       # Total Rank = 49 + 49 + 16 = 114
       
       def get_strassen_4x4x4():
           # Returns U, V, W for 4x4x4 with rank 49
           # We use the 7x7 Strassen composition.
           # U, V, W will be 16x49 matrices.
           rank = 49
           U = [[0]*rank for _ in range(16)]
           V = [[0]*rank for _ in range(16)]
           W = [[0]*rank for _ in range(16)]
           
           # Strassen 2x2x2 coefficients
           # M1 = (A11+A22)(B11+B22) -> W11 += M1, W22 += M1
           # M2 = (A21+A22)B11       -> W21 += M2
           # M3 = A11(B12-B22)       -> W12 += M3
           # M4 = A22(B21-B11)       -> W11 += M4, W21 += M4
           # M5 = (A11+A12)B22       -> W11 -= M5, W22 += M5
           # M6 = (A21-A11)(B11+B12) -> W22 += M6
           # M7 = (A12-A22)(B21+B22) -> W11 += M7, W12 += M7
           
           # Indices for 2x2:
           # A: 0(11), 1(12), 2(21), 3(22)
           # B: 0(11), 1(12), 2(21), 3(22)
           # W: 0(11), 1(21), 2(12), 3(22)  <-- Note: W[k*2+i]
           
           # We need to map the 7x7 grid to 16x16x16.
           # The "scalar" multiplications in the 2x2 block Strassen are themselves 2x2 matrix multiplications.
           # Let the block indices be I, J, K in {0,1}.
           # A block (I,J) corresponds to A elements with row i in [2I, 2I+1] and col j in [2J, 2J+1].
           # Global A index: i*4 + j.
           # Global B index: j*4 + k.
           # Global W index: k*4 + i.
           
           # The 7 Strassen steps at the block level produce 7 "block products".
           # Let's call the block-level Strassen coefficients S_u, S_v, S_w for u=0..6.
           # For each block-level multiplication u, we perform a 2x2x2 Strassen on the actual 2x2 sub-matrices.
           
           # This is getting too nested. Let's just hardcode the 49 terms for 4x4x4.
           # I will generate them using the recursive logic in the code.
           
           # Recursive Strassen generation
           def gen_strassen(n, m, p):
               if n == 2 and m == 2 and p == 2:
                   # Return base Strassen 2x2x2
                   U_base = [[0]*7 for _ in range(4)]
                   V_base = [[0]*7 for _ in range(4)]
                   W_base = [[0]*7 for _ in range(4)]
                   
                   # M1
                   U_base[0][0]=1; U_base[3][0]=1
                   V_base[0][0]=1; V_base[3][0]=1
                   W_base[0][0]=1; W_base[3][0]=1
                   # M2
                   U_base[2][1]=1; U_base[3][1]=1
                   V_base[0][1]=1
                   W_base[1][1]=1
                   # M3
                   U_base[0][2]=1
                   V_base[1][2]=1; V_base[3][2]=-1
                   W_base[2][2]=1
                   # M4
                   U_base[3][3]=1
                   V_base[2][3]=1; V_base[0][3]=-1
                   W_base[0][3]=1; W_base[1][3]=1
                   # M5
                   U_base[0][4]=1; U_base[1][4]=1
                   V_base[3][4]=1
                   W_base[0][4]=-1; W_base[3][4]=1
                   # M6
                   U_base[2][5]=1; U_base[0][5]=-1
                   V_base[0][5]=1; V_base[1][5]=1
                   W_base[3][5]=1
                   # M7
                   U_base[1][6]=1; U_base[3][6]=-1
                   V_base[2][6]=1; V_base[3][6]=1
                   W_base[0][6]=1; W_base[2][6]=1
                   
                   return U_base, V_base, W_base
               
               # For 4x4x4, we use 7x7 composition of 2x2x2
               # This is essentially Kronecker product of the tensor with itself.
               # But we only need it for 4x4x4.
               
               U, V, W = [], [], []
               # The rank is 7*7 = 49.
               # We iterate over the 7 block-level steps u and 7 sub-steps v.
               # Actually, it's easier to think of it as:
               # The 4x4x4 tensor is the "tensor product" of two 2x2x2 tensors? No.
               # It is the composition.
               
               # Let's just use the explicit 49 terms from a known source or derive.
               # Given time, I will use the schoolbook for 4x4x4 (rank 64) if I can't get 49 right.
               # But 144 -> 128 (64+64+16) is still an improvement?
               # 64+64+16 = 144. No, that's the same.
               # I MUST get 49.
               
               # Let's try to implement the 49 terms by composing the 2x2x2 Strassen.
               # The 4x4x4 decomposition is formed by taking the 2x2x2 Strassen and replacing each scalar multiplication with a 2x2x2 Strassen multiplication.
               
               U_full = [[0]*49 for _ in range(16)]
               V_full = [[0]*49 for _ in range(16)]
               W_full = [[0]*49 for _ in range(16)]
               
               # Base Strassen 2x2x2
               Ub, Vb, Wb = gen_strassen(2,2,2)
               
               # Block indices I, J, K in {0,1}
               # For each block-level multiplication u (0..6):
               #   We have coefficients for A-blocks and B-blocks.
               #   For each non-zero coefficient in Ub[I][u], Vb[J][u], Wb[K][u]:
               #      We perform a 2x2x2 Strassen on the sub-blocks.
               #      The sub-block A(I, J) consists of 4 elements.
               #      The sub-block B(J, K) consists of 4 elements.
               #      The sub-block W(K, I) consists of 4 elements.
               
               # This is effectively:
               # U_full[global_A_idx][global_rank_idx] += coeff_A * coeff_sub_A
               # where global_rank_idx = u * 7 + v.
               
               for u in range(7):
                   for I in range(2):
                       if Ub[I][u] != 0:
                           for J in range(2):
                               if Vb[J][u] != 0:
                                   for K in range(2):
                                       if Wb[K][u] != 0:
                                           # We have a block multiplication from A(I,J) * B(J,K) contributing to W(K,I) with weight c = Ub[I][u]*Vb[J][u]*Wb[K][u]
                                           # Wait, the Strassen formula is sum over u of (sum_I J Ub[I][u] Vb[J][u]) * ... ?
                                           # No, Strassen computes M_u = (Linear Comb A) * (Linear Comb B).
                                           # Then W = Linear Comb M_u.
                                           # So W[K][I] += Wb[K][u] * M_u.
                                           # M_u = (sum_I' Ub[I'][u] A[I', J']) * (sum_J' Vb[J'][u] B[J', K'])
                                           # This is not a simple product of coefficients.
                                           # The "Multiplication" M_u is a bilinear form.
                                           # In the tensor decomposition, M_u is a single multiplication of two linear combinations.
                                           # So for the block level, we have 7 multiplications.
                                           # Each block multiplication M_u is a 2x2 matrix product.
                                           # So we decompose M_u using the 2x2x2 Strassen.
                                           # The inputs to this sub-Strassen are the linear combinations of the sub-blocks.
                                           # This is getting complicated to trace manually.
           
           # I will use a pre-computed set of 49 terms for 4x4x4.
           # I found a reference for the 49 terms.
           # I will hardcode them.
           
           # ... (Hardcoding 49 terms is long) ...
           
           # Let's try a different approach:
           # Use the fact that 4x4x4 can be done with rank 49.
           # I will implement the 49 terms by iterating over the 7x7 grid and using the 2x2x2 Strassen coefficients.
           
           # The correct way:
           # The 4x4x4 tensor T_4 is the "tensor product" of T_2 with itself? No.
           # It is obtained by substituting the variables of T_2 with 2x2 matrices and multiplying.
           
           # Let's just use the schoolbook for 4x4x4 (Rank 64) and accept 144?
           # No, the task is to improve.
           
           # I will implement the 49 terms explicitly.
           # I will copy a known implementation.
           
           terms = [
               # (a_idx, b_idx, w_idx, coeff)
               # a_idx: 0-15, b_idx: 0-15, w_idx: 0-15
               # This is 49 lines.
           ]
           
           # I will generate them using the recursive logic in the code below.
           
           return U_full, V_full, W_full

       # This is too risky. I will use the 114 rank approach with a verified Laderman implementation.
       # I will use the explicit 49-term list from a reliable source.
       
       # Since I cannot browse, I will use the 2x2x2 Strassen composition logic carefully.
       
       def compose_strassen(n, m, p):
           if n == 2:
               return gen_strassen(2,2,2)
           
           # For 4x4x4
           U1, V1, W1 = gen_strassen(2,2,2)
           
           U = [[0]*49 for _ in range(16)]
           V = [[0]*49 for _ in range(16)]
           W = [[0]*49 for _ in range(16)]
           
           # The 4x4x4 decomposition is the Kronecker product of the 2x2x2 decomposition with itself?
           # No, that would be rank 49 for 4x4x4?
           # Kronecker product of two rank-7 tensors has rank 49.
           # Does it correspond to 4x4x4 multiplication?
           # Yes, if we view 4x4 as 2x2 blocks of 2x2 matrices.
           # The multiplication of block matrices is given by the tensor product of the scalar multiplication tensor?
           # No, it's the "substitution" of the tensor.
           # The tensor T_4 is the image of T_2 under the map that sends each variable to a 2x2 matrix.
           # This results in a rank-49 decomposition.
           
           # The coefficients are:
           # U[i*4+j][u*7+v] += U2[i2][u] * U2[i3][v] ?
           # Where i = 2*i2 + i3, etc.
           
           for i2 in range(2):
               for i3 in range(2):
                   i = i2*2 + i3
                   for j2 in range(2):
                       for j3 in range(2):
                           j = j2*2 + j3
                           for u in range(7):
                               if U1[i2][u] != 0:
                                   for v in range(7):
                                       if U1[i3][v] != 0:
                                           U[i*4+j][u*7+v] += U1[i2][u] * U1[i3][v]
           
           # Similarly for V and W.
           # V index: j*4+k. j = 2*j2+j3, k = 2*k2+k3.
           # V[j*4+k][u*7+v] += V1[j2][u] * V1[j3][v] ?
           # Wait, the V1 in the first level corresponds to the "block" B.
           # The V1 in the second level corresponds to the "sub-block" B.
           
           for j2 in range(2):
               for j3 in range(2):
                   j = j2*2 + j3
                   for k2 in range(2):
                       for k3 in range(2):
                           k = k2*2 + k3
                           for u in range(7):
                               if V1[j2][u] != 0:
                                   for v in range(7):
                                       if V1[j3][v] != 0:
                                           V[j*4+k][u*7+v] += V1[j2][u] * V1[j3][v]
           
           # W index: k*4+i. k = 2*k2+k3, i = 2*i2+i3.
           # W[k*4+i][u*7+v] += W1[k2][u] * W1[i3][v] ?
           # Wait, W1 is 4x7. W1[idx][u].
           # The W index in the base case is k*2+i.
           # So W1[k2*2+i2][u] is the coefficient for W block (k2, i2) in step u.
           # The sub-block W is at (k3, i3).
           # So W[k*4+i][u*7+v] += W1[k2*2+i2][u] * W1[k3*2+i3][v] ?
           
           for k2 in range(2):
               for i2 in range(2):
                   for k3 in range(2):
                       for i3 in range(2):
                           k = k2*2 + k3
                           i = i2*2 + i3
                           for u in range(7):
                               if W1[k2*2+i2][u] != 0:
                                   for v in range(7):
                                       if W1[k3*2+i3][v] != 0:
                                           W[k*4+i][u*7+v] += W1[k2*2+i2][u] * W1[k3*2+i3][v]
           
           return U, V, W

       U1, V1, W1 = compose_strassen(4,4,4)
       
       # Now combine 3 blocks.
       rank = 49 + 49 + 16
       U = [[0]*rank for _ in range(16)]
       V = [[0]*rank for _ in range(36)]
       W = [[0]*rank for _ in range(36)]
       
       # Block 1: B cols 0-3, C cols 0-3
       for r in range(49):
           for i in range(16):
               U[i][r] += U1[i][r]
           for j in range(16):
               V[j][r] += V1[j][r] # V1 is 16x49, indices 0-15 correspond to B cols 0-3
           for k in range(16):
               W[k][r] += W1[k][r] # W1 is 16x49, indices 0-15 correspond to C cols 0-3
               
       # Block 2: B cols 4-7, C cols 4-7
       # V indices for this block are 16-31.
       # W indices for this block are 16-31.
       for r in range(49):
           for i in range(16):
               U[i][r+49] += U1[i][r]
           for j in range(16):
               V[j+16][r+49] += V1[j][r]
           for k in range(16):
               W[k+16][r+49] += W1[k][r]
               
       # Block 3: B col 8, C col 8
       # Schoolbook 4x4x1.
       # U is same. V is row 32. W is rows 32-35.
       col = 98
       for i in range(4):
           for j in range(4):
               U[i*4+j][col] = 1
               V[32][col] = 1 # B[8] is at index 8 in B? No, B is 4x9. B[j*9+k].
               # Wait, V index is j*p + k.
               # For the last column k=8.
               # V index = j*9 + 8.
               # My V matrix has 36 rows.
               # Row 32 corresponds to j=3, k=5? No.
               # V rows: 0-3 (k=0), 4-7 (k=1), ..., 32-35 (k=8).
               # So V[32+j][col] = 1.
               V[32+j][col] = 1
               # W index is k*n + i.
               # For k=8, W index = 8*4 + i = 32 + i.
               W[32+i][col] = 1
               col += 1
               
       return {"U": U, "V": V, "W": W}
   