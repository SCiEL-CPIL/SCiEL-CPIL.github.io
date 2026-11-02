<<<<<<< SEARCH
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A00 is at (0,0), A11 is at (1,1)
    # B00 is at (0,0), B11 is at (1,1)
    # Contribution to C: C00 += M1, C11 += M1
    # U: (0,0) and (1,1) have value 1
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    # V: (0,0) and (1,1) have value 1
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    # W: C00 is at (0,0), C11 is at (1,1)
    W[0*2+0][0] = 1  # C00
    W[1*2+1][0] = 1  # C11
    
    # Term 2: M2 = (A10+A11)B00
    # A10 is at (1,0), A11 is at (1,1)
    # B00 is at (0,0)
    # Contribution to C: C10 += M2, C11 -= M2
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    W[0*2+1][1] = 1  # C10 (k=0, i=1)
    W[1*2+1][1] = -1  # C11 (k=1, i=1)
    
    # Term 3: M3 = A00(B10-B11)
    # A00 is at (0,0)
    # B10 is at (1,0), B11 is at (1,1)
    # Contribution to C: C01 += M3, C10 += M3
    U[0*2+0][2] = 1  # A00
    V[1*2+0][2] = 1  # B10
    V[1*2+1][2] = -1  # B11
    W[0*2+0][2] = 1  # C01 (k=0, i=0)
    W[1*2+1][2] = 1  # C10 (k=1, i=0)
    
    # Term 4: M4 = A11(B01-B00)
    # A11 is at (1,1)
    # B01 is at (0,1), B00 is at (0,0)
    # Contribution to C: C00 += M4, C10 += M4
    U[1*2+1][3] = 1  # A11
    V[0*2+1][3] = 1  # B01
    V[0*2+0][3] = -1  # B00
    W[0*2+0][3] = 1  # C00 (k=0, i=0)
    W[0*2+1][3] = 1  # C10 (k=0, i=1)
    
    # Term 5: M5 = (A00+A01)B11
    # A00 is at (0,0), A01 is at (0,1)
    # B11 is at (1,1)
    # Contribution to C: C00 -= M5, C01 += M5
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    W[0*2+0][4] = -1  # C00 (k=0, i=0)
    W[0*2+1][4] = 1  # C01 (k=0, i=1)
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # A10 is at (1,0), A00 is at (0,0)
    # B00 is at (0,0), B01 is at (0,1)
    # Contribution to C: C11 += M6
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1  # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    W[1*2+1][5] = 1  # C11 (k=1, i=1)
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # A01 is at (0,1), A11 is at (1,1)
    # B10 is at (1,0), B11 is at (1,1)
    # Contribution to C: C00 += M7
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1  # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    W[0*2+0][6] = 1  # C00 (k=0, i=0)
=======
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T[i,j,k] = 1 for the multiplication tensor.
    
    # Term r=0 (M1): U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    # This gives: 1*1*1 for (0,0,0) and (1,1,1), but also cross terms!
    # The issue: when U has two nonzeros, V has two nonzeros, W has two nonzeros,
    # the product expands to 8 terms, but we only want 2.
    # 
    # The correct way: each rank-1 term in the tensor decomposition is a single
    # outer product u ⊗ v ⊗ w where u, v, w are vectors.
    # So for M1, we need to decompose it into rank-1 tensors.
    # M1 contributes to C00 and C11, so we need:
    #   e_{0,0} ⊗ e_{0,0} ⊗ e_{0,0} + e_{1,1} ⊗ e_{1,1} ⊗ e_{1,1}
    # But that's 2 rank-1 terms, not 1!
    #
    # The key insight: in the tensor decomposition, each "term" r corresponds to
    # a single rank-1 tensor u_r ⊗ v_r ⊗ w_r. The Strassen algorithm uses 7
    # scalar multiplications, but each multiplication corresponds to a sum of
    # rank-1 tensors in the decomposition.
    #
    # Actually, I think I'm overcomplicating this. Let me reconsider.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C.
    #
    # A rank-r decomposition is: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
    #
    # For Strassen, the 7 products are:
    # M1 = (A00+A11)(B00+B11) = sum_{i,j,k} (A00+A11)_{?} ... 
    #
    # Let me think differently. The standard approach:
    # For 2x2x2, the tensor has 8 nonzero entries (all 1s).
    # Strassen shows it can be decomposed into 7 rank-1 tensors.
    #
    # The 7 rank-1 tensors are:
    # t1: u1=(1,0,1,0), v1=(1,0,1,0), w1=(1,0,1,0)  [for (0,0),(1,1) in each]
    # But this doesn't work because u1⊗v1⊗w1 has 4 nonzeros, not 2.
    #
    # I think the correct decomposition is:
    # Each Strassen product M_r corresponds to a rank-1 tensor where:
    # u_r[i,j] = coefficient of A[i,j] in the first factor
    # v_r[j,k] = coefficient of B[j,k] in the second factor  
    # w_r[k,i] = coefficient for C[k,i] ... no, w_r[i,k]
    #
    # Let me look up the actual Strassen tensor decomposition.
    #
    # For C = AB where A,B are 2x2:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, each M_r is a product of two linear combinations.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    #
    # So M1 contributes to:
    # C00: A00B00 term → u[0,0]=1, v[0,0]=1, w[0,0]=1
    # C00: A00B11 term → but this is not in C00! 
    #
    # Wait, I'm confusing myself. The point is that M1 is an intermediate
    # scalar value. It's not directly a tensor term.
    #
    # The correct way to get the tensor decomposition from Strassen:
    # Each M_r = (linear combo of A)(linear combo of B)
    # This M_r is then used in linear combinations to form C entries.
    #
    # The tensor decomposition has one rank-1 term per M_r:
    # For M_r = (sum_{i,j} a_{r,i,j} A_{i,j})(sum_{j,k} b_{r,j,k} B_{j,k}),
    # the rank-1 tensor is: u_r[i,j] = a_{r,i,j}, v_r[j,k] = b_{r,j,k},
    # w_r[i,k] = 1 for all (i,k) that M_r contributes to.
    #
    # No, that's not right either. Let me think again.
    #
    # Actually, the standard result is that the 2x2x2 matrix multiplication
    # tensor has rank 7, and the decomposition is:
    #
    # T = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r
    #
    # where the vectors are:
    # (Using indices (i,j) for A, (j,k) for B, (i,k) for C)
    #
    # r=1: u1[(0,0)]=1, u1[(1,1)]=1; v1[(0,0)]=1, v1[(1,1)]=1; w1[(0,0)]=1, w1[(1,1)]=1
    # This gives: T[0,0,0] += 1, T[0,1,0] += 0 (since v1[0,1]=0), etc.
    # Actually u1⊗v1⊗w1 has nonzeros at:
    # (0,0,0): 1*1*1 = 1
    # (0,0,1): 1*0*1 = 0
    # (0,1,0): 1*0*1 = 0  -- wait, v1 is indexed by (j,k)
    #
    # I think the indexing is: u[i,j], v[j,k], w[i,k]
    # So u1⊗v1⊗w1[i,j,k] = u1[i,j] * v1[j,k] * w1[i,k]
    #
    # For r=1: u1[0,0]=1, u1[1,1]=1; v1[0,0]=1, v1[1,1]=1; w1[0,0]=1, w1[1,1]=1
    # Nonzero terms:
    # i=0,j=0,k=0: u1[0,0]*v1[0,0]*w1[0,0] = 1*1*1 = 1
    # i=0,j=0,k=1: u1[0,0]*v1[0,1]*w1[0,1] = 1*0*0 = 0
    # i=0,j=1,k=0: u1[0,1]*v1[1,0]*w1[0,0] = 0*0*1 = 0
    # i=0,j=1,k=1: u1[0,1]*v1[1,1]*w1[0,1] = 0*1*0 = 0
    # i=1,j=0,k=0: u1[1,0]*v1[0,0]*w1[1,0] = 0*1*0 = 0
    # i=1,j=0,k=1: u1[1,0]*v1[0,1]*w1[1,1] = 0*0*1 = 0
    # i=1,j=1,k=0: u1[1,1]*v1[1,0]*w1[1,0] = 1*0*0 = 0
    # i=1,j=1,k=1: u1[1,1]*v1[1,1]*w1[1,1] = 1*1*1 = 1
    #
    # So r=1 contributes 1 to T[0,0,0] and T[1,1,1]. Good, that's part of what we want.
    #
    # But the full multiplication tensor has T[i,j,k]=1 for ALL 8 combinations.
    # We need the sum of all 7 terms to give 1 for each of the 8 (i,j,k) triples.
    #
    # Let me just hardcode the known 7-term decomposition.
    # After research, the standard Strassen tensor decomposition for 2x2x2 is:
    
    U[0][0] = 1; U[3][0] = 1    # u1: (0,0) and (1,1)
    V[0][0] = 1; V[3][0] = 1    # v1: (0,0) and (1,1)
    W[0][0] = 1; W[3][0] = 1    # w1: (0,0) and (1,1)
    
    U[2][1] = 1; U[3][1] = 1    # u2: (1,0) and (1,1)
    V[0][1] = 1                 # v2: (0,0)
    W[1][1] = 1; W[3][1] = -1   # w2: (1,0) and (1,1)
    
    U[0][2] = 1                 # u3: (0,0)
    V[2][2] = 1; V[3][2] = -1   # v3: (1,0) and (1,1)
    W[0][2] = 1; W[2][2] = 1    # w3: (0,0) and (0,1)
    
    U[3][3] = 1                 # u4: (1,1)
    V[1][3] = 1; V[0][3] = -1   # v4: (0,1) and (0,0)
    W[0][3] = 1; W[1][3] = 1    # w4: (0,0) and (1,0)
    
    U[0][4] = 1; U[1][4] = 1    # u5: (0,0) and (0,1)
    V[3][4] = 1                 # v5: (1,1)
    W[0][4] = -1; W[1][4] = 1   # w5: (0,0) and (0,1)
    
    U[2][5] = 1; U[0][5] = -1   # u6: (1,0) and (0,0)
    V[0][5] = 1; V[1][5] = 1    # v6: (0,0) and (0,1)
    W[3][5] = 1                 # w6: (1,1)
    
    U[1][6] = 1; U[3][6] = -1   # u7: (0,1) and (1,1)
    V[2][6] = 1; V[3][6] = 1    # v7: (1,0) and (1,1)
    W[0][6] = 1                 # w7: (0,0)
>>>>>>> REPLACE

I realize I need to be more careful. Let me just implement this cleanly from scratch with a verified approach.

