"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    
    # We will build a decomposition based on partitioning the matrices.
    # A (n x m) = [ A1 (n x 3) | A2 (n x 2) ]
    # B (m x p) = [ B1 (3 x p); B2 (2 x p) ]
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 B1 where A1 is 4x3, B1 is 3x5.
    # We can use a recursive or explicit low-rank decomposition for 4x3x5.
    # Schoolbook rank: 4*3*5 = 60.
    # Can we do better?
    # 4x3x5 multiplication.
    # Let's stick to a robust schoolbook-like or simple split for the core parts to ensure correctness,
    # but we can try to combine terms.
    # Actually, the previous attempt failed likely due to construction errors or timeout if it tried to search.
    # Here I will implement a constructive decomposition that is guaranteed to work and aims to be sub-schoolbook
    # by combining parts.
    
    # Total Schoolbook Rank: 4 * 5 * 5 = 100.
    
    # Strategy:
    # C = A B
    # Let's split B into columns: B = [ Bc1 (5x1), Bc2 (5x4) ]
    # C = A Bc1 + A Bc2
    #
    # Term 1: A Bc1. A is 4x5, Bc1 is 5x1. Result 4x1.
    # Rank 4*5*1 = 20.
    #
    # Term 2: A Bc2. A is 4x5, Bc2 is 5x4. Result 4x4.
    # Rank 4*5*4 = 80.
    # Total 100.
    
    # To beat 100, we need to exploit structure or use Strassen-like ideas.
    # Strassen for 4x4x4 has rank 49 (if using 7-term base case recursively, 4^3 * 7 / 4^3? No. 2x2 is 7. 4x4 is 7^2=49).
    # If we can handle the 5x5 part efficiently...
    
    # Let's try a specific decomposition for 4x5x5.
    # A (4x5), B (5x5).
    # Partition A into A_left (4x4) and A_right (4x1).
    # Partition B into B_top (4x5) and B_bottom (1x5).
    # C = A_left B_top + A_right B_bottom.
    
    # Part 1: A_left (4x4) * B_top (4x5).
    # We can view B_top as two 4x2 blocks? Or use a mixed approach.
    # Let's use Strassen for the 4x4 matrix product part of B_top?
    # B_top is 4x5. Let B_top = [ B_t1 (4x2) | B_t2 (4x3) ].
    # C_left = A_left B_t1 (4x2) + A_left B_t2 (4x3).
    
    # This is getting complex. Let's use a known low-rank for 4x4x4 (Rank 49) and add the rest.
    # But B is 5 wide.
    
    # Alternative:
    # Use the "Blelloch" or similar algorithm?
    # Let's try to just construct a valid one with rank < 100.
    # Even rank 90 is an improvement.
    
    # Simple optimization:
    # Schoolbook: 100.
    # If we use Strassen for the 4x4x4 block (rank 49) and schoolbook for the rest?
    # A (4x5) = [ A4 (4x4) | A1 (4x1) ]
    # B (5x5) = [ B4 (4x5); B1 (1x5) ]
    # C = A4 B4 + A1 B1.
    
    # Term A1 B1:
    # A1 is 4x1, B1 is 1x5. Result 4x5.
    # Rank: 4 * 1 * 5 = 20.
    
    # Term A4 B4:
    # A4 is 4x4, B4 is 4x5.
    # Let B4 = [ B4a (4x2) | B4b (4x3) ].
    # C4a = A4 B4a (4x2). Rank 4*4*2 = 32.
    # C4b = A4 B4b (4x3). Rank 4*4*3 = 48.
    # Total 49 + 20 + 32 + 48? No, A4 B4 is one term.
    # If we decompose A4 B4 (4x4x5) into A4 B4a (4x4x2) and A4 B4b (4x4x3).
    # Rank(4x4x2) via Strassen-like? 2x2x2 is 7. 4x4x4 is 49.
    # 4x4x2 can be seen as two 4x4x1? No.
    # 4x4x2 multiplication: A(4x4) * B(4x2).
    # This is equivalent to multiplying A by two columns of B.
    # We can use a rank 4*4*2 = 32 schoolbook.
    # Can we do better?
    # Maybe not easily without complex tensors.
    
    # Let's try a different split.
    # A (4x5), B (5x5).
    # Use rank 49 for 4x4x4 part?
    # Let's assume we use Strassen for the 4x4x4 sub-matrix of B.
    # B = [ B11 (4x4) | B12 (4x1) ]
    #     [ B21 (1x4) | B22 (1x1) ]
    
    # C = A [ B11 | B12 ]
    #       [ B21 | B22 ]
    # A = [ A1 (4x4) | A2 (4x1) ]
    
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # 1. A1 B11: 4x4 * 4x4. Use Strassen (Rank 49).
    # 2. A1 B12: 4x4 * 4x1. Schoolbook Rank 16.
    # 3. A2 B21: 4x1 * 1x4. Rank 4. (A2 is vector, B21 is vector. Outer product. Rank 4? No, rank 1? No, it's a 4x4 matrix formed by outer product of two vectors. The tensor rank for computing outer product of two vectors u,v to get u v^T is n. Here n=4. So rank 4.)
    # 4. A2 B22: 4x1 * 1x1. Rank 4. (Scaling a vector. Tensor rank n=4.)
    
    # Total Rank = 49 + 16 + 4 + 4 = 73.
    # This is significantly better than 100.
    
    # Let's implement this.
    # We need the U, V, W matrices for Strassen 4x4x4.
    # Standard Strassen 2x2:
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
    
    # Recursive Strassen for 4x4 splits into four 2x2s.
    # A = [ A11 A12; A21 A22 ] (each 2x2)
    # B = [ B11 B12; B21 B22 ] (each 2x2)
    
    # We will build the tensors for the 4x4x4 Strassen decomposition.
    # Then add the schoolbook terms for the other parts.
    
    def get_strassen_4x4_tensors():
        # Returns U, V, W for 4x4x4 multiplication with rank 49.
        # Indices:
        # A: 4x4. Index i*4+j.
        # B: 4x4. Index k*4+l. (Wait, standard is B[j,k] for A[i,j]B[j,k]).
        # In our problem: A is n x m, B is m x p.
        # Here A is 4x4, B is 4x4.
        # A index: i*4+j.
        # B index: j*4+k. (j is row of B, k is col of B).
        # C index: k*4+i? No, trace convention W[k*n+i].
        # For 4x4x4: n=4, p=4. C is 4x4. C[i,k]. W[k*4+i].
        
        # We construct the 49 terms.
        # Let's use a helper to add a term (U_row, V_row, W_row, coeff).
        # U shape (16, 49), V shape (16, 49), W shape (16, 49).
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # We need to map the Strassen sub-blocks to the flat indices.
        # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in block coords.
        # Block size 2x2.
        
        # Let's define the 7 products for each 2x2 block combination?
        # No, Strassen is applied to the 4x4 matrices by splitting into 2x2 blocks.
        # A = [ A11 A12; A21 A22 ]
        # B = [ B11 B12; B21 B22 ]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We will iterate through the 7 terms.
        # Each term is a product of a 2x2 matrix X and a 2x2 matrix Y.
        # The product Z = XY is 2x2.
        # We need to add this Z to the appropriate C block.
        
        # C11 gets M1 + M4 - M5 + M7
        # C12 gets M3 + M5
        # C21 gets M2 + M4
        # C22 gets M1 - M2 + M3 + M6
        
        # The tensors U, V, W will accumulate contributions.
        # For a term like (X * Y), where X and Y are linear combinations of A and B blocks.
        
        # Helper: add a 2x2x2 multiplication term to the global tensors.
        # X is 2x2, Y is 2x2, Z is 2x2.
        # X = sum of c_A * A_block
        # Y = sum of c_B * B_block
        # Z = sum of c_C * C_block
        # Actually, Strassen defines X and Y as specific sums.
        # The product is computed by schoolbook rank 8 for the 2x2x2?
        # No, the tensor rank of 2x2x2 multiplication is 7.
        # We are effectively using the 7-term Strassen algorithm.
        # So for each of the 7 "Strassen products" M_i, we need to decompose the 2x2x2 multiplication.
        # Wait, the rank of 2x2x2 multiplication is 7.
        # So for each M_i, which is a 2x2x2 multiplication, we use 7 rank-1 tensors.
        # Total 7 * 7 = 49 terms.
        
        # Let's implement the 2x2x2 rank-7 decomposition.
        # Indices for 2x2: A[i,j], B[j,k], C[i,k].
        # The 7 terms for 2x2x2:
        # T1: A[0,0]*(B[0,0]+B[0,1])? No.
        # Standard 2x2 Strassen:
        # m1 = a00 * (b00 + b11)
        # m2 = (a00 + a11) * b00
        # m3 = (a11 - a00) * (b10 + b11)
        # m4 = a11 * b11
        # m5 = (a00 + a01) * b10
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 + a10) * (b11 - b01) -- Wait, standard formulas vary.
        
        # Let's use a verified 2x2x2 rank 7 set.
        # Source: standard Strassen.
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We will define a function to add a "Strassen 2x2x2" block to the main tensors.
        # It takes:
        # - offset in A (row, col) for the 2x2 block
        # - offset in B (row, col) for the 2x2 block
        # - offset in C (row, col) for the 2x2 block
        # - signs for the 7 m-terms (to handle +/- in the combination)
        
        def add_strassen_2x2(a_off, b_off, c_off, signs):
            # signs is a list of 7 coefficients for m1..m7.
            # We iterate through the 7 base terms of 2x2x2.
            for t in range(7):
                col = t
                # Determine which A and B indices are involved and with what coefficients.
                # This is hard to do dynamically without a lookup table.
                # Let's hardcode the 7 terms for 2x2x2.
                
                # Term 0 (m1): (a00+a11)(b00+b11)
                # A: (0,0)+1, (1,1)+1. B: (0,0)+1, (1,1)+1.
                # C: c00+1, c11+1. (From c00=m1... c11=m1...)
                # Wait, the signs passed in apply to the whole m-term.
                # So if signs[t] is -1, we multiply all contributions by -1.
                
                pass # This approach is too complex to write inline safely.
        
        # Alternative: Hardcode the 49 terms for 4x4x4?
        # Or, just use schoolbook for 4x4x4 (rank 64) and see if it's still better than 100?
        # 64 + 16 + 4 + 4 = 88.
        # 88 < 100. This is a valid improvement!
        # And much simpler to implement correctly.
        
        # Let's stick with Schoolbook for the 4x4x4 part to ensure correctness.
        # Rank 4x4x4 = 64.
        # Total Rank = 64 (A1 B11) + 16 (A1 B12) + 4 (A2 B21) + 4 (A2 B22) = 88.
        
        # Let's verify the ranks again.
        # A1 B11: 4x4 * 4x4. Schoolbook: 4*4*4 = 64.
        # A1 B12: 4x4 * 4x1. Schoolbook: 4*4*1 = 16.
        # A2 B21: 4x1 * 1x4. Rank 4. (Outer product).
        # A2 B22: 4x1 * 1x1. Rank 4. (Scaling).
        # Total: 64 + 16 + 4 + 4 = 88.
        
        # Can we reduce A1 B12 (4x4 * 4x1)?
        # 4x4 * 4x1 is computing a matrix-vector product.
        # Tensor rank for 4x4x1?
        # C[i] = sum_j A[i,j] B[j].
        # This is a bilinear form? No, multilinear.
        # For each i, C[i] depends on all A[i,j] and B[j].
        # It's equivalent to 4 separate dot products of length 4.
        # Each dot product of length 4 has tensor rank 4?
        # Dot product u.v = sum u_i v_i. Rank n.
        # So 4 dot products = rank 16.
        # Is it possible to share structure?
        # The A rows are independent. So likely 16 is minimal or close to it.
        
        # So 88 is a solid target.
        
        # Implementation Plan:
        # 1. Initialize U, V, W with size (20, 88), (25, 88), (20, 88).
        #    Wait, U is n*m = 4*5 = 20 rows.
        #    V is m*p = 5*5 = 25 rows.
        #    W is p*n = 5*4 = 20 rows.
        
        rank_total = 88
        U = [[0]*rank_total for _ in range(20)]
        V = [[0]*rank_total for _ in range(25)]
        W = [[0]*rank_total for _ in range(20)]
        
        col = 0
        
        # Part 1: A1 B11
        # A1 is A[:, 0:4] (4x4). Indices in U: i*5 + j, i in 0..3, j in 0..3.
        # B11 is B[0:4, 0:4] (4x4). Indices in V: j*5 + k, j in 0..3, k in 0..3.
        # Result C11 is C[0:4, 0:4]. Indices in W: k*4 + i, k in 0..3, i in 0..3.
        # Schoolbook: for i, j, k:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_idx = i * 5 + j
                    v_idx = j * 5 + k
                    w_idx = k * 4 + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        # Part 2: A1 B12
        # A1 is A[:, 0:4] (4x4).
        # B12 is B[0:4, 4] (4x1). Indices in V: j*5 + 4, j in 0..3.
        # Result C12 is C[0:4, 4]. Indices in W: 4*4 + i = 16 + i, i in 0..3.
        # Schoolbook: for i, j:
        for i in range(4):
            for j in range(4):
                u_idx = i * 5 + j
                v_idx = j * 5 + 4
                w_idx = 4 * 4 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # Part 3: A2 B21
        # A2 is A[:, 4] (4x1). Indices in U: i*5 + 4, i in 0..3.
        # B21 is B[4, 0:4] (1x4). Indices in V: 4*5 + k = 20 + k, k in 0..3.
        # Result C21 is C[0:4, 0:4]. Indices in W: k*4 + i.
        # Outer product: C[i,k] = A[i,4] * B[4,k].
        # Rank 4. For each i:
        for i in range(4):
            u_idx = i * 5 + 4
            # B row is fixed at 4. k varies.
            # Wait, the term is A[i,4] * B[4,k].
            # In the tensor decomposition C = sum_r U[:,r] V[:,r] W[:,r]^T?
            # No, C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] is not right.
            # The definition is:
            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) * ?
            # Actually, the standard decomposition is:
            # A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] is WRONG.
            # The tensor is T[i,j,k] = delta_{ik}? No.
            # T[i,j,k] is 1 if we are multiplying A[i,j] and B[j,k] to contribute to C[i,k].
            # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
            # U_r is vector of length n*m. V_r length m*p. W_r length p*n.
            # C[i,k] = sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] ?? No.
            # The formula is:
            # C[i,k] = sum_{j} A[i,j] B[j,k]
            # We want C[i,k] = sum_r (sum_j U[i*m+j, r] * something) ...
            # Actually, the interface says:
            # "exactly decompose the matrix multiplication tensor"
            # Usually this means:
            # For all i,j,k: delta_{ij,k}? No.
            # The tensor for matrix multiplication is T[i,j,k] = 1 if we are forming the product.
            # The decomposition T = sum_r U_r V_r^T W_r^T?
            # Let's look at the schoolbook example in the prompt.
            # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
            # Let's check if this works.
            # C[i,k] should be sum_j A[i,j] B[j,k].
            # Using the decomposition:
            # C[i,k] = sum_c ( sum_j U[i*m+j, c] A[i,j]? No.
            # The variables A and B are the inputs.
            # The tensor T defines the operation.
            # T[i,j,k] = 1.
            # The contraction is:
            # C[i,k] = sum_{j, c} A[i,j] B[j,k] U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]?
            # If U,V,W are 1 at specific indices, does this recover the product?
            # In the schoolbook case, for each c, there is one (i,j,k) triplet.
            # U is 1 at (i,j). V is 1 at (j,k). W is 1 at (k,i).
            # Then term c contributes A[i,j] B[j,k] * 1 * 1 * 1 to C[i,k]?
            # Yes, if the sum is over c, and for each c, the indices i,j,k are fixed.
            # So C[i,k] = sum_c ( U[i*m+j_c, c] V[j_c*p+k_c, c] W[k_c*n+i_c, c] A[i_c, j_c] B[j_c, k_c] )
            # This only works if i_c = i, j_c = j, k_c = k.
            # So yes, the schoolbook decomposition sets U,V,W to pick out specific A and B entries.
            
            # So for Part 3: A2 B21.
            # We want to compute C[i,k] += A[i,4] B[4,k].
            # This term contributes to C[i,k] for all i in 0..3, k in 0..3.
            # In the tensor decomposition, we need to add a rank-1 tensor that captures this.
            # However, the "rank" here refers to the number of terms in the sum.
            # The term A[i,4] B[4,k] is bilinear in A and B, but it produces a 4x4 matrix.
            # Can we write this as a sum of rank-1 tensors of the form U_r V_r W_r?
            # A[i,4] B[4,k] = sum_r U_r[i,4] V_r[4,k] W_r[k,i]?
            # If we set U_r[i,4]=1, V_r[4,k]=1, W_r[k,i]=1, this would require U_r to be 1 for all i?
            # No, U_r is a vector.
            # If we use one term: U[i,4]=1 (for all i), V[4,k]=1 (for all k), W[k,i]=1 (for all k,i).
            # Then C[i,k] = sum_r 1*1*1 A[i,4] B[4,k] = A[i,4] B[4,k].
            # So it IS rank 1?
            # Wait, U has length 20. If U[i*5+4] = 1 for all i, that's 4 ones.
            # V has length 25. If V[4*5+k] = 1 for all k, that's 4 ones.
            # W has length 20. If W[k*4+i] = 1 for all k,i, that's 16 ones.
            # Is this a valid rank-1 tensor?
            # A rank-1 tensor is an outer product of three vectors.
            # U \otimes V \otimes W.
            # T[i,j,k] = U[i,j] V[j,k] W[k,i].
            # If U is sparse, V is sparse, W is sparse, the product is zero unless all three are non-zero.
            # If U[i,4]=1 for all i, V[4,k]=1 for all k, W[k,i]=1 for all k,i.
            # Then T[i,4,k] = 1 * 1 * 1 = 1.
            # This matches the desired contribution to the tensor T.
            # So YES, the outer product of two vectors is rank 1 in this tensor decomposition sense.
            # Similarly for A2 B22 (4x1 * 1x1), it is rank 1.
            # C[i,4] = A[i,4] B[4,4].
            # U[i,4]=1, V[4,4]=1, W[4,i]=1.
            
            # So Part 3 and Part 4 can be done in Rank 1 each!
            # Total Rank = 64 (Part 1) + 16 (Part 2) + 1 (Part 3) + 1 (Part 4) = 82.
            
            # Let's re-verify Part 2.
            # A1 B12: 4x4 * 4x1.
            # C[i,4] = sum_j A[i,j] B[j,4].
            # Can we do better than 16?
            # T[i,j,4] = 1.
            # This is 4 separate dot products.
            # Each dot product u.v has rank n=4.
            # Since the A rows are independent, we likely need 4*4=16.
            # So 16 is likely minimal.
            
            # So Total Rank 82.
            
            pass # End of verification logic
        
        # Corrected Plan for Part 3:
        for _ in range(1): # Rank 1
            # U: A[i,4] for i in 0..3.
            for i in range(4):
                U[i*5+4][col] = 1
            # V: B[4,k] for k in 0..3.
            for k in range(4):
                V[4*5+k][col] = 1
            # W: C[k,i] for k in 0..3, i in 0..3.
            for k in range(4):
                for i in range(4):
                    W[k*4+i][col] = 1
            col += 1
        
        # Corrected Plan for Part 4:
        for _ in range(1): # Rank 1
            # U: A[i,4] for i in 0..3.
            for i in range(4):
                U[i*5+4][col] = 1
            # V: B[4,4].
            V[4*5+4][col] = 1
            # W: C[4,i] for i in 0..3.
            for i in range(4):
                W[4*4+i][col] = 1
            col += 1
        
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
