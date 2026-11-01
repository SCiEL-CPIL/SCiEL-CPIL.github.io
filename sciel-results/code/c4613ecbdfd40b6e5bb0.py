"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    # We want to multiply a 2x5 matrix A by a 5x6 matrix B to get a 2x6 matrix C.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can decompose this into two parts:
    # 1. The first 4 columns of B (5x4) -> multiply A (2x5) by B1 (5x4) to get C1 (2x4)
    # 2. The last 2 columns of B (5x2) -> multiply A (2x5) by B2 (5x2) to get C2 (2x2)
    #
    # For part 1: 2x5x4 multiplication
    #   We can further decompose: A is 2x5, B1 is 5x4
    #   Split A into A1 (2x4) and A2 (2x1), and B1 into B11 (4x4) and B12 (1x4)
    #   Actually, let's think differently.
    #
    # Alternative approach: Use Strassen-like decomposition for the whole thing.
    #
    # Let's use a known result: The rank of 2x5x6 matrix multiplication.
    # 
    # A simple approach: 
    # - Use schoolbook for 2x3x6 part: rank = 2*3*6 = 36
    # - Use Strassen for 2x2x6 part: 
    #   For 2x2x6, we can treat it as 3 independent 2x2x2 blocks, each with Strassen rank 7.
    #   So 3 * 7 = 21.
    #   Total: 36 + 21 = 57.
    #
    # But we can do better by not splitting the 5 into 3+2 independently.
    #
    # Better: Split the 5 into 2+2+1.
    # - 2x2x6: 3 Strassen blocks = 3*7 = 21
    # - 2x2x6: 3 Strassen blocks = 3*7 = 21  
    # - 2x1x6: schoolbook = 2*1*6 = 12
    # Total: 21 + 21 + 12 = 54.
    #
    # Even better: Split 5 into 4+1.
    # - 2x4x6: We can split the 4 into 2+2, giving two 2x2x6 blocks.
    #   Each 2x2x6 uses 3 Strassen 2x2x2 blocks = 21.
    #   So 2x4x6 = 21 + 21 = 42.
    # - 2x1x6: schoolbook = 12.
    # Total: 42 + 12 = 54.
    #
    # Let's try to do even better. What if we use a smarter decomposition?
    #
    # Actually, let me reconsider. For 2x2x2 Strassen:
    #   C = A*B where A, B, C are 2x2.
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    #
    # For 2x2x6: A is 2x2, B is 2x6, C is 2x6.
    #   We can write B as [B_left, B_right] where each is 2x3.
    #   Actually, let's split B into 3 blocks of 2x2: B = [B1, B2, B3]
    #   Then C = [C1, C2, C3] where Ci = A*Bi, each Ci is 2x2.
    #   Each Ci uses 7 Strassen multiplications.
    #   But we can share some of the Strassen intermediate products!
    #   
    #   The Strassen intermediates for A*B1, A*B2, A*B3:
    #   M1_k = (A00+A11)(B1_k00 + B1_k11) for k=1,2,3
    #   M2_k = (A10+A11)B1_k00 for k=1,2,3
    #   M3_k = A00(B1_k10 - B1_k11) for k=1,2,3
    #   M4_k = A11(B1_k01 - B1_k00) for k=1,2,3
    #   M5_k = (A00+A01)B1_k11 for k=1,2,3
    #   M6_k = (A10-A00)(B1_k00 + B1_k10) for k=1,2,3
    #   M7_k = (A01-A11)(B1_k01 + B1_k11) for k=1,2,3
    #   
    #   These are all independent because they involve different B blocks.
    #   So 2x2x6 requires 3*7 = 21 multiplications. No sharing possible.
    #
    # So the best I can think of is 54. Let me verify this is correct.
    #
    # Actually, wait. Let me think about whether we can do better with a different split.
    #
    # What about using the fact that 2x5x6 = 2*(5x6)? No, that doesn't help directly.
    #
    # Let me just implement the 54-rank decomposition.
    
    n, m, p = 2, 5, 6
    
    # We'll build the U, V, W matrices.
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (30, rank)  
    # W has shape (p*n, rank) = (12, rank)
    #
    # The decomposition: C[i,k] = sum_r U[i*m+j... wait, let me re-read the convention.
    #
    # From the task: "The output factor follows the official trace convention W[k*n+i,r]"
    # And U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    #
    # So the matrix multiplication tensor is:
    # T[i,j,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    #
    # Wait, that's not right either. Let me think about this more carefully.
    #
    # Standard tensor decomposition for matrix multiplication:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T[i,j,k] = delta_{i,k} ... no.
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    #
    # Actually, the matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if k corresponds to the output position, and i,j are the input positions.
    # 
    # In the decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    #
    # So U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This means: for each rank term r, we have:
    #   U selects which (i,j) pair, V selects which (j,k) pair, W selects which (k,i) pair.
    #   The product U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] contributes to T[i,j,k].
    #
    # For the schoolbook decomposition, each term r corresponds to a unique (i,j,k) triple,
    # and U[i*m+j,r] = V[j*p+k,r] = W[k*n+i,r] = 1 for that triple.
    #
    # For Strassen, the terms are more complex. Let me work out the Strassen 2x2x2 case.
    #
    # For 2x2x2: n=m=p=2.
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    #
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    #   This contributes to C00 and C11.
    #   U[0,0]=1, U[3,0]=1 (A00 and A11)
    #   V[0,0]=1, V[3,0]=1 (B00 and B11)
    #   W[0,0]=1 (C00), W[3,0]=1 (C11)
    #   Wait, that's not right. The Strassen intermediate is a single scalar product.
    #   M1 = (A00+A11) * (B00+B11)
    #   This is a product of two linear combinations.
    #   In the tensor decomposition, this corresponds to a term where:
    #   U[0*m+0, r] = 1 (A00), U[1*m+1, r] = 1 (A11)
    #   V[0*p+0, r] = 1 (B00), V[1*p+1, r] = 1 (B11)
    #   But then the product U[i*m+j,r]*V[j*p+k,r] would be 1 for (i=0,j=0,k=0), (i=0,j=0,k=1), etc.
    #   That's not right. The Strassen term is a single multiplication, not a sum.
    #
    # I think I'm confusing myself. Let me look at this differently.
    #
    # The tensor T[i,j,k] for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] and the term involves A[i,j] and B[j,k].
    # Actually, T[i,j,k] = 1 for all valid (i,j,k) in the sense that each (i,j,k) 
    # contributes A[i,j]*B[j,k] to C[i,k].
    #
    # No wait, that's the schoolbook. The tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1.
    # No, that's not right either.
    #
    # Let me be precise. The matrix multiplication tensor T is a 3-tensor of size n x m x p
    # such that for any matrices A (n x m) and B (m x p):
    # C = A*B where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This can be written as: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    #
    # Wait, that's just saying C[i,k] = sum_j A[i,j] B[j,k], which is trivial.
    #
    # I think the tensor is defined differently. Let me reconsider.
    #
    # The matrix multiplication tensor is often defined as:
    # T[i,j,k] = 1 if the indices are consistent, i.e., T is the tensor such that
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k].
    #
    # In this case, T[i,j,k] = 1 for all i,j,k, and the contraction is:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    #
    # So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # The rank of T is the minimum number of rank-1 tensors needed to sum to T.
    # A rank-1 tensor is U ⊗ V ⊗ W where U is a vector of length n*m, V is a vector of length m*p,
    # W is a vector of length p*n.
    #
    # So T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    #
    # And T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    #
    # For the schoolbook, each term r corresponds to a unique (i,j,k), and
    # U_r[i*m+j] = 1, V_r[j*p+k] = 1, W_r[k*n+i] = 1, and 0 elsewhere.
    #
    # For Strassen 2x2x2:
    # T[i,j,k] = sum_r U_r[i*2+j] * V_r[j*2+k] * W_r[k*2+i]
    #
    # The 7 Strassen terms:
    # r=0: M1 = (A00+A11)(B00+B11)
    #   U_0 = [1, 0, 0, 1] (A00 and A11)
    #   V_0 = [1, 0, 0, 1] (B00 and B11)
    #   W_0 = [1, 0, 0, 1] (C00 and C11)
    #   Check: T[0,0,0] += U_0[0]*V_0[0]*W_0[0] = 1*1*1 = 1
    #   T[0,0,1] += U_0[0]*V_0[1]*W_0[1] = 1*0*0 = 0
    #   T[0,1,0] += U_0[1]*V_0[0]*W_0[0] = 0*1*1 = 0
    #   T[0,1,1] += U_0[1]*V_0[1]*W_0[1] = 0*0*0 = 0
    #   T[1,0,0] += U_0[2]*V_0[0]*W_0[0] = 0*1*1 = 0
    #   T[1,0,1] += U_0[2]*V_0[1]*W_0[1] = 0*0*0 = 0
    #   T[1,1,0] += U_0[3]*V_0[2]*W_0[0] = 1*0*1 = 0
    #   T[1,1,1] += U_0[3]*V_0[3]*W_0[3] = 1*1*1 = 1
    #   So this term contributes 1 to T[0,0,0] and T[1,1,1].
    #
    # r=1: M2 = (A10+A11)B00
    #   U_1 = [0, 0, 1, 1] (A10 and A11)
    #   V_1 = [1, 0, 0, 0] (B00)
    #   W_1 = [0, 0, 0, 1] (C11)
    #   Check: T[1,0,1] += U_1[2]*V_1[0]*W_1[3] = 1*1*1 = 1
    #   T[1,1,1] += U_1[3]*V_1[3]*W_1[3] = 1*0*1 = 0
    #   Hmm, that's not right. Let me re-examine.
    #
    # I think I need to be more careful. Let me look up the Strassen decomposition in terms of the tensor.
    #
    # Actually, I realize that implementing Strassen correctly in this tensor form is tricky.
    # Let me just go with the 54-rank schoolbook-like decomposition that I described earlier.
    # It's not optimal, but it's a valid improvement over 60.
    
    # Let me implement the 54-rank decomposition:
    # Split the 5 into 4+1.
    # For the 2x4x6 part: split the 4 into 2+2.
    #   For each 2x2x6: use 3 Strassen 2x2x2 blocks (rank 21 each).
    # For the 2x1x6 part: schoolbook (rank 12).
    # Total: 21 + 21 + 12 = 54.
    
    # But I need to implement Strassen correctly. Let me define the Strassen 2x2x2 terms.
    
    # Strassen 2x2x2: n=m=p=2
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to C00 and C11
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # M2 = (A10+A11)B00
        # Contributes to C10
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1               # B00
        W[2][1] = 1               # C10
        
        # M3 = A00(B10-B11)
        # Contributes to C01
        U[0][2] = 1               # A00
        V[2][2] = 1; V[3][2] = -1 # B10, -B11
        W[1][2] = 1               # C01
        
        # M4 = A11(B01-B00)
        # Contributes to C10 and C11
        U[3][3] = 1               # A11
        V[1][3] = 1; V[0][3] = -1 # B01, -B00
        W[2][3] = 1; W[3][3] = 1  # C10, C11
        
        # M5 = (A00+A01)B11
        # Contributes to C00 and C01
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1               # B11
        W[0][4] = 1; W[1][4] = 1  # C00, C01
        
        # M6 = (A10-A00)(B00+B10)
        # Contributes to C11
        U[2][5] = 1; U[0][5] = -1 # A10, -A00
        V[0][5] = 1; V[2][5] = 1  # B00, B10
        W[3][5] = 1               # C11
        
        # M7 = (A01-A11)(B01+B11)
        # Contributes to C00
        U[1][6] = 1; U[3][6] = -1 # A01, -A11
        V[1][6] = 1; V[3][6] = 1  # B01, B11
        W[0][6] = 1               # C00
        
        return U, V, W
    
    # Let me verify this Strassen decomposition by checking that the sum of all terms
    # gives T[i,j,k] = 1 for all i,j,k.
    
    U_s, V_s, W_s = strassen_2x2x2()
    # Check T[0,0,0]
    total = 0
    for r in range(7):
        total += U_s[0*2+0][r] * V_s[0*2+0][r] * W_s[0*2+0][r]
    # Should be 1
    
    # This is getting complex. Let me just trust the Strassen decomposition and move on.
    # I'll implement the 54-rank decomposition.
    
    # Actually, let me just go with a simpler approach: use schoolbook for everything.
    # The schoolbook rank is 2*5*6 = 60.
    # That's the baseline.
    
    # For an improvement, let me try the 54-rank approach, but I need to be careful
    # with the Strassen implementation.
    
    # Let me just return the schoolbook for now, since I'm not confident in the Strassen
    # tensor decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
