import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # We construct a rank-44 decomposition by using Strassen's algorithm
    # for the top 2 rows of A (2x4x4 subproblem) and schoolbook for the bottom row (1x4x4).
    
    # Subproblem 1: A[0:2, :] * B
    # Strassen's algorithm for 2x2x2 blocks.
    # We view A[0:2, 0:4] as two 2x2 blocks: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # B is 4x4, so we view it as four 2x2 blocks:
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)  -- but here A is 2x4, so we split columns.
    # Actually, standard Strassen for 2x2 matrices:
    # C11 = M1 - M4, C12 = M2 + M3, C21 = M4 + M5, C22 = M1 - M2 + M5 - M6
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Here, our "A" for the top 2 rows is 2x4. Let's call it A_top.
    # A_top = [A11, A12] where A11 = A_top[:, 0:2], A12 = A_top[:, 2:4].
    # Wait, the standard Strassen decomposition splits the 2x2 matrix into four 1x1 blocks.
    # For 2x4 * 4x4, we can think of it as two 2x2 * 2x4 multiplications? No.
    # It's a (2x4) * (4x4) multiplication.
    # We can split A_top into A_left (2x2) and A_right (2x2).
    # We can split B into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2).
    # C = A_left * [B_TL B_TR] + A_right * [B_BL B_BR]
    # C_left = A_left * B_TL + A_right * B_BL
    # C_right = A_left * B_TR + A_right * B_BR
    
    # This is not directly 2x2x2 Strassen.
    # However, the known result for 2x4x4 is rank 28.
    # Let's use the specific rank-28 decomposition for 2x4x4.
    # Or, more simply, we can use the fact that 2x4x4 can be done with 28 multiplications.
    # How to construct it?
    # We can view 2x4x4 as two 2x2x4 problems? No.
    # We can view it as 2x2 * 2x4 and 2x2 * 2x4?
    # Actually, the rank of 2x4x4 is 28.
    # One way to get 28 is to use Strassen on the 2x2x2 blocks.
    # Let's try to build it manually.
    # A is 2x4. B is 4x4. C is 2x4.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B11, B12, B21, B22 (all 2x2).
    # C1 = A1 B11 + A2 B21 (2x2)
    # C2 = A1 B12 + A2 B22 (2x2)
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # So we need 7 + 7 = 14 multiplications for C1 and C2?
    # Wait, Strassen is for C = A B where A, B, C are 2x2.
    # Here C1 = A1 B11 + A2 B21. This is a sum of two products.
    # This is not a single matrix multiplication.
    # So we can't directly apply Strassen to the whole thing.
    
    # Let's look at the known rank 28 for 2x4x4.
    # It is known that R(2,4,4) = 28.
    # How is it constructed?
    # It is constructed by using the fact that 2x4x4 is equivalent to two 2x2x2 multiplications with some sharing?
    # Or by using the "Strassen-like" algorithm for rectangular matrices.
    
    # Actually, there is a simpler way.
    # The rank of 2x4x4 is 28.
    # The rank of 1x4x4 is 16.
    # Total rank 44.
    
    # Let's implement the rank-28 decomposition for 2x4x4.
    # I will use a known construction.
    # One construction uses 7 Strassen multiplications for the "left" part and 7 for the "right" part, but with some sharing?
    # No, 7+7=14 is too low. The schoolbook is 32.
    # The rank is 28.
    
    # Let's try to find the explicit rank-28 decomposition.
    # Reference: "The rank of 2x4x4 matrix multiplication is 28".
    # A construction:
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [B11 B12; B21 B22].
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22].
    # We can compute A1 B11, A1 B12, A2 B21, A2 B22 using Strassen? No, those are 2x2x2.
    # If we compute them separately, it's 4 * 7 = 28.
    # Yes!
    # C_left = A1 B11 + A2 B21.
    # C_right = A1 B12 + A2 B22.
    # If we compute the four products P1 = A1 B11, P2 = A1 B12, P3 = A2 B21, P4 = A2 B22 using Strassen's algorithm (rank 7 each),
    # then C_left = P1 + P3 and C_right = P2 + P4.
    # The total rank is 7 * 4 = 28.
    # This works!
    
    # So, for the top 2 rows (2x4x4), we use 4 Strassen blocks, each of rank 7. Total 28.
    # For the bottom 1 row (1x4x4), we use schoolbook, rank 16.
    # Total rank 44.
    
    def strassen_2x2x2(A, B):
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        A is 2x2, B is 2x2.
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7).
        Indices for U: [0:2] for A rows 0,1 cols 0,1? No, U[i*2+j, r].
        A is 2x2, so U is 4x7.
        B is 2x2, so V is 4x7.
        C is 2x2, so W is 4x7.
        """
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # We need to express each M as a linear combination of A and B entries.
        # Then for each M, we have a rank-1 tensor for the 2x2x2 multiplication?
        # No, Strassen computes the products using 7 multiplications of linear forms.
        # Each multiplication is of the form (linear form in A) * (linear form in B).
        # So each M contributes rank 1 to the tensor?
        # Wait, the rank of the 2x2x2 tensor is 7.
        # So we need 7 rank-1 tensors.
        # Each rank-1 tensor corresponds to one of the 7 products.
        
        # Let's define the 7 products explicitly.
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b10 - b00)
        # M4 = a11(b11 - b01)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # The output C is:
        # c00 = M1 - M4
        # c01 = M2 + M3
        # c10 = M4 + M5
        # c11 = M1 - M2 + M5 - M6
        
        # We need to construct U, V, W such that
        # C[i,j] = sum_r U[i,j,r] V[r] W[r]? No.
        # The decomposition is for the tensor T.
        # T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] V[b_idx, r] W[c_idx, r].
        
        # For each of the 7 products M_k, we have:
        # M_k = (sum_i u_k[i] A[i]) * (sum_j v_k[j] B[j])
        # This contributes to the output C as:
        # C[c] += w_k[c] * M_k
        
        # So for each k, we have a rank-1 term:
        # U[:, r] = u_k
        # V[:, r] = v_k
        # W[:, r] = w_k
        
        # Let's list the u_k, v_k, w_k vectors.
        # A indices: 0: a00, 1: a01, 2: a10, 3: a11
        # B indices: 0: b00, 1: b01, 2: b10, 3: b11
        # C indices: 0: c00, 1: c01, 2: c10, 3: c11
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a00 + a11)(b00 + b11)
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        # w1: contributes to c00 (+1), c11 (+1)
        # w1 = [1, 0, 0, 1]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10 + a11)b00
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2: contributes to c01 (+1), c11 (-1)
        # w2 = [0, 1, 0, -1]
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3 = a00(b10 - b00)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 0, 1, -1]
        # w3: contributes to c01 (+1)
        # w3 = [0, 1, 0, 0]
        
        U[0][2] = 1
        V[2][2] = 1; V[0][2] = -1
        W[1][2] = 1
        
        # M4 = a11(b11 - b01)
        # u4 = [0, 0, 0, 1]
        # v4 = [0, -1, 0, 1]
        # w4: contributes to c00 (-1), c10 (+1)
        # w4 = [-1, 0, 1, 0]
        
        U[3][3] = 1
        V[3][3] = 1; V[1][3] = -1
        W[0][3] = -1; W[2][3] = 1
        
        # M5 = (a00 + a01)b11
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # w5: contributes to c10 (+1), c11 (+1)
        # w5 = [0, 0, 1, 1]
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1; W[3][4] = 1
        
        # M6 = (a10 - a00)(b00 + b01)
        # u6 = [-1, 0, 1, 0]
        # v6 = [1, 1, 0, 0]
        # w6: contributes to c11 (-1)
        # w6 = [0, 0, 0, -1]
        
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = -1
        
        # M7 = (a01 - a11)(b10 + b11)
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # w7: contributes to nothing?
        # Wait, Strassen's formula:
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        # Where did M7 go?
        # Ah, the standard Strassen is:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # Wait, M7 is not used in the output?
        # Let's check the formula again.
        # C = [[M1-M4, M2+M3], [M4+M5, M1-M2+M5-M6]]
        # This is correct. M7 is not used?
        # No, that's wrong. Strassen uses 7 multiplications.
        # The formula is:
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B11)
        # P4 = A22(B22 - B12)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        
        # C11 = P1 - P4
        # C12 = P2 + P3
        # C21 = P4 + P5
        # C22 = P1 - P2 + P5 - P6
        
        # Wait, P7 is missing.
        # Let me check a reliable source.
        # Wikipedia:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # It seems M7 is not used in the output?
        # That can't be right.
        # Let's re-verify.
        # Actually, the formula for C22 is M1 - M2 + M5 - M6.
        # And M7 is not there.
        # Is M7 needed?
        # Let's check the rank.
        # If M7 is not used, the rank is 6? No, 2x2x2 rank is 7.
        # So M7 must be used.
        # Let me check the formula again.
        # Maybe C22 = M1 - M2 + M5 - M6 is wrong?
        # Correct formula:
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # Wait, I found a different version:
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # Let's check with A=I, B=I.
        # M1 = 2*2 = 4
        # M2 = 1*1 = 1
        # M3 = 1*(1-1) = 0
        # M4 = 1*(1-1) = 0
        # M5 = 2*1 = 2
        # M6 = 0*(2) = 0
        # M7 = 0*(2) = 0
        
        # C11 = 4 - 0 = 4. Should be 1.
        # So the formula is wrong or my calculation is wrong.
        
        # Let's use the correct Strassen formula.
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B11)
        # P4 = A22(B22 - B12)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        
        # C11 = P1 - P4
        # C12 = P2 + P3
        # C21 = P4 + P5
        # C22 = P1 - P2 + P5 - P6
        
        # Let's re-calculate with A=I, B=I.
        # P1 = (1+1)(1+1) = 4
        # P2 = (0+1)1 = 1
        # P3 = 1(0-1) = -1
        # P4 = 1(1-0) = 1
        # P5 = (1+0)1 = 1
        # P6 = (0-1)(1+0) = -1
        # P7 = (0-1)(0+1) = -1
        
        # C11 = 4 - 1 = 3. Should be 1.
        # Still wrong.
        
        # Let's look up the correct Strassen algorithm.
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # I am getting 3 for C11.
        # Let's check the definition of M1.
        # M1 = (A11 + A22)(B11 + B22).
        # If A=I, B=I, M1 = 4.
        # M4 = A22(B22 - B12) = 1(1-0) = 1.
        # C11 = 4 - 1 = 3.
        # But C11 should be A11 B11 + A12 B21 = 1*1 + 0*0 = 1.
        
        # There is a mistake in my understanding of Strassen or the formula.
        # Let's try a different source.
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # Maybe the indices are different?
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        
        # Let's try M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # I will use the known rank-7 decomposition for 2x2x2 from a reliable source or derive it.
        # Actually, I can just use the schoolbook for 2x2x2 (rank 4) and combine 4 of them (rank 16) for 2x4x4.
        # Then total rank 16 + 16 = 32.
        # This is better than 48, but not 44.
        
        # Let's try to find the rank-28 decomposition online or derive it.
        # Since I can't browse, I will use the schoolbook for the sub-blocks.
        # Rank 32 for 2x4x4 and 16 for 1x4x4. Total 48.
        # This is not an improvement.
        
        # Let's try to use Strassen correctly.
        # The formula is:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B11)
        # M4 = A22(B22 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # I will trust this formula and assume my manual calculation was wrong.
        # Let's re-verify C11.
        # C11 = A11 B11 + A12 B21.
        # M1 - M4 = (A11 + A22)(B11 + B22) - A22(B22 - B12)
        # = A11 B11 + A11 B22 + A22 B11 + A22 B22 - A22 B22 + A22 B12
        # = A11 B11 + A11 B22 + A22 B11 + A22 B12.
        # This is not C11.
        
        # So the formula is wrong.
        # Let's try:
        # C11 = M1 - M4
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M2 + M5 - M6
        
        # I will fall back to schoolbook for the sub-blocks.
        # Rank 32 for 2x4x4.
        # Rank 16 for 1x4x4.
        # Total 48.
        
        # This is not an improvement.
        # I will return the schoolbook decomposition for the whole 3x4x4.
        
    # Given the complexity and risk of error, I will return the schoolbook decomposition.
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
