"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Trivial rank-48 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    # Strategy: Correct Strassen-like decomposition for 2x4 * 4x6.
    # We partition A (2x4) into four 1x2 blocks: A11, A12, A21, A22.
    # We partition B (4x6) into four 2x3 blocks: B11, B12, B13, B14.
    # 
    # A = [A11 A12]
    #     [A21 A22]
    # B = [B11 B12]
    #     [B13 B14]
    #
    # C = [C11 C12]
    #     [C21 C22]
    #
    # C11 = A11 B11 + A12 B13
    # C12 = A11 B12 + A12 B14
    # C21 = A21 B11 + A22 B13
    # C22 = A21 B12 + A22 B14
    #
    # Strassen terms (7 multiplications of 1x2 by 2x3 matrices):
    # M1 = (A11 + A22) (B11 + B14)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B14)
    # M4 = A22 (B13 - B11)
    # M5 = (A11 + A12) B14
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B13 + B14)
    #
    # Reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a product of a 1x2 matrix and a 2x3 matrix.
    # The rank of the tensor for 1x2 * 2x3 multiplication is 6 (trivial).
    # So total rank is 7 * 6 = 42.
    
    rank_total = 42
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Define the 7 terms.
    # Each term t has:
    # - A_t: a 2x4 matrix (sparse) representing the linear combination of A.
    # - B_t: a 4x6 matrix (sparse) representing the linear combination of B.
    # - signs: a 2x6 matrix (sparse) representing how M_t contributes to C.
    #   C[i, k] += signs[i, k] * (A_t * B_t)[i, k]
    
    terms = []
    
    def zero_A():
        return np.zeros((2, 4))
    
    def zero_B():
        return np.zeros((4, 6))
    
    def zero_C():
        return np.zeros((2, 6))
    
    # T1: M1 = (A11 + A22) (B11 + B14)
    A1 = zero_A()
    A1[0, 0] = 1; A1[0, 1] = 1  # A11
    A1[1, 2] = 1; A1[1, 3] = 1  # A22
    B1 = zero_B()
    B1[0, 0:3] = 1; B1[1, 0:3] = 1  # B11
    B1[2, 3:6] = 1; B1[3, 3:6] = 1  # B14
    C1 = zero_C()
    C1[0, 0:3] = 1   # C11 += M1
    C1[1, 3:6] = 1   # C22 += M1
    terms.append((A1, B1, C1))
    
    # T2: M2 = (A21 + A22) B11
    A2 = zero_A()
    A2[1, 0] = 1; A2[1, 1] = 1  # A21
    A2[1, 2] = 1; A2[1, 3] = 1  # A22
    B2 = zero_B()
    B2[0, 0:3] = 1; B2[1, 0:3] = 1  # B11
    C2 = zero_C()
    C2[1, 0:3] = 1   # C21 += M2
    C2[1, 3:6] = -1  # C22 -= M2
    terms.append((A2, B2, C2))
    
    # T3: M3 = A11 (B12 - B14)
    A3 = zero_A()
    A3[0, 0] = 1; A3[0, 1] = 1  # A11
    B3 = zero_B()
    B3[0, 3:6] = 1; B3[1, 3:6] = 1  # B12
    B3[2, 3:6] = -1; B3[3, 3:6] = -1 # -B14
    C3 = zero_C()
    C3[0, 3:6] = 1   # C12 += M3
    C3[1, 3:6] = 1   # C22 += M3
    terms.append((A3, B3, C3))
    
    # T4: M4 = A22 (B13 - B11)
    A4 = zero_A()
    A4[1, 2] = 1; A4[1, 3] = 1  # A22
    B4 = zero_B()
    B4[2, 0:3] = 1; B4[3, 0:3] = 1  # B13
    B4[0, 0:3] = -1; B4[1, 0:3] = -1 # -B11
    C4 = zero_C()
    C4[0, 0:3] = 1   # C11 += M4
    C4[1, 0:3] = 1   # C21 += M4
    terms.append((A4, B4, C4))
    
    # T5: M5 = (A11 + A12) B14
    A5 = zero_A()
    A5[0, 0] = 1; A5[0, 1] = 1  # A11
    A5[0, 2] = 1; A5[0, 3] = 1  # A12
    B5 = zero_B()
    B5[2, 3:6] = 1; B5[3, 3:6] = 1  # B14
    C5 = zero_C()
    C5[0, 0:3] = -1  # C11 -= M5
    C5[0, 3:6] = 1   # C12 += M5
    terms.append((A5, B5, C5))
    
    # T6: M6 = (A21 - A11) (B11 + B12)
    A6 = zero_A()
    A6[1, 0] = 1; A6[1, 1] = 1  # A21
    A6[0, 0] = -1; A6[0, 1] = -1 # -A11
    B6 = zero_B()
    B6[0, 0:6] = 1; B6[1, 0:6] = 1  # B11 + B12
    C6 = zero_C()
    C6[1, 3:6] = 1   # C22 += M6
    terms.append((A6, B6, C6))
    
    # T7: M7 = (A12 - A22) (B13 + B14)
    A7 = zero_A()
    A7[0, 2] = 1; A7[0, 3] = 1  # A12
    A7[1, 2] = -1; A7[1, 3] = -1 # -A22
    B7 = zero_B()
    B7[2, 0:6] = 1; B7[3, 0:6] = 1  # B13 + B14
    C7 = zero_C()
    C7[0, 0:3] = 1   # C11 += M7
    terms.append((A7, B7, C7))
    
    # Now, construct U, V, W for each term.
    # For term t, we decompose the product A_t * B_t into rank-1 tensors.
    # The product is a 2x6 matrix.
    # We iterate over all (i, j, k) such that A_t[i, j] != 0 and B_t[j, k] != 0.
    # Each such triplet defines a rank-1 tensor contribution to the term t.
    # The rank index r is global.
    
    rank_idx = 0
    for t, (A_t, B_t, C_t) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                if A_t[i, j] != 0:
                    for k in range(p):
                        if B_t[j, k] != 0:
                            # Contribution to C[i, k] is A_t[i, j] * B_t[j, k] * C_t[i, k]
                            # This corresponds to a rank-1 tensor:
                            # U[i, r] = A_t[i, j]
                            # V[j, r] = B_t[j, k]
                            # W[k, r] = C_t[i, k]
                            # Note: The standard decomposition is sum_r U[i,r] V[j,r] W[k,r] = C[i,k]
                            # Here, for a fixed (i,j,k), the contribution to C[i,k] from term t is:
                            # A_t[i,j] * B_t[j,k] * C_t[i,k]
                            # So we set:
                            U[i * m + j, rank_idx] = A_t[i, j]
                            V[j * p + k, rank_idx] = B_t[j, k]
                            W[k * n + i, rank_idx] = C_t[i, k]
                            rank_idx += 1
                            
    # The total rank used is rank_idx.
    # Let's verify it is 42.
    # T1: A1 non-zeros: (0,0), (0,1), (1,2), (1,3). B1 non-zeros: rows 0,1 cols 0-2; rows 2,3 cols 3-5.
    # Matches:
    # (0,0) with B1[0, 0:3] -> 3 terms
    # (0,1) with B1[1, 0:3] -> 3 terms
    # (1,2) with B1[2, 3:6] -> 3 terms
    # (1,3) with B1[3, 3:6] -> 3 terms
    # Total 12 terms for T1.
    
    # T2: A2 non-zeros: (1,0), (1,1), (1,2), (1,3). B2 non-zeros: rows 0,1 cols 0-2.
    # Matches:
    # (1,0) with B2[0, 0:3] -> 3 terms
    # (1,1) with B2[1, 0:3] -> 3 terms
    # (1,2) with B2[2, ...] -> 0 (B2[2,:] is 0)
    # (1,3) with B2[3, ...] -> 0
    # Total 6 terms for T2.
    
    # T3: A3 non-zeros: (0,0), (0,1). B3 non-zeros: rows 0,1 cols 3-5; rows 2,3 cols 3-5.
    # Matches:
    # (0,0) with B3[0, 3:6] -> 3 terms
    # (0,1) with B3[1, 3:6] -> 3 terms
    # Total 6 terms for T3.
    
    # T4: A4 non-zeros: (1,2), (1,3). B4 non-zeros: rows 2,3 cols 0-2; rows 0,1 cols 0-2.
    # Matches:
    # (1,2) with B4[2, 0:3] -> 3 terms
    # (1,3) with B4[3, 0:3] -> 3 terms
    # Total 6 terms for T4.
    
    # T5: A5 non-zeros: (0,0), (0,1), (0,2), (0,3). B5 non-zeros: rows 2,3 cols 3-5.
    # Matches:
    # (0,0) with B5[0, ...] -> 0
    # (0,1) with B5[1, ...] -> 0
    # (0,2) with B5[2, 3:6] -> 3 terms
    # (0,3) with B5[3, 3:6] -> 3 terms
    # Total 6 terms for T5.
    
    # T6: A6 non-zeros: (1,0), (1,1), (0,0), (0,1). B6 non-zeros: rows 0,1 cols 0-5.
    # Matches:
    # (1,0) with B6[0, 0:6] -> 6 terms
    # (1,1) with B6[1, 0:6] -> 6 terms
    # (0,0) with B6[0, 0:6] -> 6 terms
    # (0,1) with B6[1, 0:6] -> 6 terms
    # Total 24 terms for T6.
    
    # T7: A7 non-zeros: (0,2), (0,3), (1,2), (1,3). B7 non-zeros: rows 2,3 cols 0-5.
    # Matches:
    # (0,2) with B7[2, 0:6] -> 6 terms
    # (0,3) with B7[3, 0:6] -> 6 terms
    # (1,2) with B7[2, 0:6] -> 6 terms
    # (1,3) with B7[3, 0:6] -> 6 terms
    # Total 24 terms for T7.
    
    # Total rank = 12 + 6 + 6 + 6 + 6 + 24 + 24 = 84.
    # Wait, my previous analysis said 42. Why the discrepancy?
    # Ah, in Strassen, the "multiplication" is of the *blocks*.
    # If the blocks were scalars, M1 would be 1 multiplication.
    # But the blocks are 1x2 and 2x3 matrices.
    # The product of a 1x2 and a 2x3 matrix has rank 6 (trivial).
    # So M1 requires 6 rank-1 tensors.
    # M2 requires 6 rank-1 tensors.
    # ...
    # M7 requires 6 rank-1 tensors.
    # Total 7 * 6 = 42.
    
    # Why did my explicit count give 84?
    # Let's re-evaluate T1.
    # A1 = [1 1 0 0; 0 0 1 1]
    # B1 = [1 1 1 0 0 0; 1 1 1 0 0 0; 0 0 0 1 1 1; 0 0 0 1 1 1]
    # Product P = A1 * B1
    # P[0, 0] = 1*1 + 1*1 = 2
    # P[0, 1] = 1*1 + 1*1 = 2
    # P[0, 2] = 1*1 + 1*1 = 2
    # P[0, 3] = 0
    # ...
    # P[1, 0] = 0
    # P[1, 3] = 1*1 + 1*1 = 2
    # ...
    # The rank of the *operation* of multiplying these specific sparse matrices is not simply the number of non-zero triplets.
    # The rank of the tensor for 1x2 * 2x3 is 6.
    # This means we can compute the 6 entries of the 1x3 result using 6 multiplications.
    # The 6 entries are:
    # c0 = a0 b00 + a1 b10
    # c1 = a0 b01 + a1 b11
    # c2 = a0 b02 + a1 b12
    # c3 = a0 b03 + a1 b13
    # c4 = a0 b04 + a1 b14
    # c5 = a0 b05 + a1 b15
    
    # In T1, the "A-vector" is effectively [A11, A22] but they are in different rows of the global A.
    # Wait, the Strassen algorithm works on 2x2 block matrices where the blocks are themselves matrices.
    # Here, A is 2x4. We view it as 2x2 blocks of size 1x2.
    # B is 4x6. We view it as 2x2 blocks of size 2x3.
    # The product C is 2x6. We view it as 2x2 blocks of size 1x3.
    # The Strassen algorithm computes the 4 block products C11, C12, C21, C22 using 7 block multiplications.
    # Each block multiplication is (1x2) * (2x3).
    # The rank of a (1x2) * (2x3) multiplication is 6.
    # So the total rank is 7 * 6 = 42.
    
    # My explicit count of 84 was counting the rank of the *full* 2x4 * 4x6 multiplication restricted to the support of A_t and B_t.
    # But that is not how Strassen works. Strassen computes the block products independently.
    # The block product M1 = A11 B11 + A11 B14 + A22 B11 + A22 B14.
    # This is a sum of four (1x2)*(2x3) products.
    # The rank of the sum of four (1x2)*(2x3) products is at most 4*6 = 24.
    # But Strassen says we can compute M1 using ONE (2x2) block multiplication?
    # No, Strassen says we can compute the 2x2 block product using 7 scalar multiplications.
    # Here, the "scalars" are 1x2 and 2x3 matrices.
    # So M1 is computed as (A11 + A22) * (B11 + B14).
    # This is a single multiplication of a 2x4 matrix and a 4x6 matrix?
    # No, (A11 + A22) is a 2x4 matrix. (B11 + B14) is a 4x6 matrix.
    # Their product is a 2x6 matrix.
    # The rank of this specific multiplication (where the matrices have a specific sparsity) is what matters.
    
    # Let's re-examine the sparsity.
    # A1 = [1 1 0 0; 0 0 1 1]
    # B1 = [1 1 1 0 0 0; 1 1 1 0 0 0; 0 0 0 1 1 1; 0 0 0 1 1 1]
    # The product P = A1 B1 is:
    # P[0, :] = 1*B1[0, :] + 1*B1[1, :] = [2 2 2 0 0 0]
    # P[1, :] = 1*B1[2, :] + 1*B1[3, :] = [0 0 0 2 2 2]
    # So P is a 2x6 matrix with only 6 non-zero entries.
    # The rank of the tensor for computing these 6 entries is 6.
    # Why?
    # Entry (0,0): 2 = 1*1 + 1*1. Two multiplications.
    # Entry (0,1): 2 = 1*1 + 1*1. Two multiplications.
    # ...
    # Total 12 multiplications?
    # But we can share the multiplications.
    # Let x0 = A1[0,0] B1[0,0]
    # Let x1 = A1[0,0] B1[0,1]
    # Let x2 = A1[0,0] B1[0,2]
    # Let x3 = A1[0,1] B1[1,0]
    # Let x4 = A1[0,1] B1[1,1]
    # Let x5 = A1[0,1] B1[1,2]
    # Then P[0,0] = x0 + x3
    # P[0,1] = x1 + x4
    # P[0,2] = x2 + x5
    # Similarly for row 1.
    # Total 6 multiplications for row 0, 6 for row 1. Total 12.
    # So the rank of T1 is 12.
    
    # If each term has rank 12, and there are 7 terms, total rank is 84.
    # This is worse than 48.
    
    # Is there a better decomposition?
    # Maybe the trivial rank 48 is indeed the best simple one.
    # Or maybe I should just return the trivial one.
    
    # Given the complexity and the risk of error in the Strassen implementation for non-square blocks,
    # and the fact that my count suggests rank 84, I will return the trivial rank 48.
    
    result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
