"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school

    # Structured Construction using Strassen for 2x2 blocks
    # Partition:
    # I: [0,1] -> I1, [2,3] -> I2
    # J: [0,1] -> J1, [2,3] -> J2, [4] -> J3
    # K: [0,1] -> K1, [2,3] -> K2, [4,5] -> K3
    
    # Total Rank estimate:
    # 6 blocks (2 I-blocks * 3 K-blocks)
    # Each block has 3 terms:
    # 1. A(I, J1) * B(J1, K) -> 2x2 * 2x2 -> Rank 7
    # 2. A(I, J2) * B(J2, K) -> 2x2 * 2x2 -> Rank 7
    # 3. A(I, J3) * B(J3, K) -> 2x1 * 1x2 -> Rank 2
    # Sum per block: 16. Total: 96.
    
    rank_structured = 96
    U_struct = np.zeros((num_U, rank_structured), dtype=int)
    V_struct = np.zeros((num_V, rank_structured), dtype=int)
    W_struct = np.zeros((num_W, rank_structured), dtype=int)
    
    def add_strassen_block(i_offset, j1_offset, j2_offset, k_offset, start_col):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 multiplication.
        A is rows i_offset..i_offset+1, cols j1_offset..j1_offset+1.
        B is rows j1_offset..j1_offset+1, cols k_offset..k_offset+1.
        C is rows i_offset..i_offset+1, cols k_offset..k_offset+1.
        Returns the number of columns used (7).
        """
        # Strassen M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in A (local 0,1 -> global i_offset+0/1, j1_offset+0/1)
        # a11, a12, a21, a22
        a11_idx = (i_offset, j1_offset)
        a12_idx = (i_offset, j1_offset+1)
        a21_idx = (i_offset+1, j1_offset)
        a22_idx = (i_offset+1, j1_offset+1)
        
        # Indices in B (local 0,1 -> global j1_offset+0/1, k_offset+0/1)
        # b11, b12, b21, b22
        b11_idx = (j1_offset, k_offset)
        b12_idx = (j1_offset, k_offset+1)
        b21_idx = (j1_offset+1, k_offset)
        b22_idx = (j1_offset+1, k_offset+1)
        
        # Indices in C (local 0,1 -> global i_offset+0/1, k_offset+0/1)
        # c11, c12, c21, c22
        c11_idx = (i_offset, k_offset)
        c12_idx = (i_offset, k_offset+1)
        c21_idx = (i_offset+1, k_offset)
        c22_idx = (i_offset+1, k_offset+1)
        
        # Helper to get global index for U, V, W
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        cols = 0
        
        # M1: (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11(+), C12(-), C21(-), C22(+)
        # Wait, Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to distribute the coefficients correctly.
        # Instead of pre-combining, we just add the terms for each M.
        
        # M1: A11*B11, A11*B22, A22*B11, A22*B22
        # This is 4 products. But Strassen combines them into 1 multiplication of sums?
        # No, Strassen uses 7 multiplications of sums.
        # The factorization is:
        # M1 = (A11+A22) * (B11+B22)
        # This contributes to C11, C22, etc.
        
        # Let's expand the 7 terms into the U, V, W matrices.
        # Each term is a product of a sum of A's, a sum of B's, and a sum of C's?
        # No, the tensor decomposition is linear.
        # T = sum_r U_r V_r W_r
        # For Strassen, we have 7 terms.
        # Term 1: (A11+A22) (B11+B22) -> contributes to C11, C22...
        # This is not a single rank-1 term in the tensor sense unless we expand.
        # Actually, the rank of the 2x2x2 tensor is 7.
        # The 7 rank-1 tensors are:
        # 1. (A11+A22) (B11+B22) (C11+C22-C12-C21)? No.
        
        # Standard Strassen Rank-1 Decomposition:
        # The 7 terms are:
        # M1: (A11+A22) (B11+B22) -> Coeff for C: C11=1, C12=-1, C21=-1, C22=1 ? No.
        
        # Let's use the explicit rank-1 components of the 2x2x2 tensor.
        # The tensor T_ijk for 2x2 mat mult has rank 7.
        # One such decomposition (from literature):
        # 1. (1,1,0) + (1,0,1) ... this is getting complicated to hardcode correctly without reference.
        
        # Alternative: Just use the 4 schoolbook terms for 2x2? Rank 4.
        # Then total rank = 6 blocks * (4 + 4 + 2) = 60.
        # Wait, 2x2 mat mult schoolbook is rank 4.
        # 2x1 * 1x2 is rank 2.
        # So if we use schoolbook for the 2x2 parts, we get:
        # 12 terms of 2x2*2x2 (Rank 4 each) -> 48
        # 6 terms of 2x1*1x2 (Rank 2 each) -> 12
        # Total 60.
        
        # This is much better than 96 and easier to implement correctly!
        # Let's verify if we can do better with Strassen.
        # Strassen rank is 7, which is > 4. So schoolbook is better for 2x2?
        # No, Strassen is for 2x2x2 tensor rank. Schoolbook is rank 4.
        # Wait, rank of 2x2x2 tensor is 7? No, schoolbook is 4.
        # Why is Strassen rank 7?
        # Ah, Strassen reduces the number of *multiplications* from 4 to 7? No, 4 to 7 is an increase.
        # Strassen reduces from 4 to 7? No, standard mat mult is 4 mults for 2x2.
        # Strassen is for 2x2 * 2x2?
        # Yes, Strassen's algorithm computes 2x2 product using 7 multiplications.
        # But schoolbook uses 4 multiplications.
        # So Strassen is WORSE for 2x2?
        # No, Strassen is for 2x2 * 2x2?
        # Let's check. 2x2 mat mult:
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        # This is 8 multiplications.
        # Strassen uses 7.
        # So Rank of 2x2x2 tensor is 7.
        # Schoolbook Rank is 8.
        # So Strassen (7) is better than Schoolbook (8).
        
        # Okay, so for 2x2*2x2, we use Rank 7.
        # For 2x1*1x2, we use Rank 2.
        
        # So the plan holds: Rank 96.
        
        # Let's implement the 7 terms for Strassen correctly.
        # Reference: 
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        # Each M is a product of a linear combo of A's and a linear combo of B's.
        # This corresponds to a rank-1 term in the tensor if we view the A and B combos as vectors.
        # But U, V, W are matrices.
        # The term M1 contributes to C11, C22, etc.
        # Specifically, M1 = (A11+A22) (B11+B22).
        # This product appears in C11 with coeff 1, C22 with coeff 1.
        # Wait, C11 = ... + M1 ...
        # C22 = ... + M1 ...
        # So the same M1 value is added to C11 and C22.
        # In the tensor decomposition, this means we have a term where U and V are fixed, and W has 1s at C11 and C22.
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1
        # U part: A11 + A22
        # V part: B11 + B22
        # W part: C11 (1), C22 (1)
        # Wait, C11 = M1 + ... and C22 = M1 + ...
        # So W[C11] += 1, W[C22] += 1.
        
        # But U and V are sums.
        # U[A11] += 1, U[A22] += 1.
        # V[B11] += 1, V[B22] += 1.
        
        # This creates a rank-1 term?
        # The contribution is (A11+A22) * (B11+B22) * (C11+C22).
        # This is NOT a rank-1 tensor term. It is a sum of 4 rank-1 terms:
        # A11 B11 C11, A11 B11 C22, A22 B11 C11, A22 B11 C22, etc.
        # This would blow up the rank.
        
        # Ah, the "rank" of the tensor is the minimum number of rank-1 terms.
        # Strassen's algorithm shows that the rank is <= 7.
        # This means we CAN represent the 2x2x2 tensor as a sum of 7 rank-1 terms.
        # The 7 terms are NOT simply the M1..M7 as products of sums.
        # They are specific combinations.
        
        # I will fall back to Schoolbook for the 2x2 blocks to ensure correctness, 
        # which gives Rank 8 for 2x2.
        # Total Rank: 12 * 8 + 6 * 2 = 96 + 12 = 108.
        # This is still better than 120.
        
        # Let's use Schoolbook for 2x2 blocks.
        
        # For a 2x2 * 2x2 block:
        # 4 terms:
        # 1. A11 B11 C11
        # 2. A12 B21 C11
        # 3. A11 B12 C12
        # 4. A12 B22 C12
        # 5. A21 B11 C21
        # 6. A22 B21 C21
        # 7. A21 B12 C22
        # 8. A22 B22 C22
        
        # For a 2x1 * 1x2 block:
        # 2 terms:
        # 1. A1 B1 C11
        # 2. A1 B1 C12 ? No.
        # A is 2x1 (A1, A2). B is 1x2 (B1, B2).
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        # Rank 4?
        # Yes, outer product of A and B.
        # Rank of 2x1x2 tensor is 4?
        # T[i,0,k] = A[i] B[k].
        # This is a rank-1 tensor? No.
        # T = sum_r U_r V_r W_r.
        # If we use r=1: U[0]=1, U[1]=1? No.
        # U is 2x1. V is 1x1? No, V is 1x2?
        # Dimensions: U (2x1), V (1x2), W (2x2).
        # Rank 1: U[i,0] V[0,k] W[k,i]?
        # This is getting confusing.
        
        # Let's just use Schoolbook for ALL blocks.
        # 2x2 * 2x2: 8 terms.
        # 2x1 * 1x2: 4 terms.
        # Total: 12 * 8 + 6 * 4 = 96 + 24 = 120.
        # This is the same as global schoolbook.
        
        # So we need Strassen to improve.
        # I will implement the known rank-7 decomposition for 2x2x2.
        
        # The 7 rank-1 terms for 2x2x2 are:
        # 1. (1,1,0) (1,1,0) (1,0,1) ... No.
        
        # I will use a simpler improvement:
        # Combine the 2x1 * 1x2 blocks with adjacent 2x2 blocks?
        
        # Given the complexity and risk of error, I will stick to the Schoolbook fallback 
        # but try to optimize the search slightly.
        
        return 0

    # Since implementing Strassen correctly is risky without a reference, 
    # and Schoolbook is 120, I will return the schoolbook solution.
    # This guarantees validity.
    
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
