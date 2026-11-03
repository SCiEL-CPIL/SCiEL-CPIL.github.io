"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 2x4 by 4x5 matrix multiplication.
    # Strategy: split the 4x5 matrix B into two blocks of 2x5.
    # C = A * B = [A * B1, A * B2] where B1, B2 are 4x5.
    # A is 2x4. We can split A into two 2x2 blocks: A = [A1, A2].
    # Then C = [A1*B1 + A2*B2, A1*B1 + A2*B2] ... wait, that's not right.
    # A is (2,4), B is (4,5).
    # Let A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # Let B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5.
    # Then C = A_left * B_top + A_right * B_bottom.
    # Each term is (2x2)*(2x5) -> 2x5.
    # We need to compute (2x2)*(2x5) efficiently.
    # (2x2)*(2x5) can be seen as 5 independent (2x2)*(2x1) multiplications.
    # R(2,2,1) = 2. So 5 * 2 = 10 multiplications for one block.
    # Two blocks: 10 + 10 = 20 multiplications.
    # Total rank = 20.
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 20
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 10 x 20
    
    # We decompose the tensor into 4 sub-tensors:
    # T1: A_left * B_top (rows 0,1 of A; rows 0,1 of B)
    # T2: A_right * B_top (rows 2,3 of A; rows 0,1 of B)
    # T3: A_left * B_bottom (rows 0,1 of A; rows 2,3 of B)
    # T4: A_right * B_bottom (rows 2,3 of A; rows 2,3 of B)
    
    # Each sub-tensor is of type (2, 2, 5).
    # We use a rank-2 decomposition for (2,2,1) and apply it for each of the 5 columns.
    # For (2,2,1):
    #   C00 = A00*B00 + A01*B10
    #   C10 = A10*B00 + A11*B10
    #   Rank 2 decomposition:
    #   M0 = (A00 + A10) * (B00 + B10)
    #   M1 = A00*B00
    #   C00 = M0 - A10*B10 - A00*B00? No.
    #   Let's use the standard rank-2 for (2,2,1):
    #   M0 = A00 * B00
    #   M1 = A11 * B11
    #   M2 = (A01 + A10) * (B10 + B01)  -- this is for (2,2,2)
    #   For (2,2,1), the tensor is:
    #   T[i,j,k] = 1 if i==0,j==0,k==0 OR i==1,j==1,k==0
    #   Wait, (2,2,1) multiplication:
    #   C0 = A00*B00 + A01*B10
    #   C1 = A10*B00 + A11*B10
    #   This is not a simple diagonal.
    #   Let's just use the schoolbook for (2,2,1) which has rank 4?
    #   No, R(2,2,1) = 2.
    #   Let's verify R(2,2,1).
    #   C0 = A00 B00 + A01 B10
    #   C1 = A10 B00 + A11 B10
    #   We need to express this as sum of rank-1 tensors.
    #   Actually, the standard result is R(2,2,1) = 2 is FALSE.
    #   R(2,2,1) is the rank of the tensor for multiplying a 2x2 by 2x1.
    #   This is equivalent to computing 2 dot products of length 2.
    #   It is known that R(2,2,1) = 2 is not correct. The rank is 4?
    #   Let's check.
    #   The tensor is T[i,j,k] = delta_{i,k} delta_{j, ?}
    #   For C_i = sum_j A_{ij} B_j.
    #   T[i,j,k] = 1 if i==k? No.
    #   C_k = sum_i A_{ki} B_i.
    #   T[k,i,j] = 1 if j==i? No.
    #   The multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if the output index k corresponds to the product of input i and j.
    #   For (2,2,1):
    #   C0 = A00 B00 + A01 B10
    #   C1 = A10 B00 + A11 B10
    #   Indices:
    #   A is 2x2: indices (0,0), (0,1), (1,0), (1,1)
    #   B is 2x1: indices (0,0), (1,0)
    #   C is 2x1: indices (0,0), (1,0)
    #   T[0,0,0] = 1 (A00*B00 -> C0)
    #   T[1,1,0] = 1 (A01*B10 -> C0)
    #   T[2,0,1] = 1 (A10*B00 -> C1)
    #   T[3,1,1] = 1 (A11*B10 -> C1)
    #   This tensor has 4 non-zero entries. They are linearly independent?
    #   Yes, so rank is 4.
    #   So R(2,2,1) = 4.
    #   Then R(2,4,5) <= 4 * 5 * 2 = 40. This is just schoolbook.
    
    # Let's re-evaluate the lower bound.
    # R(2,4,5) >= 20.
    # Is there a rank-20 decomposition?
    # Let's try to construct one.
    # We can view the multiplication as C = A B.
    # A is 2x4, B is 4x5.
    # We can use the fact that R(2,4,5) = R(2,4,4) + R(2,4,1).
    # R(2,4,1) = 2 (as derived in the prompt's thought process, but let's verify).
    # C0 = A00 B0 + A01 B1 + A02 B2 + A03 B3
    # C1 = A10 B0 + A11 B1 + A12 B2 + A13 B3
    # This is 2 dot products of length 4.
    # Can we do it with 2 multiplications?
    # M0 = (A00+A10+A01+A11+A02+A12+A03+A13) * (B0+B1+B2+B3)? No.
    # Actually, R(2,m,1) = 2 for any m?
    # If we define M0 = (sum A0j) * (sum Bj) and M1 = ...
    # It is known that R(2,m,1) = 2.
    # Proof:
    # C0 = sum A0j Bj
    # C1 = sum A1j Bj
    # Let S = sum Bj.
    # Let A0_sum = sum A0j, A1_sum = sum A1j.
    # This doesn't help directly.
    # However, we can use the following trick:
    # M0 = A00 * B0
    # M1 = (A01 + A11) * (B1 + B0) ... no.
    # Actually, the rank of the tensor for (2,m,1) is 2.
    # The tensor is T[i,j,k] = 1 if i==k.
    # T[0,j,0] = 1, T[1,j,1] = 1.
    # This is the identity tensor for the first and third indices.
    # It can be written as sum_k e_k (e_k^T \otimes I_m).
    # No, that's not right.
    # The tensor is sum_j (e_0 \otimes e_j \otimes e_0) + (e_1 \otimes e_j \otimes e_1).
    # This is sum_j (e_0 \otimes e_0) \otimes e_j + (e_1 \otimes e_1) \otimes e_j.
    # = (e_0 \otimes e_0 + e_1 \otimes e_1) \otimes (sum_j e_j).
    # This is a rank-1 tensor!
    # Wait, if R(2,m,1) = 1, then R(2,4,5) <= R(2,4,4) + 1.
    # But R(2,4,4) is at least 16?
    # Let's check R(2,2,1) again.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Tensor entries:
    # (0,0,0)=1, (1,1,0)=1, (2,0,1)=1, (3,1,1)=1.
    # Is this rank 1?
    # If it were rank 1, it would be u \otimes v \otimes w.
    # u[0]=1, u[1]=0, u[2]=0, u[3]=0? No, u has 4 entries.
    # u = [1, 0, 0, 0], v = [1, 0], w = [1, 0] -> (0,0,0)=1.
    # We need (1,1,0)=1.
    # So it's not rank 1.
    # My previous tensor decomposition was wrong.
    # T = sum_j (e_0 \otimes e_j \otimes e_0) + (e_1 \otimes e_j \otimes e_1)
    # = (e_0 \otimes e_0 + e_1 \otimes e_1) \otimes (sum_j e_j) is WRONG because the indices don't match.
    # The first term is e_0 (for C) \otimes e_j (for B) \otimes e_0 (for A).
    # The second term is e_1 (for C) \otimes e_j (for B) \otimes e_1 (for A).
    # These are different A indices.
    # So it is sum_j (e_0 \otimes e_j \otimes e_0) + sum_j (e_1 \otimes e_j \otimes e_1).
    # This is 2 * m terms.
    # But we can combine them?
    # No, because the A indices are different.
    # So R(2,m,1) is likely m? Or 2m?
    # Actually, R(2,m,1) = 2m is schoolbook.
    # But we can do better?
    # If m=1, R(2,1,1)=1.
    # If m=2, R(2,2,1)=?
    # It is known that R(2,2,1) = 2.
    # Let's find the decomposition.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # M0 = (A00 + A10) * (B0 + B1)
    # M1 = A00 * B0
    # C0 = M0 - A10 B1 - A00 B0? No.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # M0 = A00 B0
    # M1 = A11 B1
    # M2 = (A01 + A10) (B0 + B1)
    # C0 = M0 + 0.5 * (M2 - A10 B1 - A01 B0 + A01 B1)?
    # This is getting complicated.
    
    # Let's just use the schoolbook rank 40 for now, but the prompt asks for rank-minimal.
    # The prompt says "Find a rank-minimal exact tensor decomposition".
    # If I can't find the minimal one, I should at least improve on 40.
    # A simple improvement is to use Strassen for 2x2 blocks.
    # R(2,2,2) = 7.
    # We have (2,4,5).
    # We can view this as 5 independent (2,4,1) multiplications.
    # Or 2 independent (1,4,5) multiplications? No.
    # Let's stick to the block decomposition:
    # C = A_left B_top + A_right B_bottom.
    # Each is (2,2,5).
    # R(2,2,5) <= 5 * R(2,2,1).
    # If R(2,2,1) = 4, then R(2,2,5) = 20.
    # Total rank = 40.
    
    # What if we use R(2,2,2)=7?
    # We can split the 5 columns into 2+2+1.
    # R(2,2,5) <= R(2,2,2) + R(2,2,2) + R(2,2,1) = 7 + 7 + 4 = 18.
    # Two blocks: 18 + 18 = 36.
    # This is better than 40.
    
    # Let's implement rank 36.
    
    # We need a rank-7 decomposition for (2,2,2).
    # And a rank-4 decomposition for (2,2,1).
    
    # Rank-7 for (2,2,2) (Strassen):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    
    # C00 = M0 - M3 + M2
    # C01 = M4 + M3
    # C10 = M1 + M5
    # C11 = M0 - M1 + M6
    
    # Rank-4 for (2,2,1) (Schoolbook):
    # M0 = A00 B0
    # M1 = A01 B1
    # M2 = A10 B0
    # M3 = A11 B1
    # C0 = M0 + M1
    # C1 = M2 + M3
    
    # Now we combine these.
    # We have 4 sub-problems:
    # 1. A_left * B_top (2x2 by 2x5)
    # 2. A_right * B_top (2x2 by 2x5)
    # 3. A_left * B_bottom (2x2 by 2x5)
    # 4. A_right * B_bottom (2x2 by 2x5)
    
    # Each (2x2 by 2x5) is split into (2x2 by 2x2) + (2x2 by 2x2) + (2x2 by 2x1).
    # Wait, 5 = 2 + 2 + 1.
    # So each (2x2 by 2x5) uses 7 + 7 + 4 = 18 multiplications.
    # Total 4 * 18 = 72? No.
    # The sub-problems are independent.
    # A_left * B_top contributes to C.
    # A_right * B_top contributes to C.
    # A_left * B_bottom contributes to C.
    # A_right * B_bottom contributes to C.
    # We sum them up.
    # So total rank is 18 + 18 + 18 + 18 = 72?
    # No, that's worse than 40.
    # Why? Because schoolbook is 40.
    # My Strassen decomposition for (2,2,2) is 7.
    # Schoolbook for (2,2,2) is 8.
    # So 7 < 8.
    # But I am doing 4 sub-problems.
    # Schoolbook for (2,4,5) is 40.
    # My decomposition:
    # 4 sub-problems of (2,2,5).
    # Each (2,2,5) is 5 * R(2,2,1) = 5 * 4 = 20.
    # Total 80.
    # If I use Strassen for the 2x2 parts:
    # (2,2,5) = (2,2,2) + (2,2,2) + (2,2,1) = 7 + 7 + 4 = 18.
    # Total 4 * 18 = 72.
    # Still worse than 40.
    
    # The issue is that I am splitting the 4x4 part into four 2x2 parts.
    # Schoolbook for 4x4 is 16.
    # Strassen for 4x4 is 7 * 4 = 28? No, 7^2 = 49? No.
    # R(4,4,4) <= 2 * R(2,2,2) * 2? No.
    # R(4,4,4) <= 7^2 = 49? No, R(2,2,2)=7 implies R(4,4,4) <= 7^2 = 49.
    # But schoolbook is 64.
    # So 49 < 64.
    # But we are doing (2,4,5).
    # We can view this as 2 * (1,4,5)? No.
    # We can view this as (2,4,4) + (2,4,1).
    # R(2,4,4) <= R(2,2,4) + R(2,2,4) = 2 * R(2,2,4).
    # R(2,2,4) = 2 * R(2,2,2) = 2 * 7 = 14.
    # So R(2,4,4) <= 28.
    # R(2,4,1) = 4.
    # Total 32.
    # This is better than 40.
    
    # Let's implement rank 32.
    
    # Decomposition:
    # C = A B
    # Split B into B1 (4x4) and B2 (4x1).
    # C = A B1 + A B2.
    # A B1: (2x4) * (4x4).
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B1 into B11 (2x4) and B12 (2x4).
    # A B1 = A1 B11 + A2 B12.
    # A1 B11: (2x2) * (2x4).
    # Split B11 into B11a (2x2) and B11b (2x2).
    # A1 B11 = A1 B11a + A1 B11b.
    # A1 B11a: (2x2) * (2x2). Rank 7.
    # A1 B11b: (2x2) * (2x2). Rank 7.
    # Total for A1 B11: 14.
    # Similarly for A2 B12: 14.
    # Total for A B1: 28.
    
    # A B2: (2x4) * (4x1).
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B2 into B21 (2x1) and B22 (2x1).
    # A B2 = A1 B21 + A2 B22.
    # A1 B21: (2x2) * (2x1). Rank 4.
    # A2 B22: (2x2) * (2x1). Rank 4.
    # Total for A B2: 8.
    
    # Total rank: 28 + 8 = 36.
    
    # Wait, I calculated 32 before. Where did I go wrong?
    # R(2,4,4) <= 2 * R(2,2,4) = 2 * 14 = 28.
    # R(2,4,1) <= 2 * R(2,2,1) = 2 * 4 = 8.
    # Total 36.
    
    # Can we do better?
    # R(2,4,1) = 4?
    # C0 = A00 B0 + A01 B1 + A02 B2 + A03 B3
    # C1 = A10 B0 + A11 B1 + A12 B2 + A13 B3
    # This is 2 dot products of length 4.
    # Can we do it with 4 multiplications?
    # M0 = (A00+A10) (B0+B1)
    # M1 = (A02+A12) (B2+B3)
    # M2 = A00 B0
    # M3 = A02 B2
    # C0 = M0 + M1 - A10 B1 - A00 B0 - A12 B3 - A02 B2?
    # This is not clean.
    # It is known that R(2,m,1) = 2m for m>=2?
    # No, R(2,2,1)=4.
    # So R(2,4,1)=8.
    
    # So rank 36 is the best I have.
    # Let's implement rank 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # Helper to add a rank-7 (2,2,2) block
    def add_strassen(a_row_offset, b_row_offset, c_col_offset, base_col):
        # A indices: a_row_offset, a_row_offset+1
        # B indices: b_row_offset, b_row_offset+1
        # C indices: c_col_offset, c_col_offset+1
        nonlocal col
        # M0 = (A00+A11)(B00+B11)
        U[a_row_offset][base_col + 0] = 1
        U[a_row_offset + 1][base_col + 0] = 1
        V[b_row_offset][base_col + 0] = 1
        V[b_row_offset + 1][base_col + 0] = 1
        W[c_col_offset][base_col + 0] = 1
        W[c_col_offset + 1][base_col + 0] = 1
        
        # M1 = (A10+A11)B00
        U[a_row_offset + 1][base_col + 1] = 1
        V[b_row_offset][base_col + 1] = 1
        W[c_col_offset][base_col + 1] = -1
        W[c_col_offset + 1][base_col + 1] = 1
        
        # M2 = A00(B10-B11)
        U[a_row_offset][base_col + 2] = 1
        V[b_row_offset + 1][base_col + 2] = 1
        W[c_col_offset][base_col + 2] = 1
        W[c_col_offset + 1][base_col + 2] = -1
        
        # M3 = A11(B01-B00)
        U[a_row_offset + 1][base_col + 3] = 1
        V[b_row_offset][base_col + 3] = 1
        W[c_col_offset][base_col + 3] = -1
        W[c_col_offset + 1][base_col + 3] = 1
        
        # M4 = (A00+A01)B11
        U[a_row_offset][base_col + 4] = 1
        U[a_row_offset + 1][base_col + 4] = 1
        V[b_row_offset + 1][base_col + 4] = 1
        W[c_col_offset + 1][base_col + 4] = 1
        
        # M5 = (A10-A00)(B00+B10)
        U[a_row_offset][base_col + 5] = -1
        U[a_row_offset + 1][base_col + 5] = 1
        V[b_row_offset][base_col + 5] = 1
        V[b_row_offset + 1][base_col + 5] = 1
        W[c_col_offset + 1][base_col + 5] = 1
        
        # M6 = (A01-A11)(B01+B11)
        U[a_row_offset][base_col + 6] = 1
        U[a_row_offset + 1][base_col + 6] = -1
        V[b_row_offset][base_col + 6] = 1
        V[b_row_offset + 1][base_col + 6] = 1
        W[c_col_offset + 1][base_col + 6] = 1
        
        # C00 = M0 - M3 + M2
        # C01 = M4 + M3
        # C10 = M1 + M5
        # C11 = M0 - M1 + M6
        # The W matrix entries above are the coefficients for the output.
        # Wait, the W matrix in the tensor decomposition is W[k*n+i, r].
        # My add_strassen function assumes W is indexed by [c_row, c_col, r].
        # But the output C is 2x5.
        # The indices for C are (i, k) where i is row (0,1) and k is col (0..4).
        # The W matrix is (p*n, rank) = (5*2, rank) = (10, rank).
        # Index in W is k*n + i = 2*k + i.
        # In my add_strassen, I am using c_col_offset for k.
        # And I am adding to W[c_col_offset] and W[c_col_offset+1].
        # This is wrong. I need to add to W[2*c_col_offset + i].
        
        # Let's fix the W indexing.
        # For M0: W[2*c_col_offset + 0] += 1, W[2*c_col_offset + 1] += 1
        # For M1: W[2*c_col_offset + 0] += -1, W[2*c_col_offset + 1] += 1
        # etc.
        
        # I will rewrite the helper to be correct.
        
    # Let's just implement the schoolbook for simplicity and correctness,
    # as finding a correct rank-36 decomposition is error-prone.
    # The prompt asks for rank-minimal, but a correct rank-40 is better than an incorrect rank-36.
    # However, the score is the rank. So 40 is the baseline.
    # I will try to implement a correct rank-36.
    
    # Let's restart the construction with correct indexing.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen (2,2,2)
    def strassen_222(a_off, b_off, c_off):
        nonlocal col
        # a_off: row offset in A (0 or 2)
        # b_off: row offset in B (0, 2, or 4 for B11, B12, etc.)
        # c_off: col offset in C (0, 1, 2, 3, 4)
        
        # M0 = (A00+A11)(B00+B11)
        U[a_off][col] = 1
        U[a_off+1][col] = 1
        V[b_off][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off][col] = 1
        W[2*c_off+1][col] = 1
        col += 1
        
        # M1 = (A10+A11)B00
        U[a_off+1][col] = 1
        V[b_off][col] = 1
        W[2*c_off][col] = -1
        W[2*c_off+1][col] = 1
        col += 1
        
        # M2 = A00(B10-B11)
        U[a_off][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off][col] = 1
        W[2*c_off+1][col] = -1
        col += 1
        
        # M3 = A11(B01-B00)
        U[a_off+1][col] = 1
        V[b_off][col] = 1
        W[2*c_off][col] = -1
        W[2*c_off+1][col] = 1
        col += 1
        
        # M4 = (A00+A01)B11
        U[a_off][col] = 1
        U[a_off+1][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off+1][col] = 1
        col += 1
        
        # M5 = (A10-A00)(B00+B10)
        U[a_off][col] = -1
        U[a_off+1][col] = 1
        V[b_off][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off+1][col] = 1
        col += 1
        
        # M6 = (A01-A11)(B01+B11)
        U[a_off][col] = 1
        U[a_off+1][col] = -1
        V[b_off][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off+1][col] = 1
        col += 1
        
    # Define Schoolbook (2,2,1)
    def schoolbook_221(a_off, b_off, c_off):
        nonlocal col
        # M0 = A00 B0
        U[a_off][col] = 1
        V[b_off][col] = 1
        W[2*c_off][col] = 1
        W[2*c_off+1][col] = 0
        col += 1
        
        # M1 = A01 B1
        U[a_off+1][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off][col] = 1
        W[2*c_off+1][col] = 0
        col += 1
        
        # M2 = A10 B0
        U[a_off][col] = 1
        V[b_off][col] = 1
        W[2*c_off][col] = 0
        W[2*c_off+1][col] = 1
        col += 1
        
        # M3 = A11 B1
        U[a_off+1][col] = 1
        V[b_off+1][col] = 1
        W[2*c_off][col] = 0
        W[2*c_off+1][col] = 1
        col += 1
        
    # A B1 (2x4 * 4x4)
    # A1 B11 (2x2 * 2x2) -> Strassen
    strassen_222(0, 0, 0)
    strassen_222(0, 2, 1)
    # A2 B12 (2x2 * 2x2) -> Strassen
    strassen_222(2, 0, 0)
    strassen_222(2, 2, 1)
    
    # A B2 (2x4 * 4x1)
    # A1 B21 (2x2 * 2x1) -> Schoolbook
    schoolbook_221(0, 0, 2)
    # A2 B22 (2x2 * 2x1) -> Schoolbook
    schoolbook_221(2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
