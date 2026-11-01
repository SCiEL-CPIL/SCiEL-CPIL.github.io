"""
Strategy: Systematic Block Decomposition + Local Search
1. Construct a valid rank-42 decomposition using Strassen (2x2x2, rank 7) blocks.
   Split 2x4 * 4x6 into 1 (2x2) * 3 (2x2) blocks of size 2x2 * 2x2.
   Total rank = 1 * 3 * 7 = 21? 
   Wait, 2x4 * 4x6.
   Split rows of A (2) into 1 block of 2.
   Split cols of A / rows of B (4) into 2 blocks of 2.
   Split cols of B (6) into 3 blocks of 2.
   So we have 1 * 2 * 3 = 6 blocks of size 2x2 * 2x2.
   Each block requires rank 7 (Strassen).
   Total rank = 6 * 7 = 42.
   
   The seed program (schoolbook) is rank 48.
   So rank 42 is a guaranteed improvement (score 48/42 = 1.14).
   
   Can we do better?
   Maybe mix in 2x3x2 blocks? 
   Strassen-Winograd for 2x3x2 has rank 9.
   If we use 2x3x2 blocks:
   Split cols of B (6) into 2 blocks of 3.
   Split cols of A / rows of B (4) into 2 blocks of 2.
   1 * 2 * 2 = 4 blocks.
   Rank = 4 * 9 = 36.
   
   Let's verify Strassen-Winograd 2x3x2 rank 9.
   Yes, it is a known result.
   
   So rank 36 is achievable.
   Score = 48 / 36 = 1.333.
   
   Can we go lower?
   Lower bound for 2x4x6 is at least max(2,4,6)*... no, lower bound is roughly n*m*p / (something).
   Actually, the rank of 2x4x6 is at least 24 (from max(nm, mp, pn) logic? No, rank >= max(n,m,p) * min(n,m,p) is not a lower bound. 
   Rank >= max(n*m, m*p, p*n) is false.
   Rank >= n*m*p / (max(n,m,p)) ? No.
   
   Known ranks:
   2x2x2 = 7
   2x3x2 = 9
   2x4x2 = 11 (Winograd)
   2x4x4 = 22? Or 2*2x2x2 = 14? 
   If we use 2x4x2 (rank 11) and 2x2x2 (rank 7).
   
   Let's stick to the guaranteed 36 first, then try to optimize.
   
   Implementation of 2x3x2 rank 9:
   A is 2x3, B is 3x2.
   A = [[a11, a12, a13], [a21, a22, a23]]
   B = [[b11, b12], [b21, b22], [b31, b32]]
   
   Strassen-Winograd algorithm for 2x3x2 (rank 9):
   M1 = (a11 + a12 + a13) * (b11 + b21 + b31)
   M2 = (a11 - a12) * (b11 - b21)
   M3 = (a21 - a22) * (b21 - b31)
   M4 = a22 * b22
   M5 = (a22 + a23) * (b22 + b32)
   M6 = (a11 - a13) * (b11 + b31)
   M7 = (a12 - a13) * (b12 + b32)
   M8 = (a11 + a13) * (b11 - b31)
   M9 = (a12 + a13) * (b12 - b32)
   
   C11 = M1 - M2 + M6 - M8
   C12 = M1 - M7 + M8 - M9
   C21 = M1 - M3 + M8 - M6
   C22 = M1 - M5 + M3 + M9
   
   Wait, I need to verify these formulas. 
   Standard 2x3x2 rank 9 decomposition:
   
   Let's use a known valid one.
   U, V, W for 2x3x2.
   Indices:
   A: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> 6 elements
   B: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1) -> 6 elements
   C: (0,0), (0,1), (1,0), (1,1) -> 4 elements
   
   We need 9 terms.
   
   Actually, let's just implement the 2x2x2 Strassen (rank 7) and 2x3x2 (rank 9) blocks and combine them.
   
   2x2x2 Strassen:
   A 2x2, B 2x2.
   M1 = (a00+a11)(b00+b11)
   M2 = (a10+a11)b00
   M3 = a00(b10-b11)
   M4 = a11(b01-b00)
   M5 = (a00+a01)b11
   M6 = (a10-a00)(b00+b10)
   M7 = (a01-a11)(b01+b11)
   
   C00 = M1 + M4 - M5 + M7
   C01 = M3 + M5
   C10 = M2 + M4
   C11 = M1 - M2 + M3 + M6
   
   U (2x2 A, 7 cols):
   Row 0 (a00): [1, 0, 1, 0, 1, -1, -1]
   Row 1 (a10): [0, 1, 0, 0, 0, 1, 0]
   Row 2 (a11): [1, 1, 0, 1, 0, 0, -1]
   Row 3 (a01): [0, 0, 0, 0, 1, 0, 1]
   
   Wait, indices for A in 2x2:
   0: a00
   1: a01
   2: a10
   3: a11
   
   Let's map carefully.
   A = [[a00, a01], [a10, a11]]
   B = [[b00, b01], [b10, b11]]
   
   M1 = (a00+a11)(b00+b11)
   M2 = (a10+a11)b00
   M3 = a00(b10-b11)
   M4 = a11(b01-b00)
   M5 = (a00+a01)b11
   M6 = (a10-a00)(b00+b10)
   M7 = (a01-a11)(b01+b11)
   
   U (A factors, 4 rows x 7 cols):
   r0 (a00): [1, 0, 1, 0, 1, -1, 0]
   r1 (a01): [0, 0, 0, 0, 1, 0, 1]
   r2 (a10): [0, 1, 0, 0, 0, 1, 0]
   r3 (a11): [1, 1, 0, 1, 0, 0, -1]
   
   V (B factors, 4 rows x 7 cols):
   r0 (b00): [1, 1, 0, -1, 0, 1, 0]
   r1 (b01): [0, 0, 0, 1, 0, 0, 1]
   r2 (b10): [0, 0, 1, 0, 0, 1, 0]
   r3 (b11): [1, 0, -1, 0, 1, 0, 1]
   
   W (C factors, 4 rows x 7 cols):
   C = [[c00, c01], [c10, c11]]
   c00 = M1 + M4 - M5 + M7
   c01 = M3 + M5
   c10 = M2 + M4
   c11 = M1 - M2 + M3 + M6
   
   r0 (c00): [1, 0, 0, 1, -1, 0, 1]
   r1 (c01): [0, 0, 1, 0, 1, 0, 0]
   r2 (c10): [0, 1, 0, 1, 0, 0, 0]
   r3 (c11): [1, -1, 1, 0, 0, 1, 0]
   
   Let's double check C01 = M3 + M5.
   M3 = a00(b10-b11)
   M5 = (a00+a01)b11
   Sum = a00 b10 - a00 b11 + a00 b11 + a01 b11 = a00 b10 + a01 b11. Correct.
   
   C10 = M2 + M4.
   M2 = (a10+a11)b00
   M4 = a11(b01-b00)
   Sum = a10 b00 + a11 b00 + a11 b01 - a11 b00 = a10 b00 + a11 b01. Correct.
   
   C11 = M1 - M2 + M3 + M6.
   M1 = (a00+a11)(b00+b11) = a00 b00 + a00 b11 + a11 b00 + a11 b11
   -M2 = -(a10+a11)b00 = -a10 b00 - a11 b00
   M3 = a00(b10-b11) = a00 b10 - a00 b11
   M6 = (a10-a00)(b00+b10) = a10 b00 + a10 b10 - a00 b00 - a00 b10
   
   Sum:
   a00 b00: 1 - 1 = 0? 
   Wait.
   M1 has a00 b00.
   M6 has -a00 b00.
   So a00 b00 cancels.
   
   a00 b11: 1 - 1 = 0.
   a11 b00: 1 - 1 = 0.
   a11 b11: 1.
   a10 b00: -1 + 1 = 0.
   a10 b10: 1.
   a00 b10: -1 - 1 = -2?
   
   Let's re-evaluate C11 formula.
   Standard Strassen C11 = M1 - M2 + M3 + M6.
   Let's check a00 b10 term.
   M3 has a00 b10.
   M6 has -a00 b10.
   Sum 0.
   
   Let's check a10 b10 term.
   M6 has a10 b10.
   Sum 1.
   
   So C11 = a11 b11 + a10 b10. Correct.
   
   So the 2x2x2 decomposition is valid.
   
   Now for 2x3x2 rank 9.
   A 2x3, B 3x2.
   A = [[a00, a01, a02], [a10, a11, a12]]
   B = [[b00, b01], [b10, b11], [b20, b21]]
   
   Known rank 9 decomposition (Winograd):
   M1 = (a00+a01+a02)(b00+b10+b20)
   M2 = (a00-a01)(b00-b10)
   M3 = (a10-a11)(b10-b20)
   M4 = a11 b11
   M5 = (a11+a12)(b11+b21)
   M6 = (a00-a02)(b00+b20)
   M7 = (a01-a02)(b01+b21)
   M8 = (a00+a02)(b00-b20)
   M9 = (a01+a02)(b01-b21)
   
   C00 = M1 - M2 + M6 - M8
   C01 = M1 - M7 + M8 - M9
   C10 = M1 - M3 + M8 - M6
   C11 = M1 - M5 + M3 + M9
   
   Let's verify C00.
   M1 = (a00+a01+a02)(b00+b10+b20)
   -M2 = -(a00-a01)(b00-b10) = -a00 b00 + a00 b10 + a01 b00 - a01 b10
   M6 = (a00-a02)(b00+b20) = a00 b00 + a00 b20 - a02 b00 - a02 b20
   -M8 = -(a00+a02)(b00-b20) = -a00 b00 + a00 b20 - a02 b00 + a02 b20
   
   Sum a00 b00: 1 - 1 + 1 - 1 = 0.
   Sum a00 b10: 1 + 1 = 2?
   Wait.
   M1 has a00 b10.
   -M2 has a00 b10.
   So 2 a00 b10.
   This is wrong. C00 should have a00 b00.
   
   Let's check the formula for 2x3x2 rank 9 again.
   There are different variants.
   
   Let's use a simpler approach.
   We can decompose 2x3x2 into 2x2x2 + 2x1x2.
   2x2x2 rank 7.
   2x1x2 rank 2.
   Total rank 9.
   
   A = [A1, A2] where A1 is 2x2, A2 is 2x1.
   B = [B1; B2] where B1 is 2x2, B2 is 1x2.
   
   C = A1 B1 + A2 B2.
   
   We can compute A1 B1 with Strassen (rank 7).
   And A2 B2 with rank 2 (schoolbook for 2x1 * 1x2).
   
   A2 = [[a02], [a12]]
   B2 = [[b20, b21]]
   
   M8 = a02 * b20
   M9 = a02 * b21
   M10 = a12 * b20
   M11 = a12 * b21
   
   Wait, that's rank 4.
   But A2 B2 is outer product of column and row.
   [[a02 b20, a02 b21], [a12 b20, a12 b21]]
   
   We can do this with 4 multiplications.
   So total rank = 7 + 4 = 11.
   
   Is there a rank 9?
   Yes, Winograd found rank 9 for 2x3x2.
   
   Let's look up the exact 2x3x2 rank 9 decomposition.
   
   From "The rank of matrix multiplication" literature:
   
   M1 = (a00 + a01 + a02) * (b00 + b10 + b20)
   M2 = (a00 - a01) * (b00 - b10)
   M3 = (a10 - a11) * (b10 - b20)
   M4 = a11 * b11
   M5 = (a11 + a12) * (b11 + b21)
   M6 = (a00 - a02) * (b00 + b20)
   M7 = (a01 - a02) * (b01 + b21)
   M8 = (a00 + a02) * (b00 - b20)
   M9 = (a01 + a02) * (b01 - b21)
   
   C00 = M1 - M2 + M6 - M8
   C01 = M1 - M7 + M8 - M9
   C10 = M1 - M3 + M8 - M6
   C11 = M1 - M5 + M3 + M9
   
   Let's re-verify C00 with these formulas.
   Terms involving a00 b00:
   M1: +1
   M2: -(-1) = +1? No, M2 = (a00-a01)(b00-b10). Coeff of a00 b00 is +1. So -M2 has -1.
   M6: (a00-a02)(b00+b20). Coeff of a00 b00 is +1.
   M8: (a00+a02)(b00-b20). Coeff of a00 b00 is +1. So -M8 has -1.
   
   Sum: 1 - 1 + 1 - 1 = 0.
   This is still 0.
   
   Maybe the formula for C00 is different.
   
   Let's try:
   C00 = M1 - M2 + M6 - M8
   C01 = M1 - M7 + M8 - M9
   C10 = M1 - M3 + M8 - M6
   C11 = M1 - M5 + M3 + M9
   
   Let's check a00 b10 term in C00.
   M1: +1
   M2: -(-1) = +1 (from -a00 * -b10)
   M6: 0
   M8: 0
   Sum: 2.
   
   This implies the formula is wrong or I'm misinterpreting.
   
   Let's use the 2x2x2 (rank 7) and 2x1x2 (rank 2) split?
   No, 2x1x2 is rank 2?
   A 2x1, B 1x2.
   A = [[a], [b]], B = [[c, d]].
   C = [[ac, ad], [bc, bd]].
   M1 = a*c
   M2 = a*d
   M3 = b*c
   M4 = b*d
   Rank 4.
   
   So 7+4=11.
   
   Is rank 9 possible?
   Yes.
   
   Let's use a known valid 2x3x2 rank 9 from a paper.
   
   Actually, let's just use the 2x2x2 blocks.
   2x4 * 4x6.
   Split into 6 blocks of 2x2 * 2x2.
   Rank 42.
   
   This is safe and guaranteed to work.
   Score 48/42 = 1.14.
   
   We can then try to merge blocks or use 2x3x2 if we find a valid formula.
   
   For now, I will implement the rank 42 solution.
   
   Then, I will try to use a local search to reduce the rank further.
   
   The local search will start from the rank 42 solution and try to set some entries to 0 or merge columns.
   
   Actually, a simpler improvement:
   Use 2x2x2 (rank 7) for the 2x2 * 2x2 blocks.
   
   The structure is:
   A (2x4) is split into A1 (2x2) and A2 (2x2) horizontally.
   B (4x6) is split into B11, B12, B13 (top 2x2, 2x2, 2x2) and B21, B22, B23 (bottom 2x2, 2x2, 2x2).
   
   C = A1 * [B11, B12, B13] + A2 * [B21, B22, B23]
   
   C = [A1 B11 + A2 B21, A1 B12 + A2 B22, A1 B13 + A2 B23]
   
   Each term is a sum of two 2x2 matrix products.
   We can compute each 2x2 product with Strassen (rank 7).
   So for each column block (e.g., A1 B11 + A2 B21), we use 2 * 7 = 14 multiplications.
   There are 3 column blocks.
   Total rank = 3 * 14 = 42.
   
   This matches.
   
   Implementation details:
   U (2x4 A, rank 42)
   V (4x6 B, rank 42)
   W (6x2 C, rank 42) -- Wait, W shape is (p*n, rank) = (6*2, rank) = (12, rank).
   Indices for W: k*n+i. k is 0..5, i is 0..1.
   
   Let's build the matrices.
   
   U has 8 rows (2*4).
   V has 24 rows (4*6).
   W has 12 rows (6*2).
   
   For each of the 3 column blocks (j=0,1,2 in the 4x6 split into 2x2, 2x2, 2x2? No, 4x6 split into 2x2, 2x2, 2x2 vertically? No.
   
   A is 2x4. B is 4x6.
   Split A into A_top (2x2) and A_bottom (2x2)? No, A has 2 rows.
   Split A horizontally: A_left (2x2), A_right (2x2).
   Split B vertically: B_top (2x6), B_bottom (2x6).
   Split B_top into B_t1 (2x2), B_t2 (2x2), B_t3 (2x2).
   Split B_bottom into B_b1 (2x2), B_b2 (2x2), B_b3 (2x2).
   
   C = A_left * B_top + A_right * B_bottom
   C = A_left * [B_t1, B_t2, B_t3] + A_right * [B_b1, B_b2, B_b3]
   C = [A_left B_t1 + A_right B_b1, A_left B_t2 + A_right B_b2, A_left B_t3 + A_right B_b3]
   
   Each block is 2x2 * 2x2 + 2x2 * 2x2.
   We can compute A_left B_t1 with Strassen (rank 7).
   And A_right B_b1 with Strassen (rank 7).
   Sum them.
   
   So for the first 2 columns of C, we use 14 terms.
   For the next 2, 14 terms.
   For the last 2, 14 terms.
   Total 42.
   
   This is correct.
   
   Now, can we do better?
   Maybe use 2x3x2 rank 9?
   If we group 3 columns of B.
   B is 4x6.
   Split into 2 blocks of 3 columns.
   B1 (4x3), B2 (4x3).
   A is 2x4.
   
   C = A * B1 + A * B2? No.
   C = A * [B1, B2].
   
   We need to compute A * B1 (2x4 * 4x3) and A * B2 (2x4 * 4x3).
   
   If we could decompose 2x4 * 4x3 into something smaller.
   
   Let's stick to 42. It's a solid improvement.
   
   Code structure:
   1. Define Strassen 2x2x2 factors.
   2. Construct U, V, W for the 6 blocks.
   3. Return.
   
   I will also add a check to ensure the decomposition is valid.
   
   Then, I will try a simple local search to see if I can reduce the rank.
   Local search:
   - Start with rank 42.
   - Try to remove a column (set to 0) and see if the decomposition still holds (it won't, unless the column was redundant).
   - Try to merge two columns (if they are proportional).
   
   Given the time constraint, I will just output the rank 42 solution.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U_s (4x7), V_s (4x7), W_s (4x7)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify Strassen
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A @ B
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    
    # We will construct U, V, W for the full problem
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices mapping:
    # A (2x4): i in 0..1, j in 0..3. Index i*m + j
    # B (4x6): j in 0..3, k in 0..5. Index j*p + k
    # C (6x2): k in 0..5, i in 0..1. Index k*n + i
    
    # Block 1: A_left (2x2) * B_t1 (2x2) -> C_col 0,1
    # A_left indices: i=0,1; j=0,1
    # B_t1 indices: j=0,1; k=0,1
    # C indices: k=0,1; i=0,1
    
    def add_strassen_block(U, V, W, A_i, A_j, B_j, B_k, C_k, C_i, offset):
        """
        Add a Strassen block to the global matrices.
        A_i: list of i indices for A (2)
        A_j: list of j indices for A (2)
        B_j: list of j indices for B (2)
        B_k: list of k indices for B (2)
        C_k: list of k indices for C (2)
        C_i: list of i indices for C (2)
        offset: column offset in the rank dimension
        """
        for r in range(7):
            # U part
            for local_i in range(2):
                global_i = A_i[local_i]
                local_j = 0 # placeholder
                # We need to map local (li, lj) to global (gi, gj)
                # U_s[local_i, local_j, r]
                
            # Let's do it element by element
            for li in range(2):
                for lj in range(2):
                    val_u = U_s[li, r] # Wait, U_s is (4,7). Rows are i*2+j?
                    # U_s rows: 0: a00, 1: a01, 2: a10, 3: a11
                    # So row index is li*2 + lj?
                    # li=0, lj=0 -> 0
                    # li=0, lj=1 -> 1
                    # li=1, lj=0 -> 2
                    # li=1, lj=1 -> 3
                    # Yes.
                    
                    global_i = A_i[li]
                    global_j = A_j[lj]
                    U[global_i * m + global_j, offset + r] += U_s[li*2 + lj, r]
                    
            for lj in range(2):
                for lk in range(2):
                    # V_s rows: 0: b00, 1: b01, 2: b10, 3: b11
                    # Row index is lj*2 + lk?
                    # lj=0, lk=0 -> 0
                    # lj=0, lk=1 -> 1
                    # lj=1, lk=0 -> 2
                    # lj=1, lk=1 -> 3
                    # Yes.
                    
                    global_j = B_j[lj]
                    global_k = B_k[lk]
                    V[global_j * p + global_k, offset + r] += V_s[lj*2 + lk, r]
                    
            for lk in range(2):
                for li in range(2):
                    # W_s rows: 0: c00, 1: c01, 2: c10, 3: c11
                    # Row index is lk*2 + li?
                    # lk=0, li=0 -> 0
                    # lk=0, li=1 -> 1
                    # lk=1, li=0 -> 2
                    # lk=1, li=1 -> 3
                    # Yes.
                    
                    global_k = C_k[lk]
                    global_i = C_i[li]
                    W[global_k * n + global_i, offset + r] += W_s[lk*2 + li, r]
                    
    # Block 1: A_left (j=0,1) * B_t1 (j=0,1, k=0,1) -> C (k=0,1, i=0,1)
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Block 2: A_right (j=2,3) * B_b1 (j=2,3, k=0,1) -> C (k=0,1, i=0,1)
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
    
    # Block 3: A_left (j=0,1) * B_t2 (j=0,1, k=2,3) -> C (k=2,3, i=0,1)
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [2,3], [2,3], [0,1], 14)
    
    # Block 4: A_right (j=2,3) * B_b2 (j=2,3, k=2,3) -> C (k=2,3, i=0,1)
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [2,3], [2,3], [0,1], 21)
    
    # Block 5: A_left (j=0,1) * B_t3 (j=0,1, k=4,5) -> C (k=4,5, i=0,1)
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [4,5], [4,5], [0,1], 28)
    
    # Block 6: A_right (j=2,3) * B_b3 (j=2,3, k=4,5) -> C (k=4,5, i=0,1)
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [4,5], [4,5], [0,1], 35)
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    def check(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not check(U_list, V_list, W_list):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U_list, "V": V_list, "W": W_list}
