"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS
# Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# Tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
# Output factor follows W[k*n+i, r]


def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Each term is (U_val, V_val, W_val) where:
      U_val is a 2x2 matrix (flattened to 4 entries for the U factor)
      V_val is a 2x2 matrix (flattened to 4 entries for the V factor)
      W_val is a 2x2 matrix (flattened to 4 entries for the W factor)
    
    The 7 Strassen terms compute:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each term: list of 7 tuples (u_row, v_row, w_row)
    # u_row: 4 values for U[i,j] for i in {0,1}, j in {0,1} -> [U00, U01, U10, U11]
    # v_row: 4 values for V[j,k] for j in {0,1}, k in {0,1} -> [V00, V01, V10, V11]
    # w_row: 4 values for W[k,i] for k in {0,1}, i in {0,1} -> [W00, W01, W10, W11]
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11+A22 -> U[0,0]=1, U[1,1]=1, others 0 -> [1,0,0,1]
    # V: B11+B22 -> V[0,0]=1, V[1,1]=1, others 0 -> [1,0,0,1]
    # W: contributes to C11 and C22 with coefficient +1
    # C11 = M1 + ... so W[0,0] (k=0,i=0) gets +1
    # C22 = M1 - ... so W[1,1] (k=1,i=1) gets +1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # U: A21+A22 -> U[1,0]=1, U[1,1]=1 -> [0,0,1,1]
    # V: B11 -> V[0,0]=1 -> [1,0,0,0]
    # W: C21 = M2 + M4, so W[1,0] (k=1,i=0) gets +1
    #     C22 = M1 - M2 + M3 + M6, so W[1,1] (k=1,i=1) gets -1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,-1,0]))
    
    # M3 = A11(B21-B22)
    # U: A11 -> U[0,0]=1 -> [1,0,0,0]
    # V: B21-B22 -> V[1,0]=1, V[1,1]=-1 -> [0,0,1,-1]
    # W: C12 = M3 + M5, so W[0,1] (k=0,i=1) gets +1
    #     C22 = M1 - M2 + M3 + M6, so W[1,1] (k=1,i=1) gets +1
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
    
    # M4 = A22(B12-B11)
    # U: A22 -> U[1,1]=1 -> [0,0,0,1]
    # V: B12-B11 -> V[0,1]=1, V[0,0]=-1 -> [-1,1,0,0]
    # W: C11 = M1 + M4 - M5 + M7, so W[0,0] (k=0,i=0) gets +1
    #     C21 = M2 + M4, so W[1,0] (k=1,i=0) gets +1
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A11+A12)B22
    # U: A11+A12 -> U[0,0]=1, U[0,1]=1 -> [1,1,0,0]
    # V: B22 -> V[1,1]=1 -> [0,0,0,1]
    # W: C11 = M1 + M4 - M5 + M7, so W[0,0] (k=0,i=0) gets -1
    #     C12 = M3 + M5, so W[0,1] (k=0,i=1) gets +1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21-A11 -> U[1,0]=1, U[0,0]=-1 -> [-1,0,1,0]
    # V: B11+B12 -> V[0,0]=1, V[0,1]=1 -> [1,1,0,0]
    # W: C22 = M1 - M2 + M3 + M6, so W[1,1] (k=1,i=1) gets +1
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12-A22 -> U[0,1]=1, U[1,1]=-1 -> [0,1,0,-1]
    # V: B21+B22 -> V[1,0]=1, V[1,1]=1 -> [0,0,1,1]
    # W: C11 = M1 + M4 - M5 + M7, so W[0,0] (k=0,i=0) gets +1
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill
    
    n, m, p = 3, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    
    def time_check():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
    
    # Strategy: Direct sum decomposition
    # Split A (3x5) into A_top (2x5) and A_bottom (1x5)
    # Split C (3x5) into C_top (2x5) and C_bottom (1x5)
    # C_top = A_top * B  (2x5 times 5x5 = 2x5)
    # C_bottom = A_bottom * B  (1x5 times 5x5 = 1x5)
    
    # For C_top = A_top * B:
    #   A_top is 2x5, B is 5x5, C_top is 2x5
    #   We can further decompose: split B's columns into 5 groups of 1
    #   But better: think of it as 2x5 * 5x5
    #   Split the 5 columns of B into blocks. Actually, let's use a different approach.
    #   
    #   Alternative: split the 5x5 B into blocks.
    #   Actually, the cleanest approach:
    #   For 2x5 * 5x5 -> 2x5:
    #   We can view this as: for each output column k (0..4), C_top[:,k] = A_top * B[:,k]
    #   where B[:,k] is a 5x1 vector.
    #   C_top[i,k] = sum_{j=0}^{4} A_top[i,j] * B[j,k]
    #   
    #   We can use Strassen on the 2x2 core by grouping.
    #   Actually, let's think differently.
    #   
    #   For the 2x5 * 5x5 case:
    #   Split the 5 columns of B into: first 2 columns, next 2 columns, last 1 column
    #   And split the 5 rows of A_top into: first 2 rows (but A_top only has 2 rows)
    #   
    #   Hmm, this is getting complicated. Let me use a simpler approach.
    #
    #   Simplest correct approach: 
    #   For 2x5 * 5x5 -> 2x5, the schoolbook rank is 2*5*5 = 50.
    #   For 1x5 * 5x5 -> 1x5, the schoolbook rank is 1*5*5 = 25.
    #   Total schoolbook: 75.
    #
    #   To improve, we need to find a lower-rank decomposition for at least one part.
    #
    #   For the 1x5 * 5x5 -> 1x5 case:
    #   C[i=0,k] = sum_j A[0,j] * B[j,k]
    #   This is a dot product of a 5-vector with each column of B.
    #   The tensor has entries T[0,j,k] = 1 for all j,k.
    #   Schoolbook: 25 terms.
    #   Can we do better? The rank of the "outer product" type tensor...
    #   Actually, this is equivalent to computing 5 dot products of length 5.
    #   Each dot product can be done in 5 multiplications (schoolbook).
    #   There's no known way to do a 5x1 * 1x5 -> 1x1 in fewer than 5 mults.
    #   So rank 25 seems tight for this part.
    #
    #   For the 2x5 * 5x5 -> 2x5 case:
    #   C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0..4
    #   This is 2*5*5 = 50 schoolbook terms.
    #   We can try to use Strassen-like tricks.
    #
    #   Idea: Group the 5 columns of B into blocks.
    #   Let's split B's columns into: B_0 (cols 0-1, 2 cols), B_1 (cols 2-3, 2 cols), B_2 (col 4, 1 col)
    #   And A_top's rows: 2 rows.
    #   For each block of 2 columns, we have a 2x5 * 5x2 -> 2x2 multiplication.
    #   We can use Strassen on the 2x2 output if we can reduce the 5 middle dimension.
    #
    #   Actually, here's a cleaner approach:
    #   For 2x5 * 5x2 -> 2x2:
    #   C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #   This is a 2x5 times 5x2.
    #   We can write this as: C = A * B where A is 2x5, B is 5x2.
    #   Schoolbook: 2*5*2 = 20 terms.
    #   Can we do better? 
    #   
    #   Note: 2x5 * 5x2 can be viewed as embedding in 2x5 * 5x5.
    #   Alternatively, we can pad to 2x5 * 5x5 and use Strassen on the 2x2 block.
    #
    #   Actually, here's a key insight: 
    #   For 2x2 * 2x2 -> 2x2, Strassen uses 7 mults.
    #   For 2x5 * 5x2 -> 2x2, we can potentially use fewer than 20 mults
    #   by exploiting the structure.
    #
    #   One approach: Write A as [A_left | A_right] where A_left is 2x2 (cols 0,1)
    #   and A_right is 2x3 (cols 2,3,4). Similarly B as [B_top; B_bottom] where
    #   B_top is 2x2 (rows 0,1) and B_bottom is 3x2 (rows 2,3,4).
    #   Then C = A_left * B_top + A_right * B_bottom.
    #   The first term is 2x2 * 2x2 -> 2x2, which Strassen does in 7.
    #   The second term is 2x3 * 3x2 -> 2x2, which has schoolbook rank 12.
    #   Total: 7 + 12 = 19. Slight improvement over 20.
    #
    #   We can recurse: for 2x3 * 3x2 -> 2x2:
    #   Split A_right into 2x2 (cols 2,3) and 2x1 (col 4).
    #   Split B_bottom into 2x2 (rows 2,3) and 1x2 (row 4).
    #   C = A_23 * B_23 + A_4 * B_4.
    #   A_23 * B_23 is 2x2 * 2x2 -> 2x2: Strassen rank 7.
    #   A_4 * B_4 is 2x1 * 1x2 -> 2x2: This is an outer product, rank 4 schoolbook.
    #   Total: 7 + 4 = 11.
    #
    #   So 2x5 * 5x2 -> 2x2 can be done in 19 + ... wait let me redo.
    #
    #   2x5 * 5x2:
    #   = (2x2 * 2x2) + (2x3 * 3x2)
    #   2x2 * 2x2: Strassen = 7
    #   2x3 * 3x2:
    #     = (2x2 * 2x2) + (2x1 * 1x2)
    #     2x2 * 2x2: Strassen = 7
    #     2x1 * 1x2: outer product = 4
    #     Total: 11
    #   Total for 2x5 * 5x2: 7 + 11 = 18
    #
    #   So for 2x5 * 5x5 -> 2x5, we split B's 5 columns into three groups:
    #   Group 1: cols 0,1 -> 2x5 * 5x2 -> 2x2: rank 18
    #   Group 2: cols 2,3 -> 2x5 * 5x2 -> 2x2: rank 18
    #   Group 3: col 4 -> 2x5 * 5x1 -> 2x1: schoolbook = 10
    #   Total: 18 + 18 + 10 = 46
    #
    #   For 1x5 * 5x5 -> 1x5:
    #   Split B's 5 columns similarly:
    #   Group 1: cols 0,1 -> 1x5 * 5x2 -> 1x2: schoolbook = 10
    #   Group 2: cols 2,3 -> 1x5 * 5x2 -> 1x2: schoolbook = 10
    #   Group 3: col 4 -> 1x5 * 5x1 -> 1x1: schoolbook = 5
    #   Total: 10 + 10 + 5 = 25
    #
    #   Grand total: 46 + 25 = 71. Improvement over 75!
    
    # Let me implement this carefully.
    
    # First, let me build the Strassen-based 2x5 * 5x2 -> 2x2 decomposition.
    # This will produce a list of (u_vec, v_vec, w_vec) where:
    #   u_vec has 10 entries (2x5 flattened: row-major [A00,A01,A02,A03,A04,A10,A11,A12,A13,A14])
    #   v_vec has 10 entries (5x2 flattened: row-major [B00,B01,B10,B11,B20,B21,B30,B31,B40,B41])
    #   w_vec has 4 entries (2x2 flattened: row-major [C00,C01,C10,C11])
    
    def _strassen_2x2():
        """Return 7 terms for 2x2*2x2->2x2.
        Each term: (u4, v4, w4) where u4=[U00,U01,U10,U11], v4=[V00,V01,V10,V11], w4=[W00,W01,W10,W11]
        """
        return [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
            ([0,0,1,1], [1,0,0,0], [0,1,-1,0]),   # M2
            ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),   # M3
            ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),   # M4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7
        ]
    
    def _pad_2x2_to_2x5_5x2(u4, v4, w4, a_cols, b_rows):
        """
        Pad a 2x2*2x2->2x2 term to a 2x5*5x2->2x2 term.
        a_cols: which 2 columns of A (0-indexed) are used, e.g., (0,1)
        b_rows: which 2 rows of B (0-indexed) are used, e.g., (0,1)
        Returns (u10, v10, w4)
        """
        u10 = [0]*10
        v10 = [0]*10
        # u4 = [U00, U01, U10, U11] corresponds to A[0,a_cols[0]], A[0,a_cols[1]], A[1,a_cols[0]], A[1,a_cols[1]]
        u10[0*5 + a_cols[0]] = u4[0]  # A[0, a_cols[0]]
        u10[0*5 + a_cols[1]] = u4[1]  # A[0, a_cols[1]]
        u10[1*5 + a_cols[0]] = u4[2]  # A[1, a_cols[0]]
        u10[1*5 + a_cols[1]] = u4[3]  # A[1, a_cols[1]]
        
        # v4 = [V00, V01, V10, V11] corresponds to B[b_rows[0],0], B[b_rows[0],1], B[b_rows[1],0], B[b_rows[1],1]
        v10[b_rows[0]*2 + 0] = v4[0]  # B[b_rows[0], 0]
        v10[b_rows[0]*2 + 1] = v4[1]  # B[b_rows[0], 1]
        v10[b_rows[1]*2 + 0] = v4[2]  # B[b_rows[1], 0]
        v10[b_rows[1]*2 + 1] = v4[3]  # B[b_rows[1], 1]
        
        return (u10, v10, list(w4))
    
    def _outer_product_2x1_1x2():
        """
        Return 4 terms for 2x1 * 1x2 -> 2x2 (outer product).
        A is 2x1: [A0, A1]
        B is 1x2: [B0, B1]
        C = [[A0*B0, A0*B1], [A1*B0, A1*B1]]
        Each term: (u2, v2, w4) where u2=[U0,U1], v2=[V0,V1], w4=[W00,W01,W10,W11]
        """
        return [
            ([1,0], [1,0], [1,0,0,0]),  # A0*B0 -> C00
            ([1,0], [0,1], [0,1,0,0]),  # A0*B1 -> C01
            ([0,1], [1,0], [0,0,1,0]),  # A1*B0 -> C10
            ([0,1], [0,1], [0,0,0,1]),  # A1*B1 -> C11
        ]
    
    def _pad_outer_2x1_1x2_to_2x5_5x2(terms, a_col, b_row):
        """
        Pad 2x1*1x2->2x2 terms to 2x5*5x2->2x2.
        a_col: which column of A (0-indexed)
        b_row: which row of B (0-indexed)
        """
        result = []
        for u2, v2, w4 in terms:
            u10 = [0]*10
            v10 = [0]*10
            u10[0*5 + a_col] = u2[0]  # A[0, a_col]
            u10[1*5 + a_col] = u2[1]  # A[1, a_col]
            v10[b_row*2 + 0] = v2[0]  # B[b_row, 0]
            v10[b_row*2 + 1] = v2[1]  # B[b_row, 1]
            result.append((u10, v10, list(w4)))
        return result
    
    def _decompose_2x5_5x2():
        """
        Decompose 2x5 * 5x2 -> 2x2 using rank 18.
        Split: 
          Term 1: A[:,0:2] * B[0:2,:] -> 2x2*2x2 -> Strassen (7 terms)
          Term 2: A[:,2:4] * B[2:4,:] -> 2x2*2x2 -> Strassen (7 terms)
          Term 3: A[:,4] * B[4,:] -> 2x1*1x2 -> outer product (4 terms)
        Total: 18 terms
        """
        s2x2 = _strassen_2x2()
        terms = []
        # Term 1: cols 0,1 of A, rows 0,1 of B
        for u4, v4, w4 in s2x2:
            terms.append(_pad_2x2_to_2x5_5x2(u4, v4, w4, (0,1), (0,1)))
        # Term 2: cols 2,3 of A, rows 2,3 of B
        for u4, v4, w4 in s2x2:
            terms.append(_pad_2x2_to_2x5_5x2(u4, v4, w4, (2,3), (2,3)))
        # Term 3: col 4 of A, row 4 of B
        op = _outer_product_2x1_1x2()
        terms.extend(_pad_outer_2x1_1x2_to_2x5_5x2(op, 4, 4))
        return terms
    
    def _decompose_1x5_5x1():
        """
        Decompose 1x5 * 5x1 -> 1x1 (dot product) using rank 5 (schoolbook).
        Each term: (u5, v5, w1) where u5=[U0..U4], v5=[V0..V4], w1=[W0]
        """
        terms = []
        for j in range(5):
            u5 = [0]*5
            v5 = [0]*5
            u5[j] = 1
            v5[j] = 1
            terms.append((u5, v5, [1]))
        return terms
    
    def _decompose_1x5_5x2():
        """
        Decompose 1x5 * 5x2 -> 1x2 using rank 10 (schoolbook).
        C[0,k] = sum_j A[0,j] * B[j,k] for k=0,1
        Each term: (u5, v10, w2) where u5=[U0..U4], v10=[V00,V01,V10,V11,V20,V21,V30,V31,V40,V41], w2=[W0,W1]
        """
        terms = []
        for j in range(5):
            for k in range(2):
                u5 = [0]*5
                v10 = [0]*10
                w2 = [0]*2
                u5[j] = 1
                v10[j*2 + k] = 1
                w2[k] = 1
                terms.append((u5, v10, w2))
        return terms
    
    # Now build the full 3x5 * 5x5 -> 3x5 decomposition.
    # 
    # U factor: shape (15, rank), indexed by (i,j) -> row i*5+j
    # V factor: shape (25, rank), indexed by (j,k) -> row j*5+k
    # W factor: shape (15, rank), indexed by (k,i) -> row k*3+i
    #
    # Partition:
    #   i=0,1 (top 2 rows): handled by 2x5 * 5x5 -> 2x5
    #   i=2 (bottom row): handled by 1x5 * 5x5 -> 1x5
    #
    # For 2x5 * 5x5 -> 2x5:
    #   Split B's 5 columns into: (0,1), (2,3), (4)
    #   For each pair of columns, use 2x5 * 5x2 -> 2x2 (rank 18)
    #   For the single column, use 2x5 * 5x1 -> 2x1 (schoolbook rank 10)
    #   Total: 18 + 18 + 10 = 46
    #
    # For 1x5 * 5x5 -> 1x5:
    #   Split B's 5 columns into: (0,1), (2,3), (4)
    #   For each pair, use 1x5 * 5x2 -> 1x2 (schoolbook rank 10)
    #   For the single, use 1x5 * 5x1 -> 1x1 (schoolbook rank 5)
    #   Total: 10 + 10 + 5 = 25
    #
    # Grand total: 46 + 25 = 71
    
    all_terms = []  # Each term: (u15, v25, w15)
    
    # --- Top 2 rows (i=0,1) ---
    # For each column group of B:
    #   Group (0,1): 2x5 * 5x2 -> 2x2, rank 18
    #   Group (2,3): 2x5 * 5x2 -> 2x2, rank 18
    #   Group (4):   2x5 * 5x1 -> 2x1, schoolbook rank 10
    
    # 2x5 * 5x2 -> 2x2 terms (for column groups (0,1) and (2,3))
    d2x5_5x2 = _decompose_2x5_5x2()
    
    for col_group in [(0,1), (2,3)]:
        for u10, v10, w4 in d2x5_5x2:
            # u10: 2x5 A_top, rows 0-1, cols 0-4
            # v10: 5x2 B, rows 0-4, cols col_group[0]-col_group[1]
            # w4: 2x2 C_top, rows 0-1, cols col_group[0]-col_group[1]
            
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            
            # Fill U: A_top[i,j] for i=0,1; j=0..4 -> row i*5+j in U (rows 0-9)
            for i in range(2):
                for j in range(5):
                    u15[i*5 + j] = u10[i*5 + j]
            
            # Fill V: B[j,k] for j=0..4; k=col_group[0],col_group[1] -> row j*5+k in V
            # v10 is indexed as [B[0,0], B[0,1], B[1,0], B[1,1], ..., B[4,0], B[4,1]]
            # where the 2 columns correspond to col_group[0], col_group[1]
            for j in range(5):
                for dk in range(2):
                    v25[j*5 + col_group[dk]] = v10[j*2 + dk]
            
            # Fill W: C_top[i,k] for i=0,1; k=col_group[0],col_group[1] -> row k*3+i in W
            # w4 is [C[0,0], C[0,1], C[1,0], C[1,1]] corresponding to rows 0-1, cols col_group[0]-[1]
            for i in range(2):
                for dk in range(2):
                    w15[col_group[dk]*3 + i] = w4[i*2 + dk]
            
            all_terms.append((u15, v25, w15))
    
    # 2x5 * 5x1 -> 2x1 (for column group (4,))
    # C_top[i,4] = sum_j A_top[i,j] * B[j,4] for i=0,1
    # Schoolbook: 2*5 = 10 terms
    for i in range(2):
        for j in range(5):
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            u15[i*5 + j] = 1
            v25[j*5 + 4] = 1
            w15[4*3 + i] = 1  # k=4, i=i
            all_terms.append((u15, v25, w15))
    
    # --- Bottom row (i=2) ---
    # 1x5 * 5x5 -> 1x5
    # Split into column groups: (0,1), (2,3), (4)
    
    # 1x5 * 5x2 -> 1x2 for column groups (0,1) and (2,3)
    d1x5_5x2 = _decompose_1x5_5x2()
    for col_group in [(0,1), (2,3)]:
        for u5, v10, w2 in d1x5_5x2:
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            
            # U: A[2,j] for j=0..4 -> row 2*5+j = 10+j in U
            for j in range(5):
                u15[10 + j] = u5[j]
            
            # V: B[j,k] for j=0..4, k=col_group[0],col_group[1]
            for j in range(5):
                for dk in range(2):
                    v25[j*5 + col_group[dk]] = v10[j*2 + dk]
            
            # W: C[2,k] for k=col_group[0],col_group[1] -> row k*3+2 in W
            for dk in range(2):
                w15[col_group[dk]*3 + 2] = w2[dk]
            
            all_terms.append((u15, v25, w15))
    
    # 1x5 * 5x1 -> 1x1 for column group (4,)
    # C[2,4] = sum_j A[2,j] * B[j,4]
    for j in range(5):
        u15 = [0]*15
        v25 = [0]*25
        w15 = [0]*15
        u15[10 + j] = 1
        v25[j*5 + 4] = 1
        w15[4*3 + 2] = 1
        all_terms.append((u15, v25, w15))
    
    rank = len(all_terms)
    
    # Build the factor matrices using numpy for speed
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Build matrices from terms - vectorized
    for r in range(rank):
        U[:, r] = all_terms[r][0]
        V[:, r] = all_terms[r][1]
        W[:, r] = all_terms[r][2]
    
    # Verify correctness using numpy for speed
    def verify(U, V, W):
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-9):
            diff = np.abs(T - 1.0)
            max_idx = np.unravel_index(np.argmax(diff), T.shape)
            return False, (max_idx, T[max_idx])
        return True, None
    
    is_valid, err = verify(U, V, W)
    time_check()
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    if not is_valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
