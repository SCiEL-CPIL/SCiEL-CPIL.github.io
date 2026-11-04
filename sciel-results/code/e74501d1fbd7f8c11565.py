"""
Program to construct a rank-minimal exact tensor decomposition for matrix multiplication.
Target: 3x4 matrix by 4x4 matrix.
Dimensions: n=3, m=4, p=4.
Schoolbook rank: 3*4*4 = 48.

Strategy: Use Strassen's rank-7 algorithm for 2x2 blocks where possible.
For the top 2 rows (i=0,1) of the output, we can apply Strassen to 2x2 sub-blocks.
For the bottom row (i=2), we use schoolbook.

Actually, let's think more carefully about the structure.
The matrix multiplication C = A * B where A is 3x4 and B is 4x4.
C[i,k] = sum_j A[i,j] * B[j,k]

We can decompose the computation into blocks.
Since n=3, m=4, p=4, we can view A as having 3 rows and B as having 4 rows and 4 columns.

Let's try a different approach: use the known result that the tensor rank of matrix multiplication
M_{n,m,p} is at most something less than n*m*p for certain dimensions.

For 2x2x2, the rank is 7 (Strassen).
For 3x4x4, we can potentially use block structure.

Let's try to use Strassen's algorithm on 2x2 blocks.
We can partition A into blocks: A_top (2x4) and A_bottom (1x4).
We can partition B into blocks: B_left (4x2) and B_right (4x2).

Then C = A * B = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]

The top-left 2x2 block: A_top * B_left. Here A_top is 2x4, B_left is 4x2.
This is not directly a 2x2 multiplication.

Let me think differently. We can use the fact that:
rank(M_{3,4,4}) <= rank(M_{2,2,2}) * something + remaining terms.

Actually, let's just implement a known decomposition.

For 3x4x4, one approach:
- Use 2x2 Strassen for the top 2 rows with the first 2 columns of B, giving 7 terms for C[0:2, 0:2]
- But this doesn't directly work because A has 4 columns, not 2.

Let me try a cleaner approach. We'll use the following strategy:
Partition the computation into 2x2 blocks where possible.

A is 3x4, B is 4x4, C is 3x4.

We can write C as:
C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
C[0:2, 2:4] = A[0:2, 0:4] * B[0:4, 2:4]
C[2:3, 0:2] = A[2:3, 0:4] * B[0:4, 0:2]
C[2:3, 2:4] = A[2:3, 0:4] * B[0:4, 2:4]

Each of these is a (2x4)*(4x2) or (1x4)*(4x2) multiplication.

For (2x4)*(4x2): We can view this as computing 2x2 output from 2x4 and 4x2 inputs.
This can be done with rank... let's think.

Actually, (2x4)*(4x2) can be computed by first splitting the 4 into 2+2:
A[0:2, 0:4] = [A11, A12] where A11, A12 are 2x2
B[0:4, 0:2] = [B21; B22] where B21, B22 are 2x2

Then A*B = A11*B21 + A12*B22

Each 2x2*2x2 can be done with rank 7 (Strassen), so total rank 14 for this block.

Similarly for other blocks.

Total:
C[0:2, 0:2]: rank 14
C[0:2, 2:4]: rank 14
C[2:3, 0:2]: (1x4)*(4x2) - this is simpler, rank 8 (schoolbook) or less
C[2:3, 2:4]: (1x4)*(4x2) - rank 8

Total: 14 + 14 + 8 + 8 = 44.

Let me implement this.

For (2x4)*(4x2) using two 2x2 Strassen multiplications:
Let A = [A11 A12] (2x4), B = [B21; B22] (4x2)
C = A11*B21 + A12*B22

For A11*B21 (2x2*2x2), use Strassen with 7 terms.
For A12*B22 (2x2*2x2), use Strassen with 7 terms.

For (1x4)*(4x2):
Let A = [a0 a1 a2 a3] (1x4), B = [b00 b01; b10 b11; b20 b21; b30 b31] (4x2)
C = [a0*b00 + a1*b10 + a2*b20 + a3*b30, a0*b01 + a1*b11 + a2*b21 + a3*b31]
This is 8 terms (schoolbook).

So total rank = 14 + 14 + 8 + 8 = 44.

Let me code this up.
"""

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication C = A * B.
    
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    # We'll build the decomposition using Strassen for 2x2 blocks
    # Total rank will be 44
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # For matrices X, Y, Z (2x2 each), Z = X*Y:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y01+Y11)
    # Z00 = M1 - M4 + M5 + M7
    # Z01 = M3 + M4
    # Z10 = M2 + M6
    # Z11 = M1 - M3 + M6 + M7
    
    # We'll create a helper to add a Strassen block
    # Parameters: 
    #   A_row_offset, A_col_offset: offset in A (2x2 block)
    #   B_row_offset, B_col_offset: offset in B (2x2 block)
    #   C_row_offset, C_col_offset: offset in C (2x2 block)
    #   U, V, W: the matrices we're building
    #   rank_offset: current rank offset
    
    def add_strassen_block(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, U, V, W, rank_offset):
        """Add 7 Strassen terms for a 2x2 block multiplication."""
        # A is 3x4, B is 4x4, C is 3x4
        # U index: i*m + j where i is row of A, j is col of A
        # V index: j*p + k where j is row of B, k is col of B
        # W index: k*n + i where k is col of C, i is row of C
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            [(A_r0, A_c0), (A_r0+1, A_c0+1)],  # A factors
            [(B_r0, B_c0), (B_r0+1, B_c0+1)],  # B factors
            [(C_r0, C_c0), (C_r0+1, C_c0+1)],  # C factors
            [1, 1, -1, -1]  # signs: C00 += M1, C11 += M1
        ))
        
        # M2 = (A10+A11)B00
        terms.append((
            [(A_r0+1, A_c0), (A_r0+1, A_c0+1)],
            [(B_r0, B_c0)],
            [(C_r0+1, C_c0)],
            [1, 1, 1]  # C10 += M2
        ))
        
        # M3 = A00(B10-B11)
        terms.append((
            [(A_r0, A_c0)],
            [(B_r0+1, B_c0), (B_r0+1, B_c0+1)],
            [(C_r0, C_c0+1), (C_r0+1, C_c0+1)],
            [1, 1, 1, -1]  # C01 += M3, C11 += -M3
        ))
        
        # M4 = A11(B01-B00)
        terms.append((
            [(A_r0+1, A_c0+1)],
            [(B_r0, B_c0+1), (B_r0, B_c0)],
            [(C_r0, C_c0), (C_r0+1, C_c0)],
            [-1, -1, 1, 1]  # C00 += -M4, C10 += M4
        ))
        
        # M5 = (A00+A01)B11
        terms.append((
            [(A_r0, A_c0), (A_r0, A_c0+1)],
            [(B_r0+1, B_c0+1)],
            [(C_r0, C_c0), (C_r0, C_c0+1)],
            [-1, 1, 1, 1]  # C00 += -M5, C01 += M5
        ))
        
        # M6 = (A10-A00)(B00+B10)
        terms.append((
            [(A_r0+1, A_c0), (A_r0, A_c0)],
            [(B_r0, B_c0), (B_r0+1, B_c0)],
            [(C_r0+1, C_c0+1)],
            [1, -1, 1, 1, 1]  # C11 += M6
        ))
        
        # M7 = (A01-A11)(B01+B11)
        terms.append((
            [(A_r0, A_c0+1), (A_r0+1, A_c0+1)],
            [(B_r0, B_c0+1), (B_r0+1, B_c0+1)],
            [(C_r0, C_c0)],
            [1, -1, 1, 1, 1]  # C00 += M7
        ))
        
        for a_factors, b_factors, c_factors, signs in terms:
            r = rank_offset
            # Add to U
            for (ai, aj) in a_factors:
                idx = ai * m + aj
                U[idx][r] = 1.0
            # Add to V
            for (bi, bj) in b_factors:
                idx = bi * p + bj
                V[idx][r] = 1.0
            # Add to W
            for (ci, ck), sign in zip(c_factors, signs):
                idx = ck * n + ci
                W[idx][r] = sign
            rank_offset += 1
        
        return rank_offset
    
    def add_schoolbook_block(A_r0, A_c0, A_rows, A_cols, B_r0, B_c0, B_rows, B_cols, C_r0, C_c0, U, V, W, rank_offset):
        """Add schoolbook terms for a block multiplication."""
        for i in range(A_rows):
            for j in range(A_cols):
                for k in range(B_cols):
                    r = rank_offset
                    # U: A[i, j]
                    u_idx = (A_r0 + i) * m + (A_c0 + j)
                    U[u_idx][r] = 1.0
                    # V: B[j, k]
                    v_idx = (B_r0 + j) * p + (B_c0 + k)
                    V[v_idx][r] = 1.0
                    # W: C[i, k]
                    w_idx = (C_c0 + k) * n + (C_r0 + i)
                    W[w_idx][r] = 1.0
                    rank_offset += 1
        return rank_offset
    
    # Total rank: 14 + 14 + 8 + 8 = 44
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Block 1: C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
    # Split into A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    rank_offset = add_strassen_block(0, 0, 0, 0, 0, 0, U, V, W, rank_offset)  # 7 terms
    rank_offset = add_strassen_block(0, 2, 2, 0, 0, 0, U, V, W, rank_offset)  # 7 terms
    
    # Block 2: C[0:2, 2:4] = A[0:2, 0:4] * B[0:4, 2:4]
    # Split into A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    rank_offset = add_strassen_block(0, 0, 0, 2, 0, 2, U, V, W, rank_offset)  # 7 terms
    rank_offset = add_strassen_block(0, 2, 2, 2, 0, 2, U, V, W, rank_offset)  # 7 terms
    
    # Block 3: C[2:3, 0:2] = A[2:3, 0:4] * B[0:4, 0:2]
    # Schoolbook: 1*4*2 = 8 terms
    rank_offset = add_schoolbook_block(2, 0, 1, 4, 0, 0, 4, 2, 2, 0, U, V, W, rank_offset)
    
    # Block 4: C[2:3, 2:4] = A[2:3, 0:4] * B[0:4, 2:4]
    # Schoolbook: 1*4*2 = 8 terms
    rank_offset = add_schoolbook_block(2, 0, 1, 4, 0, 2, 4, 2, 2, 2, U, V, W, rank_offset)
    
    return {"U": U, "V": V, "W": W}

