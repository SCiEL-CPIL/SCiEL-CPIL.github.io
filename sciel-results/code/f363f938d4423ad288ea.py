"""Structural change: use Strassen's algorithm on the 2x2 sub-block (first two of the 3 columns)
combined with schoolbook for the remaining column, exploiting block structure.

The matrix multiplication is (2x3) * (3x10) = 2x10.
We split the middle dimension m=3 into 2+1.
- The 2x2 block (rows 1-2 of A, cols 1-2 of B) can use Strassen with rank 7 per output column pair.
  Actually, let's think more carefully.

We have A: 2x3, B: 3x10.
Output C: 2x10.

Split B's columns into two groups: B1 (3x2) and B2 (3x8).
Split A's rows... no, A is 2x3, we split its columns: A1 (2x2) and A2 (2x1).

C = A1*B1 + A2*B2.

A1*B1: (2x2)*(2x2) -> can use Strassen rank 7.
A2*B2: (2x1)*(1x8) -> rank 2*1*8 = 16 schoolbook.

Total rank = 7 + 16 = 23.

Wait, but the tensor decomposition needs to be for the full 2x3 by 3x10 multiplication.
Let me reconsider. The standard schoolbook is rank n*m*p = 2*3*10 = 60.

Using block structure:
C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,9}

Split j into j1 in {0,1} and j2 = {2}.
C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + A[i,2]*B[2,k]

The first part is (2x2)*(2x2) for each output column k, but Strassen works on 2x2 matrix multiply.
Actually, we can view the 2x2 part as: for each k, we compute a 2-vector from A[:,0:2] and B[0:2,k].

Better approach: Group the output columns. We have 10 output columns.
For the 2x2 sub-matrix multiplication (A1: 2x2, B1: 2x10), we can use Strassen per 2x2 block of B1.
But B1 is 2x10, so we have 5 blocks of 2x2 if we group columns in pairs... no, that's not how Strassen works.

Actually, the cleanest approach: 
- Treat the problem as C = A * B where A is 2x3, B is 3x10.
- Partition the inner dimension: j = 0,1,2.
- For j=0,1: this is a (2x2) x (2x10) multiplication. We can use Strassen's 7-multiplication scheme for 2x2 matrices, but B is 2x10 not 2x2.

Let me think of it differently. The tensor for (2x3)x(3x10) has rank at most 60 (schoolbook).
We can do better by noting that the 2x2 sub-problem (using only j=0,1) has a known rank-7 decomposition.

For the full problem:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]

The first two terms form a (2x2)x(2x10) product. We can decompose this using Strassen's algorithm applied column-wise, but Strassen is for 2x2 * 2x2.

Alternative: Use the fact that (2x2) x (2xn) multiplication has rank 7*ceil(n/2) if we block the columns... no, that's not right either.

Actually, the simplest structural improvement: 
- The sub-problem (2x2) x (2x10) can be done with rank 7*5 = 35 if we use Strassen on pairs of columns.
  Wait, no. Let me use a known result: the rank of (2x2)x(2xn) matrix multiplication is 7*ceil(n/2) for even n... actually it's more nuanced.

Let me just use a straightforward block approach:
- Split B into B_top (2x10) and B_bottom (1x10).
- Split A into A_left (2x2) and A_right (2x1).
- C = A_left * B_top + A_right * B_bottom.

For A_right * B_bottom: (2x1)*(1x10) = rank 20 schoolbook.
For A_left * B_top: (2x2)*(2x10). 

For (2x2)*(2x10), we can use Strassen on each pair of columns. Split B_top into 5 blocks of 2x2.
Each 2x2 * 2x2 uses rank 7. Total: 5*7 = 35.

Total rank: 35 + 20 = 55.

This is better than 60. Let me implement this.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W uses trace convention: W[k*n + i, r]
    
    # We'll build the decomposition in two parts:
    # Part 1: (2x2) x (2x10) using Strassen on 5 column-pairs, rank 35
    # Part 2: (2x1) x (1x10) using schoolbook, rank 20
    # Total rank: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a10)*b11
    # M6 = (a11-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Part 1: A_left (2x2) * B_top (2x10)
    # B_top columns 0..9, split into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    col_offset = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        # For each of the 7 Strassen multiplications
        # We need to set up U, V, W entries
        # 
        # The 7 products are:
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # M6 = (A[1,1]-A[0,0]) * (B[0,k0]+B[1,k0])
        # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
        
        # Output contributions:
        # C[0,k0] += M1 + M4 - M5 + M7
        # C[0,k1] += M3 + M5
        # C[1,k0] += M2 + M4
        # C[1,k1] += M1 - M2 + M3 + M6
        
        strassen_defs = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: dict from (i,j) to coefficient for A[i,j]
            # V_coeffs: dict from (j,k) to coefficient for B[j,k]
            # W_coeffs: dict from (k,i) to coefficient for C[i,k]
            
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,k0): 1, (1,k1): 1}, 
             {(k0,0): 1, (k1,0): 0, (k0,1): 1, (k1,1): 0}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,k0): 1}, 
             {(k0,0): 0, (k1,0): 0, (k0,1): 1, (k1,1): 0}),
            # M3
            ({(0,0): 1}, {(0,k1): 1, (1,k1): -1}, 
             {(k0,0): 0, (k1,0): 0, (k0,1): 0, (k1,1): 1}),
            # M4
            ({(1,1): 1}, {(1,k0): 1, (0,k0): -1}, 
             {(k0,0): 1, (k1,0): 0, (k0,1): 1, (k1,1): 0}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,k1): 1}, 
             {(k0,0): -1, (k1,0): 0, (k0,1): 0, (k1,1): 0}),
            # M6
            ({(1,1): 1, (0,0): -1}, {(0,k0): 1, (1,k0): 1}, 
             {(k0,0): 0, (k1,0): 0, (k0,1): 0, (k1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(0,k1): 1, (1,k1): 1}, 
             {(k0,0): 1, (k1,0): 0, (k0,1): 0, (k1,1): 0}),
        ]
        
        for u_coeff, v_coeff, w_coeff in strassen_defs:
            # Set U entries: U[i*m + j, col_offset]
            for (i, j), coeff in u_coeff.items():
                if i < 2 and j < 2:  # Only the top-left 2x2 block of A
                    U[i * m + j][col_offset] += coeff
            
            # Set V entries: V[j*p + k, col_offset]
            for (j, k), coeff in v_coeff.items():
                if j < 2:  # Only the top 2 rows of B
                    V[j * p + k][col_offset] += coeff
            
            # Set W entries: W[k*n + i, col_offset]
            for (k, i), coeff in w_coeff.items():
                W[k * n + i][col_offset] += coeff
            
            col_offset += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x10)
    # A_right is A[:, 2], B_bottom is B[2, :]
    # C[i,k] += A[i,2] * B[2,k]
    # Schoolbook: for each i in {0,1}, k in {0..9}
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col_offset] = 1  # A[i, 2]
            V[2 * p + k][col_offset] = 1  # B[2, k]
            W[k * n + i][col_offset] = 1  # C[i, k]
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
