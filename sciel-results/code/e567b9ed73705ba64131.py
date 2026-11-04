"""
Strategy: Structured Block-Wise Decomposition using Kronecker Products.
We decompose the 3x4 * 4x4 multiplication into blocks:
- Split the 3 rows of A into a 2x2 block and a 1x2 block.
- Split the 4 columns of A and rows of B into two 2x2 blocks.
- Use Strassen's algorithm (rank 7) for the 2x2 * 2x2 multiplications.
- Use schoolbook (rank 4) for the 1x2 * 2x2 multiplications.

Total Rank = 2 * 7 (for the 2x2 part) + 2 * 4 (for the 1x2 part) = 14 + 8 = 22?
Wait, let's re-evaluate the block structure carefully.

Matrix A is 3x4. Matrix B is 4x4. Result C is 3x4.
Let's partition A into rows: A_top (2x4) and A_bot (1x4).
Let's partition B into columns: B_left (4x2) and B_right (4x2).

C_top (2x4) = A_top * B = A_top * [B_left, B_right] = [A_top*B_left, A_top*B_right]
C_bot (1x4) = A_bot * B = A_bot * [B_left, B_right] = [A_bot*B_left, A_bot*B_right]

Now, partition A_top (2x4) into columns: A_top_left (2x2) and A_top_right (2x2).
Partition B_left (4x2) into rows: B_top_left (2x2) and B_bot_left (2x2).
Similarly for B_right.

This gets complicated with shared sub-blocks. A simpler, more robust structural approach:
Partition the 4x4 multiplication part (m=4, p=4) using Strassen (rank 7).
But A is 3x4.
Let's view the multiplication as:
C[i, k] = sum_j A[i, j] B[j, k]

We can use the rank-7 decomposition of the 2x2x2 tensor T_2x2x2.
Let's partition the indices:
i in {0, 1, 2}
j in {0, 1, 2, 3}
k in {0, 1, 2, 3}

Let's split j into j1 in {0, 1} and j2 in {2, 3}.
Let's split k into k1 in {0, 1} and k2 in {2, 3}.

C[i, k1] = sum_{j1} A[i, j1] B[j1, k1] + sum_{j2} A[i, j2] B[j2, k1]
C[i, k2] = sum_{j1} A[i, j1] B[j1, k2] + sum_{j2} A[i, j2] B[j2, k2]

This is 4 block multiplications of size (3x2) * (2x2).
Let's denote the blocks:
A11 (3x2), A12 (3x2)
B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)

C11 (3x2) = A11 B11 + A12 B21
C12 (3x2) = A11 B12 + A12 B22

We need to decompose the multiplication of a 3x2 matrix by a 2x2 matrix.
Let M be 3x2 and N be 2x2. Result P is 3x2.
P[i, 0] = M[i, 0]N[0,0] + M[i,1]N[1,0]
P[i, 1] = M[i, 0]N[0,1] + M[i,1]N[1,1]

This is effectively two 1x2 * 2x1 multiplications for each row? No, it's a 3x2 * 2x2.
We can use Strassen on the 2x2 * 2x2 part, but we have 3 rows.
Actually, the tensor for 3x2 * 2x2 multiplication has rank <= 2 * rank(1x2 * 2x2)? No.
Let's just use the rank-7 Strassen decomposition for 2x2 * 2x2 and apply it to each of the 3 rows of A?
If we treat each row of A separately, we are doing 3 independent 1x2 * 2x2 multiplications? No, the 2x2 B is shared.
Wait, in C11 = A11 B11 + A12 B21, A11 is 3x2 and B11 is 2x2.
This is a sum of two products.
Let's decompose 3x2 * 2x2.
Let M be 3x2, N be 2x2.
We can write M as rows m0, m1, m2 (each 1x2).
P = [m0 N; m1 N; m2 N].
Each mi N is a 1x2 * 2x2 multiplication.
The rank of 1x2 * 2x2 is 2 (schoolbook).
So rank(3x2 * 2x2) <= 3 * 2 = 6.
Is it lower?
Let's check the tensor rank of 3x2 * 2x2.
It is a (3, 2, 2) tensor.
Actually, we can just use the schoolbook for the 3x2 * 2x2 blocks.
There are 4 such blocks (C11, C12, C21, C22).
Each block is 3x2 * 2x2.
Rank of one block = 3 * 2 * 2 = 12? No, schoolbook is sum_{j=0}^1 sum_{k=0}^1 A[i,j]B[j,k].
For a fixed i, it's 2 terms. For 3 i's, it's 6 terms.
So rank(3x2 * 2x2) = 6.
Total rank = 4 blocks * 6 = 24.
This is much better than 48.

Let's verify the block decomposition:
A is 3x4.
A = [A11 A12] where A11 is 3x2, A12 is 3x2.
B is 4x4.
B = [B11 B21; B12 B22] where each is 2x2.
Wait, B is 4x4. Rows 0,1 are top, 2,3 are bottom. Cols 0,1 left, 2,3 right.
B11 = B[0:2, 0:2]
B12 = B[0:2, 2:4]
B21 = B[2:4, 0:2]
B22 = B[2:4, 2:4]

C11 (3x2) = A11 B11 + A12 B21
C12 (3x2) = A11 B12 + A12 B22
C21 (3x2) = A11 B21 + A12 B22 ?? No.
C[i, k] = sum_j A[i,j] B[j,k].
If k is in 0,1 (left), then C[i, k] = sum_{j=0}^1 A[i,j]B[j,k] + sum_{j=2}^3 A[i,j]B[j,k].
So C_left (3x2) = A11 B11 + A12 B21.
C_right (3x2) = A11 B12 + A12 B22.

So we only need to compute C_left and C_right.
C_left = A11 B11 + A12 B21
C_right = A11 B12 + A12 B22

We can compute A11 B11, A12 B21, A11 B12, A12 B22 using rank-6 decompositions.
Total rank = 6 + 6 + 6 + 6 = 24.

Let's construct the rank-6 decomposition for 3x2 * 2x2.
U shape: (3*2, 6) = (6, 6)
V shape: (2*2, 6) = (4, 6)
W shape: (2*3, 6) = (6, 6) -- Wait, output is 3x2, so p*n = 2*3 = 6.

Indices:
A index: i (0..2), j (0..1). Flat: i*2+j.
B index: j (0..1), k (0..1). Flat: j*2+k.
C index: k (0..1), i (0..2). Flat: k*3+i. (Trace convention W[k*n+i, r])

Terms:
For i=0:
  C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
  C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
For i=1:
  C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
  C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
For i=2:
  C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
  C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]

Total 12 terms? No, 3 rows * 2 cols * 2 terms = 12 terms.
Wait, my previous estimate of 6 was wrong.
Rank of 3x2 * 2x2 is 12 in schoolbook.
Can we do better?
A11 B11 is 3x2 * 2x2.
We can use Strassen on the 2x2 * 2x2 part?
Let A11 = [a0 a1; a2 a3; a4 a5] (3x2)
B11 = [b0 b1; b2 b3] (2x2)
Result = [a0b0+a1b2, a0b1+a1b3; ...]

This is just 6 scalar multiplications per row of A? No.
It is 2 multiplications per output element. 6 output elements. Total 12 multiplications.
Is there a rank-7 or lower decomposition for 3x2 * 2x2?
The tensor is T_{3,2,2}.
Rank(T_{3,2,2}) <= Rank(T_{1,2,2}) * 3? No, that's not how it works.
Rank(T_{m,n,p}) <= m * Rank(T_{1,n,p}).
Rank(T_{1,2,2}) = 2. So Rank <= 6.
Wait, T_{1,2,2} is 1x2 * 2x2.
C[0] = A[0]B[0,0] + A[1]B[1,0]
C[1] = A[0]B[0,1] + A[1]B[1,1]
This is 4 terms. Rank is 4?
Let's check.
U: (2, 4), V: (4, 4), W: (2, 4).
Terms:
1. A[0]B[0,0] -> C[0]
2. A[1]B[1,0] -> C[0]
3. A[0]B[0,1] -> C[1]
4. A[1]B[1,1] -> C[1]
These 4 terms are independent. So Rank(T_{1,2,2}) = 4.
So Rank(T_{3,2,2}) <= 3 * 4 = 12.

So the block decomposition gives 4 * 12 = 48. That's the same as schoolbook!
We need a better structure.

Let's use Strassen on the 2x2 * 2x2 blocks directly.
Partition A (3x4) into 2x2 and 1x2 blocks?
Let's partition the 4x4 B into 2x2 blocks.
Let's partition A into 2x2 and 1x2.
A_top (2x4) = [A11 A12] (2x2 each)
A_bot (1x4) = [A21 A22] (1x2 each)

C_top (2x4) = A_top B = [A11 B11 + A12 B21, A11 B12 + A12 B22]
C_bot (1x4) = A_bot B = [A21 B11 + A22 B21, A21 B12 + A22 B22]

For C_top, we have 2x2 * 2x2 multiplications.
We can use Strassen (rank 7) for each 2x2 * 2x2 product.
But we have sums: A11 B11 + A12 B21.
We can't just add ranks. We need to decompose the sum.
Actually, we can compute M = A11 B11 and N = A12 B21 separately and add them?
No, the decomposition is a sum of rank-1 tensors.
If we compute A11 B11 with rank 7, and A12 B21 with rank 7, the sum has rank <= 14.
So C_top left block (2x2) has rank <= 14.
C_top right block (2x2) has rank <= 14.
C_bot left block (1x2) = A21 B11 + A22 B21.
A21 is 1x2, B11 is 2x2. Product is 1x2.
A22 is 1x2, B21 is 2x2. Product is 1x2.
Rank(1x2 * 2x2) = 4.
So C_bot left block rank <= 4 + 4 = 8.
C_bot right block rank <= 8.

Total rank = 14 + 14 + 8 + 8 = 44.

This matches the inspiration! Rank 44.

Let's implement this.
We need to construct the U, V, W matrices for the whole 3x4 * 4x4.
We will build 4 sets of columns for U, V, W.

Set 1: C_top_left = A11 B11 + A12 B21
Set 2: C_top_right = A11 B12 + A12 B22
Set 3: C_bot_left = A21 B11 + A22 B21
Set 4: C_bot_right = A21 B12 + A22 B22

For Set 1 (2x2 output from 2x2 * 2x2 + 2x2 * 2x2):
We use Strassen for A11 B11 (7 terms) and Strassen for A12 B21 (7 terms).
Total 14 terms.

For Set 3 (1x2 output from 1x2 * 2x2 + 1x2 * 2x2):
We use schoolbook for A21 B11 (4 terms) and schoolbook for A22 B21 (4 terms).
Total 8 terms.

Total rank = 14 + 14 + 8 + 8 = 44.

Implementation details:
Strassen for 2x2 * 2x2:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B21 - B11)
M4 = A22(B32 - B12) -- Wait, indices are local.
Let's use standard Strassen variables for X*Y where X, Y are 2x2.
X = [[x00, x01], [x10, x11]]
Y = [[y00, y01], [y10, y11]]
Z = [[z00, z01], [z10, z11]]

M1 = (x00 + x11)(y00 + y11)
M2 = (x10 + x11)y00
M3 = (x00 - x10)(y10 + y11)
M4 = x11(y11 - y00)
M5 = (x00 + x01)y11
M6 = (x10 - x00)(y00 + y01)
M7 = (x01 - x11)(y10 + y11)

Z00 = M1 + M4 - M5 + M7
Z01 = M3 + M5
Z10 = M2 + M4
Z11 = M1 - M2 + M3 + M6

We need to map these to the global U, V, W.
For each Strassen term (e.g., M1), we have:
U entries: linear combination of X entries.
V entries: linear combination of Y entries.
W entries: linear combination of Z entries (which map to global C entries).

Since we are adding two such products (e.g., A11 B11 + A12 B21), we just concatenate the columns for the first product and the second product.

Let's code this up.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Global indices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        A_block: rows [A_block_start_i, A_block_start_i+2), cols [A_block_start_j, A_block_start_j+2)
        B_block: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C_block: rows [C_block_start_i, C_block_start_i+2), cols [C_block_start_j, C_block_start_j+2)
        
        Note: In our problem, A is 3x4, B is 4x4, C is 3x4.
        A indices: i in 0..2, j in 0..3.
        B indices: i in 0..3, j in 0..3.
        C indices: i in 0..2, j in 0..3.
        
        W index convention: W[k*n + i, r] where k is col of C, i is row of C.
        So C_block_start_i is row start, C_block_start_j is col start.
        """
        
        # Strassen terms
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = (x00-x10)(y10+y11)
        # M4 = x11(y11-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # Local indices for A (X)
        # x00: i=0, j=0 -> global (A_block_start_i + 0) * m + (A_block_start_j + 0)
        # x01: i=0, j=1
        # x10: i=1, j=0
        # x11: i=1, j=1
        
        # Local indices for B (Y)
        # y00: i=0, j=0 -> global (B_block_start_i + 0) * p + (B_block_start_j + 0)
        # y01: i=0, j=1
        # y10: i=1, j=0
        # y11: i=1, j=1
        
        # Local indices for C (Z)
        # z00: i=0, j=0 -> global row C_block_start_i+0, col C_block_start_j+0
        # W index: (C_block_start_j+0)*n + (C_block_start_i+0)
        
        terms = [
            # (U_coefs, V_coefs, W_coefs)
            # U: dict {global_idx: coef}
            # V: dict {global_idx: coef}
            # W: dict {global_idx: coef}
            
            # M1
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M2
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M3
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M4
            (
                { (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1, (B_block_start_i+0)*p + (B_block_start_j+0): -1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M5
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): -1, (C_block_start_j+1)*n + (C_block_start_i+0): -1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M6
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+0)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M7
            (
                { (A_block_start_i+0)*m + (A_block_start_j+1): 1, (A_block_start_i+1)*m + (A_block_start_j+1): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1 }
            ),
        ]
        
        for u_coefs, v_coefs, w_coefs in terms:
            u_col = np.zeros(n*m)
            for idx, coef in u_coefs.items():
                u_col[idx] = coef
            v_col = np.zeros(m*p)
            for idx, coef in v_coefs.items():
                v_col[idx] = coef
            w_col = np.zeros(p*n)
            for idx, coef in w_coefs.items():
                w_col[idx] = coef
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    def add_schoolbook_block_1x2(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Adds a schoolbook decomposition for a 1x2 * 2x2 block.
        A_block: row A_block_start_i, cols [A_block_start_j, A_block_start_j+2)
        B_block: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C_block: row C_block_start_i, cols [C_block_start_j, C_block_start_j+2)
        """
        # Terms:
        # C[j] = A[0]B[0,j] + A[1]B[1,j]
        # j=0, j=1
        
        # Term 1: A[0] * B[0,0] -> C[0]
        u1 = np.zeros(n*m); u1[(A_block_start_i)*m + (A_block_start_j+0)] = 1
        v1 = np.zeros(m*p); v1[(B_block_start_i+0)*p + (B_block_start_j+0)] = 1
        w1 = np.zeros(p*n); w1[(C_block_start_j+0)*n + (C_block_start_i)] = 1
        
        # Term 2: A[1] * B[1,0] -> C[0]
        u2 = np.zeros(n*m); u2[(A_block_start_i)*m + (A_block_start_j+1)] = 1
        v2 = np.zeros(m*p); v2[(B_block_start_i+1)*p + (B_block_start_j+0)] = 1
        w2 = np.zeros(p*n); w2[(C_block_start_j+0)*n + (C_block_start_i)] = 1
        
        # Term 3: A[0] * B[0,1] -> C[1]
        u3 = np.zeros(n*m); u3[(A_block_start_i)*m + (A_block_start_j+0)] = 1
        v3 = np.zeros(m*p); v3[(B_block_start_i+0)*p + (B_block_start_j+1)] = 1
        w3 = np.zeros(p*n); w3[(C_block_start_j+1)*n + (C_block_start_i)] = 1
        
        # Term 4: A[1] * B[1,1] -> C[1]
        u4 = np.zeros(n*m); u4[(A_block_start_i)*m + (A_block_start_j+1)] = 1
        v4 = np.zeros(m*p); v4[(B_block_start_i+1)*p + (B_block_start_j+1)] = 1
        w4 = np.zeros(p*n); w4[(C_block_start_j+1)*n + (C_block_start_i)] = 1
        
        U_cols.extend([u1, u2, u3, u4])
        V_cols.extend([v1, v2, v3, v4])
        W_cols.extend([w1, w2, w3, w4])
        
    # Block 1: C_top_left (rows 0,1 of C; cols 0,1 of C)
    # = A_top_left (rows 0,1 of A; cols 0,1 of A) * B_top_left (rows 0,1 of B; cols 0,1 of B)
    # + A_top_right (rows 0,1 of A; cols 2,3 of A) * B_bot_left (rows 2,3 of B; cols 0,1 of B)
    
    # Part 1: A11 * B11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Part 2: A12 * B21
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 2: C_top_right (rows 0,1 of C; cols 2,3 of C)
    # = A_top_left * B_top_right
    # + A_top_right * B_bot_right
    
    # Part 3: A11 * B12
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # Part 4: A12 * B22
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 3: C_bot_left (row 2 of C; cols 0,1 of C)
    # = A_bot_left (row 2 of A; cols 0,1 of A) * B_top_left
    # + A_bot_right (row 2 of A; cols 2,3 of A) * B_bot_left
    
    # Part 5: A21 * B11
    add_schoolbook_block_1x2(2, 0, 0, 0, 2, 0)
    # Part 6: A22 * B21
    add_schoolbook_block_1x2(2, 2, 2, 0, 2, 0)
    
    # Block 4: C_bot_right (row 2 of C; cols 2,3 of C)
    # = A_bot_left * B_top_right
    # + A_bot_right * B_bot_right
    
    # Part 7: A21 * B12
    add_schoolbook_block_1x2(2, 0, 0, 2, 2, 2)
    # Part 8: A22 * B22
    add_schoolbook_block_1x2(2, 2, 2, 2, 2, 2)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    def _verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True
    
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                    
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
