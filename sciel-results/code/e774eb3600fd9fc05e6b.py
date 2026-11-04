import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition approach.
    # R(3,5,6) <= R(2,4,5) + R(1,1,1)
    # We claim R(2,4,5) = 17.
    # The 1x1x1 block is rank 1.
    # Total rank = 18.
    
    # However, constructing the explicit rank-17 decomposition for 2x4x5 is non-trivial.
    # Let's try to construct a valid decomposition with a rank that is definitely better than 90.
    # R(3,5,6) <= R(3,5,4) + R(3,5,2)
    # R(3,5,4) <= R(3,4,4) + R(3,1,4) = R(3,4,4) + 12
    # R(3,4,4) <= 2 * R(3,2,4)? No.
    # R(3,4,4) <= R(3,4,3) + R(3,4,1) = R(3,4,3) + 12
    # R(3,4,3) <= R(3,3,3) + R(3,1,3) = 23 + 9 = 32?
    # This is getting complicated.
    
    # Let's use Strassen on 2x2x2 blocks where possible.
    # 3x5x6.
    # Pad to 4x6x6?
    # R(4,6,6) <= 2 * R(4,3,6)?
    # R(4,3,6) <= 2 * R(4,3,3)?
    # R(4,3,3) <= 2 * R(2,3,3) + 2 * R(2,0,3)? No.
    # R(4,3,3) = R(4,3,3).
    # Known: R(2,2,2)=7.
    # R(4,4,4) <= 28.
    
    # Let's try a simple improvement over schoolbook (90).
    # Use Strassen 2x2x2 (rank 7) for the 2x2x2 sub-blocks.
    # 3x5x6 can be split into:
    # (2x2) x (2x2) block at (0:2, 0:2, 0:2) -> rank 7
    # (2x2) x (2x2) block at (0:2, 2:4, 0:2) -> rank 7
    # (2x2) x (2x2) block at (0:2, 0:2, 2:4) -> rank 7
    # (2x2) x (2x2) block at (0:2, 2:4, 2:4) -> rank 7
    # Remaining parts:
    # Row 2 of A (1x5)
    # Cols 4 of B (5x2)
    # Rows 2 of C (1x2)
    # Cols 4 of C (2x2)
    # This is messy.
    
    # Let's stick to a guaranteed valid decomposition that is better than 90.
    # R(3,5,6) <= R(3,5,5) + R(3,5,1)
    # R(3,5,1) = 15.
    # R(3,5,5) <= R(3,4,5) + R(3,1,5)
    # R(3,1,5) = 15.
    # R(3,4,5) <= R(3,4,4) + R(3,4,1)
    # R(3,4,1) = 12.
    # R(3,4,4) <= 2 * R(3,2,4)
    # R(3,2,4) <= R(3,2,3) + R(3,2,1)
    # R(3,2,1) = 6.
    # R(3,2,3) <= R(3,2,2) + R(3,2,1)
    # R(3,2,2) <= R(3,1,2) + R(3,1,2) ? No.
    # R(3,2,2) is small.
    
    # Let's just use the schoolbook for now but with a different structure to pass novelty check?
    # No, the novelty check failed because it was a duplicate of a previous schoolbook submission.
    # I need a *different* program structure.
    
    # Let's implement the 2x2x2 Strassen decomposition explicitly for the 2x2x2 blocks of the 3x5x6 matrix.
    # We will partition the 3x5x6 tensor into blocks.
    # A is 3x5, B is 5x6, C is 3x6.
    # We can use Strassen for the 2x2x2 block at A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2].
    # This saves 8-7=1 rank.
    # We can do this for multiple disjoint 2x2x2 blocks.
    # Block 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Block 2: A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2] -> Overlaps in C.
    # We need disjoint blocks in C.
    # C is 3x6.
    # We can fit one 2x2 block in C[0:2, 0:2].
    # Can we fit another? C[0:2, 2:4] is another 2x2 block.
    # For C[0:2, 2:4], we need A[0:2, :] and B[:, 2:4].
    # If we use A[0:2, 2:4] and B[2:4, 2:4], that's a valid 2x2x2 block.
    # So we have two disjoint 2x2x2 blocks.
    # Block 1: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Block 2: A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
    # Block 3: A[0:2, 4:5], B[4:5, 2:4]? No, dimensions don't match for 2x2x2.
    # Block 3: A[0:2, 4:5] is 2x1.
    
    # Let's just use the schoolbook for the rest.
    # Total rank = 7 + 7 + (90 - 16) = 94? No, 90 - 2*8 + 2*7 = 90 - 16 + 14 = 88.
    
    # Let's implement this.
    rank = 90
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: Strassen 2x2x2
    # A indices: 0,1 x 0,1
    # B indices: 0,1 x 0,1
    # C indices: 0,1 x 0,1
    # Strassen coefficients for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col
        # M1
        U[a_row_start*m + a_col_start][col] = 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] = 1
        V[b_row_start*p + b_col_start][col] = 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] = 1
        W[c_col_start*n + c_row_start][col] = 1
        W[(c_col_start+1)*n + (c_row_start+1)][col] = 1
        col += 1
        
        # M2
        U[(a_row_start+1)*m + a_col_start][col] = 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] = 1
        V[b_row_start*p + b_col_start][col] = 1
        W[(c_col_start+1)*n + (c_row_start+1)][col] = 1
        col += 1
        
        # M3
        U[a_row_start*m + a_col_start][col] = 1
        V[(b_row_start+1)*p + b_col_start][col] = 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] = -1
        W[c_col_start*n + c_row_start][col] = 1
        W[(c_col_start+1)*n + c_row_start][col] = 1
        col += 1
        
        # M4
        U[(a_row_start+1)*m + (a_col_start+1)][col] = 1
        V[b_row_start*p + b_col_start][col] = 1
        V[b_row_start*p + (b_col_start+1)][col] = -1
        W[c_col_start*n + c_row_start][col] = 1
        W[(c_col_start+1)*n + (c_row_start+1)][col] = 1
        col += 1
        
        # M5
        U[a_row_start*m + a_col_start][col] = 1
        U[a_row_start*m + (a_col_start+1)][col] = 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] = 1
        W[c_col_start*n + c_row_start][col] = -1
        W[(c_col_start+1)*n + (c_row_start+1)][col] = 1
        col += 1
        
        # M6
        U[(a_row_start+1)*m + a_col_start][col] = 1
        U[a_row_start*m + a_col_start][col] = -1
        V[b_row_start*p + b_col_start][col] = 1
        V[b_row_start*p + (b_col_start+1)][col] = 1
        W[(c_col_start+1)*n + c_row_start][col] = 1
        col += 1
        
        # M7
        U[a_row_start*m + (a_col_start+1)][col] = 1
        U[(a_row_start+1)*m + (a_col_start+1)][col] = -1
        V[(b_row_start+1)*p + b_col_start][col] = 1
        V[(b_row_start+1)*p + (b_col_start+1)][col] = 1
        W[c_col_start*n + c_row_start][col] = 1
        col += 1
        
    # Block 1
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # Block 2
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    # Remaining schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Block 1 or Block 2
                in_block1 = (i < 2 and j < 2 and k < 2)
                in_block2 = (i < 2 and 2 <= j < 4 and 2 <= k < 4)
                if not in_block1 and not in_block2:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
                    
    # Trim if necessary (should be exactly 88)
    rank_final = col
    U = [row[:rank_final] for row in U]
    V = [row[:rank_final] for row in V]
    W = [row[:rank_final] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
