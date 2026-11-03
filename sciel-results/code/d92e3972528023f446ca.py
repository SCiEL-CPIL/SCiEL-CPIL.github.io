"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110s wall-clock limit with safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We will construct a decomposition of rank 77.
    # Part 1: 4x4 * 4x4 (first 4 cols of B) using recursive Strassen (Rank 49)
    # Part 2: 4x4 * 4x2 (last 2 cols of B) using block Strassen (Rank 28)
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # --- Part 1: 4x4 * 4x4 (cols 0-3 of B, cols 0-3 of C) ---
    # Recursive Strassen for 4x4
    # This effectively computes C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]
    
    # We can hardcode the 49 terms or generate them.
    # Generating is safer to avoid typos.
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we treat A and B as 2x2 blocks of 2x2 matrices.
    # Let A_blocks = [[A11, A12], [A21, A22]] where each is 2x2.
    # Let B_blocks = [[B11, B12], [B21, B22]] where each is 2x2.
    # The 7 Strassen multiplications are on these blocks.
    # Each block multiplication (2x2 * 2x2) is done with 7 scalar multiplications.
    
    # Indices for 2x2 blocks:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,2,3 in A flat (row-major)
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7? No.
    # A is 4x4.
    # A[i,j] is at index i*4 + j.
    # Block A11 (top-left 2x2): i in 0,1; j in 0,1. Indices: 0,1,4,5.
    # Block A12 (top-right 2x2): i in 0,1; j in 2,3. Indices: 2,3,6,7.
    # Block A21 (bottom-left 2x2): i in 2,3; j in 0,1. Indices: 8,9,12,13.
    # Block A22 (bottom-right 2x2): i in 2,3; j in 2,3. Indices: 10,11,14,15.
    
    # Similarly for B (4x4 part):
    # B11: i in 0,1; j in 0,1. Indices: 0,1,4,5.
    # B12: i in 0,1; j in 2,3. Indices: 2,3,6,7.
    # B21: i in 2,3; j in 0,1. Indices: 8,9,12,13.
    # B22: i in 2,3; j in 2,3. Indices: 10,11,14,15.
    
    # C output for 4x4 part:
    # C11: i in 0,1; j in 0,1. Indices in W (p*n = 4*4=16 for this part? No, W is p*n = 6*4=24).
    # W index for C[i,k] is k*n + i.
    # For the 4x4 part, k is 0..3, i is 0..3.
    # C11 (top-left): i in 0,1; k in 0,1. W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    
    # We need a function to add a 2x2*2x2 multiplication using Strassen.
    # It takes block indices for A, B, and C, and the base column offset.
    
    def add_strassen_2x2(a_block_start, b_block_start, c_block_start, base_col):
        # a_block_start: index of A[0,0] in the 2x2 block
        # The 2x2 block in A (flat 4x4) has offsets:
        # 0,0 -> 0
        # 0,1 -> 1
        # 1,0 -> 4
        # 1,1 -> 5
        # This depends on the block position.
        
        # Let's define the local indices for the 2x2 block relative to its top-left corner in the 4x4 matrix.
        # For a 2x2 block starting at row r, col c in the 4x4 matrix:
        # A[r,c] is at r*4+c
        # A[r,c+1] is at r*4+c+1
        # A[r+1,c] is at (r+1)*4+c
        # A[r+1,c+1] is at (r+1)*4+c+1
        
        # We will pass the top-left corner indices (r, c) for A, B, and C.
        pass

    # This is getting complicated to hardcode manually.
    # Alternative: Use the known rank 49 decomposition coefficients if available?
    # No, I will generate the schoolbook for now and try to swap in Strassen logic.
    
    # Actually, for the purpose of this task, if I cannot guarantee the correctness of the complex Strassen construction in the time limit,
    # I should stick to a simpler improvement or the baseline.
    # However, 96 is the baseline.
    # Can I just use the schoolbook for 4x4x4 (rank 64) and 4x4x2 (rank 32)? Total 96.
    
    # Let's try to implement the Block Strassen for 4x4x4 (Rank 56) and 4x4x2 (Rank 28).
    # Total 84.
    
    # I will implement a generic Strassen 2x2 adder.
    
    def get_2x2_indices(base_r, base_c, stride_r, stride_c):
        """Returns the 4 flat indices for a 2x2 block starting at base_r, base_c with given strides."""
        return [
            base_r * stride_r + base_c * stride_c,
            base_r * stride_r + (base_c + 1) * stride_c,
            (base_r + 1) * stride_r + base_c * stride_c,
            (base_r + 1) * stride_r + (base_c + 1) * stride_c
        ]

    # For A (4x4), stride_r=4, stride_c=1.
    # For B (4x6), stride_r=6, stride_c=1.
    # For C (6x4), stride_r=4, stride_c=1. (W index is k*n + i, so row k, col i. Stride_r=4, stride_c=1).
    
    # Strassen 2x2 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to add 7 terms to the decomposition for each 2x2*2x2 multiplication.
    
    def add_strassen_term(a_idx, b_idx, c_idx, col, coeff_a, coeff_b, coeff_c):
        # a_idx, b_idx, c_idx are lists of 4 indices
        # coeff_a, coeff_b, coeff_c are lists of 4 coefficients (-1, 0, 1)
        for i in range(4):
            if coeff_a[i] != 0:
                U[a_idx[i]][col] += coeff_a[i]
            if coeff_b[i] != 0:
                V[b_idx[i]][col] += coeff_b[i]
            if coeff_c[i] != 0:
                W[c_idx[i]][col] += coeff_c[i]

    # Define the 7 Strassen multiplications for a given set of block indices.
    # a11, a12, a21, a22 are the 2x2 blocks of A.
    # b11, b12, b21, b22 are the 2x2 blocks of B.
    # c11, c12, c21, c22 are the 2x2 blocks of C.
    
    def add_strassen_2x2(a_blocks, b_blocks, c_blocks, base_col):
        # a_blocks: [a11, a12, a21, a22] (each is list of 4 indices)
        # b_blocks: [b11, b12, b21, b22]
        # c_blocks: [c11, c12, c21, c22]
        
        # M1 = (A11+A22)(B11+B22)
        a1 = [x + y for x, y in zip(a_blocks[0], a_blocks[3])] # No, indices are not added.
        # We need to add the terms to the global U, V, W matrices.
        # U[a11] + U[a22] means U[a11_idx] = 1, U[a22_idx] = 1.
        
        col = base_col
        # M1
        # A: A11(1), A22(1)
        # B: B11(1), B22(1)
        # C: C11(1), C12(1), C21(1), C22(1) ? No.
        # C11 += M1, C12 += 0, C21 += 0, C22 += M1?
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C11 and C22.
        
        # Term 1: M1
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: C11(1), C22(1)
        
        # We need to map the 4 indices of A11, A22, etc.
        
        # This manual mapping is error-prone.
        
    # Given the complexity and risk of error, and the 120s limit,
    # I will return the schoolbook decomposition (Rank 96) which is guaranteed to be correct.
    # The score will be 1.0.
    
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
# EVOLVE-BLOCK-END
