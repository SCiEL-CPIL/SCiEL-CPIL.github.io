"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen (rank 7) at 2x2 block level × schoolbook 3x3 (rank 27) = 189 total rank
    n, m, p = DIMENSIONS  # 6, 6, 6
    b = 3  # block size
    
    # We partition 6x6 into 2x2 blocks of 3x3.
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications.
    # Each "multiplication" is a 3x3 * 3x3 product, which takes rank 9 (schoolbook).
    # Total rank = 7 * 9 = 63... wait, that's not right.
    # 
    # Actually, the tensor decomposition rank for block Strassen:
    # Each Strassen multiplication M1..M7 corresponds to a 3x3x3 tensor of rank 9.
    # But we need to be careful about how the U, V, W matrices are constructed.
    #
    # Let's think more carefully. The matrix multiplication tensor for 6x6 * 6x6 -> 6x6
    # has rank at most 216 (schoolbook). 
    #
    # Strassen at block level: we view A, B, C as 2x2 block matrices with 3x3 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]]
    #
    # Strassen computes:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A22 - A12) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 3x3 matrix product. The rank of a 3x3 matrix multiplication tensor is 9.
    # So total rank = 7 * 9 = 63.
    #
    # Wait, but the knowledge says: "Strassen (rank 7) at 2x2 block level × schoolbook 3x3 (rank 27) = 189 total rank"
    # That's 7 * 27 = 189, not 7 * 9 = 63.
    #
    # Hmm, let me reconsider. The rank of 3x3 matrix multiplication is 9, not 27.
    # 3*3*3 = 27 is the schoolbook rank for 3x3*3x3 -> 3x3? No, that's not right either.
    # For n x m * m x p, the schoolbook rank is n*m*p. For 3x3 * 3x3 -> 3x3, that's 3*3*3 = 27.
    # But the actual rank of 3x3 matrix multiplication is known to be 9? No, that's not right.
    # The rank of n x n matrix multiplication is at most n^3 (schoolbook), and for n=2 it's 7 (Strassen).
    # For n=3, the best known upper bound is 23, but it's definitely not 9.
    #
    # So the schoolbook rank for 3x3 * 3x3 -> 3x3 is 3*3*3 = 27.
    # Therefore, Strassen at block level with schoolbook inner products gives 7 * 27 = 189.
    #
    # Let me construct the U, V, W matrices for this decomposition.
    
    rank = 7 * 27  # 189
    
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)  
    # W has shape (p*n, rank) = (36, 189)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indexing:
    # A is 6x6, indexed as A[i, j] where i in 0..5, j in 0..5
    # U row index: i * m + j = i * 6 + j (for A[i, j])
    # B is 6x6, indexed as B[j, k] where j in 0..5, k in 0..5
    # V row index: j * p + k = j * 6 + k (for B[j, k])
    # C is 6x6, indexed as C[i, k] where i in 0..5, k in 0..5
    # W row index: k * n + i = k * 6 + i (for C[i, k]) following W[k*n+i, r]
    
    # Block indices: 
    # Block (bi, bj) of A covers rows bi*3 to bi*3+2, cols bj*3 to bj*3+2
    # So A[i, j] is in block (i//3, j//3) at position (i%3, j%3)
    
    # For each Strassen multiplication s in 0..6:
    #   For each inner product term (ii, jj, kk) in 0..2:
    #     The contribution to U, V, W
    
    # Strassen formulas (using 0-indexed blocks: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    # Let's use block indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[1,1] - A[0,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we have a 3x3*3x3 -> 3x3 product.
    # The schoolbook decomposition of 3x3*3x3 -> 3x3 has rank 27.
    # For inner indices (ii, jj, kk) in 0..2:
    #   U contribution: A block element at (block_row, ii, block_col, jj) with coefficient
    #   V contribution: B block element at (block_row, jj, block_col, kk) with coefficient
    #   W contribution: C block element at (block_row, kk, block_col, ii) with coefficient
    
    # Let's define the Strassen multiplications as (U_blocks, V_blocks, W_blocks) where each is a list of (block_idx, coeff)
    
    # U_blocks for each M: which A blocks with what coefficients
    # V_blocks for each M: which B blocks with what coefficients
    # W_blocks: which C blocks receive the result with what coefficients
    
    strassen = []
    # M1
    strassen.append((
        [(0, 1), (3, 1)],  # A[0,0] + A[1,1]
        [(0, 1), (3, 1)],  # B[0,0] + B[1,1]
        [(0, 1), (2, 1), (3, -1), (1, 1)]  # C[0,0] += M1, C[1,1] += M1... wait
    ))
    
    # Let me be more careful. C[bi, bj] = sum of contributions from each M.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So for each M_s, we need to add its contribution to the appropriate C blocks.
    
    # Let me restructure: for each s in 0..6, define:
    #   u_blocks[s] = list of (block_idx, coeff) for A
    #   v_blocks[s] = list of (block_idx, coeff) for B
    #   w_blocks[s] = list of (block_idx, coeff) for C (where the result of M_s is added)
    
    u_blocks = [
        [(0, 1), (3, 1)],       # M1: A[0,0] + A[1,1]
        [(2, 1), (3, 1)],       # M2: A[1,0] + A[1,1]
        [(0, 1)],               # M3: A[0,0]
        [(3, 1)],               # M4: A[1,1]
        [(0, 1), (1, 1)],       # M5: A[0,0] + A[0,1]
        [(2, 1), (0, -1)],      # M6: A[1,0] - A[0,0]
        [(3, 1), (1, -1)],      # M7: A[1,1] - A[0,1]
    ]
    
    v_blocks = [
        [(0, 1), (3, 1)],       # M1: B[0,0] + B[1,1]
        [(0, 1)],               # M2: B[0,0]
        [(1, 1), (3, -1)],      # M3: B[0,1] - B[1,1]
        [(2, 1), (0, -1)],      # M4: B[1,0] - B[0,0]
        [(3, 1)],               # M5: B[1,1]
        [(0, 1), (1, 1)],       # M6: B[0,0] + B[0,1]
        [(2, 1), (3, 1)],       # M7: B[1,0] + B[1,1]
    ]
    
    w_blocks = [
        [(0, 1), (3, 1)],       # M1: C[0,0] += M1, C[1,1] += M1
        [(2, 1), (3, -1)],      # M2: C[1,0] += M2, C[1,1] -= M2
        [(1, 1), (3, 1)],       # M3: C[0,1] += M3, C[1,1] += M3
        [(0, 1), (2, 1)],       # M4: C[0,0] += M4, C[1,0] += M4
        [(0, -1), (1, 1)],      # M5: C[0,0] -= M5, C[0,1] += M5
        [(3, 1)],               # M6: C[1,1] += M6
        [(0, 1)],               # M7: C[0,0] += M7
    ]
    
    # Now construct U, V, W
    for s in range(7):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    col = s * 27 + ii * 9 + jj * 3 + kk
                    
                    # U: for each (block_idx, coeff) in u_blocks[s]
                    for (bl, coeff) in u_blocks[s]:
                        bri, bcj = bl // 2, bl % 2  # block row, block col
                        # A element at global row = bri*3 + ii, col = bcj*3 + jj
                        row_u = (bri * 3 + ii) * 6 + (bcj * 3 + jj)
                        U[row_u][col] += coeff
                    
                    # V: for each (block_idx, coeff) in v_blocks[s]
                    for (bl, coeff) in v_blocks[s]:
                        bri, bcj = bl // 2, bl % 2
                        # B element at global row = bri*3 + jj, col = bcj*3 + kk
                        row_v = (bri * 3 + jj) * 6 + (bcj * 3 + kk)
                        V[row_v][col] += coeff
                    
                    # W: for each (block_idx, coeff) in w_blocks[s]
                    for (bl, coeff) in w_blocks[s]:
                        bri, bcj = bl // 2, bl % 2
                        # C element at global row = bri*3 + kk, col = bcj*3 + ii
                        # W index: k*n + i where k = col of C, i = row of C
                        # C[i_global, k_global] where i_global = bri*3 + ii... wait
                        # C[bi, bj] block: C row = bi*3 + kk, C col = bj*3 + ii
                        # W index = k * n + i = (bj*3 + ii) * 6 + (bi*3 + kk)
                        row_w = (bcj * 3 + ii) * 6 + (bri * 3 + kk)
                        W[row_w][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
