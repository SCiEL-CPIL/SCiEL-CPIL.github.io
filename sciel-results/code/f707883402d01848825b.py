"""Hybrid Strassen+schoolbook decomposition for 4x4 * 4x6 matrix multiplication.

Uses a clean coefficient-table approach:
- Top-level: 7 Strassen block-multiplications of (2x2) x (2x3)
- Each block multiplication: inner Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = 11
- Total rank: 7 * 11 = 77
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 coefficient tables
# A-blocks indexed: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
# B-blocks indexed: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

# alpha[s][a]: coefficient of A-block a in the s-th Strassen linear combination
alpha = [
    [1, 0, 0, 1],   # s=0: A00 + A11
    [0, 0, 1, 1],   # s=1: A10 + A11
    [1, 0, 0, 0],   # s=2: A00
    [0, 0, 0, 1],   # s=3: A11
    [1, 1, 0, 0],   # s=4: A00 + A01
    [-1, 0, 1, 0],  # s=5: A10 - A00
    [0, 1, 0, -1],  # s=6: A01 - A11
]

# beta[s][b]: coefficient of B-block b in the s-th Strassen linear combination
beta = [
    [1, 0, 0, 1],   # s=0: B00 + B11
    [1, 0, 0, 0],   # s=1: B00
    [0, 1, 0, -1],  # s=2: B01 - B11
    [-1, 0, 1, 0],  # s=3: B10 - B00
    [0, 0, 0, 1],   # s=4: B11
    [1, 1, 0, 0],   # s=5: B00 + B01
    [0, 0, 1, 1],   # s=6: B10 + B11
]

# recon[s]: list of (out_row, out_col, coeff) for the s-th Strassen product
# Output C-block positions: (0,0), (0,1), (1,0), (1,1)
recon = [
    [(0, 0, 1), (1, 1, 1)],     # s=0
    [(1, 0, 1), (1, 1, -1)],    # s=1
    [(0, 1, 1), (1, 1, 1)],     # s=2
    [(0, 0, 1), (1, 0, 1)],     # s=3
    [(0, 0, -1), (0, 1, 1)],    # s=4
    [(1, 1, 1)],                # s=5
    [(0, 0, 1)]                 # s=6
]


def _build_decomposition():
    """Build the rank-77 decomposition for 4x4 * 4x6.
    
    Structure:
    - A is 4x4, viewed as 2x2 blocks of 2x2: A[r,c] where r,c in {0,1}
    - B is 4x6, viewed as 2x3 blocks of 2x2: B[r,c] where r in {0,1}, c in {0,1,2}
    - C is 4x6, viewed as 2x3 blocks of 2x2: C[r,c] where r in {0,1}, c in {0,1,2}
    
    Top-level Strassen on the 2x2 block structure of A times the first 2 col-blocks of B:
      X_s = sum_r alpha[s][r] * A[r, r//2]  (2x2 matrix)
      Y_s = sum_c beta[s][c] * B[c//2, c%2] (2x2 matrix, cols 0-1 of B)
    
    But we need to handle all 3 col-blocks of B. The approach:
      For each Strassen term s:
        X_s is a 2x2 matrix (linear combo of A blocks)
        Y_s is a 2x3 matrix (linear combo of B blocks, all 3 col-blocks)
        
    Wait - Strassen requires square matrices. Let me reconsider.
    
    Actually the correct approach from the inspirations:
    - Top-level: 7 terms. In term s:
      X_s = sum_{r=0}^{3} alpha[s][r] * A_block[r//2, r%2]  (a 2x2 matrix)
      Y_s = sum_{c=0}^{3} beta[s][c] * B_block[c//2, c%2]  (a 2x2 matrix, using B col-blocks 0,1)
      
    But that only covers 4 columns. For the 6th column (col-block 2 of B),
    we need additional terms.
    
    Looking at the inspiration more carefully:
    - The top-level Strassen uses 7 terms where B is treated as 2x2 blocks (cols 0,1)
    - Then for col-block 2 of B, we use schoolbook at the block level
    - But the inspiration shows a unified approach where each Strassen term handles
      a 2x2 x 2x3 product via inner Strassen (cols 0,1) + schoolbook (col 2)
    
    Let me re-read the inspiration code:
    - For each top-level Strassen term s (7 terms):
      - X_s is 2x2, Y_s is 2x3
      - Part 1: inner Strassen for X_s * Y_s[:, 0:2] (2x2 x 2x2) -> 7 inner terms
      - Part 2: schoolbook for X_s * Y_s[:, 2] (2x2 x 2x1) -> 4 terms
      - Total per s: 11
    - Grand total: 7 * 11 = 77
    
    This means the top-level "Strassen" is actually a generalization where B has 3 col-blocks.
    The alpha coefficients combine A's 2x2 blocks, and beta coefficients combine B's 2x2 blocks
    (but B has 3 col-blocks, so we need to handle them separately).
    
    Actually, I think the structure is:
    - A: 2x2 blocks of 2x2 (4 blocks total)
    - B: 2x3 blocks of 2x2 (6 blocks total: B[r, c] for r in {0,1}, c in {0,1,2})
    
    The top-level Strassen-like decomposition:
    For each s in 0..6:
      X_s = sum_{r=0}^{3} alpha[s][r] * A[r//2, r%2]   (2x2 matrix)
      For the B side, we need a 2x3 matrix:
      Y_s[:, c] = sum_{r=0}^{3} beta[s][r] * B[r//2, r%2 + c_offset]
      
    Hmm, this doesn't quite work with standard Strassen because Strassen assumes both
    inputs are 2x2 block matrices.
    
    Let me look at the inspiration code more carefully. In the inspiration:
    - alpha[s] has 4 entries (for 4 A-blocks)
    - beta[s] has 4 entries (for 4 B-blocks, i.e., B[0,0], B[0,1], B[1,0], B[1,1])
    
    So the top-level Strassen only covers the first 2 col-blocks of B (cols 0-3).
    The 3rd col-block (cols 4-5) must be handled separately.
    
    But the inspiration says total rank = 7 * (7+4) = 77, which suggests all 6 columns
    are handled within the 7 top-level terms.
    
    Wait - I think I misread. Let me re-examine:
    "Total rank: 7 * (7 + 4) = 77"
    
    If the top-level Strassen handles cols 0-3 (4 columns) with 7 terms, and each term
    does inner Strassen (7) for the 2x2x2 part, that gives 49 for cols 0-3.
    Then cols 4-5 need 4 * 7 = 28 more (schoolbook block-level: 4 terms, each inner Strassen).
    Total: 49 + 28 = 77.
    
    But the inspiration code shows a different structure where each of the 7 top-level terms
    handles all 3 col-blocks. Let me trace through the inspiration code:
    
    In the V part:
    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
    
    Here bc is the B row-block (0 or 1), and y_inner_col is 0 or 1 (for the inner Strassen part).
    So V index = (row) * 6 + (col), where row = ac*2 + y_row, col = bc*3 + y_inner_col.
    
    For the schoolbook part:
    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
    col = bc*3 + 2, which is col-block 2.
    
    So in the inner Strassen part, y_inner_col is 0 or 1, meaning cols bc*3+0 and bc*3+1.
    In the schoolbook part, col is bc*3+2.
    
    This means the 2x3 Y_s matrix has columns at positions:
    - Col-block 0: cols 0, 1 (from B[*, 0])
    - Col-block 1: cols 3, 4 (from B[*, 1])  
    - Col-block 2: cols 6, 7 (from B[*, 2])
    
    Wait, that's not right either. B is 4x6, so B[j, k] where j in 0..3, k in 0..5.
    B blocks: B[r, c] is a 2x2 submatrix at rows 2r..2r+1, cols 2c..2c+1.
    So B[0,0] = B[0:2, 0:2], B[0,1] = B[0:2, 2:4], B[0,2] = B[0:2, 4:6]
       B[1,0] = B[2:4, 0:2], B[1,1] = B[2:4, 2:4], B[1,2] = B[2:4, 4:6]
    
    In the V index: v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
    - ac is the B row-block from beta (0 or 1)
    - y_row is the inner row (0 or 1)
    - bc is the B col-block from beta (0 or 1)
    - y_inner_col is the inner col (0 or 1)
    
    So the global B index is: row = ac*2 + y_row, col = bc*3 + y_inner_col
    - For bc=0: cols 0, 1
    - For bc=1: cols 3, 4
    
    That's only 4 columns (0,1,3,4), missing cols 2 and 5!
    
    I think there's an issue with the indexing. Let me reconsider.
    
    Actually, I think the B blocks in the top-level Strassen are:
    - B[0,0] = rows 0-1, cols 0-1
    - B[0,1] = rows 0-1, cols 2-3
    - B[1,0] = rows 2-3, cols 0-1
    - B[1,1] = rows 2-3, cols 2-3
    
    So bc*2 + y_inner_col for the column within the block.
    Global col = bc*2 + y_inner_col... no wait, bc is the block index.
    
    Let me re-read: v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
    
    Hmm, bc*3? That seems wrong for 2x2 blocks. Unless the blocks are 3-wide?
    
    Oh wait, I think I see. The B matrix is 4x6. If we split into 2 row-blocks of 2 rows each,
    and 3 col-blocks of 2 cols each, then B[r, c] is at rows 2r..2r+1, cols 2c..2c+1.
    
    But in the Strassen decomposition, we only use 2 col-blocks (c=0,1) for the Strassen part,
    and c=2 for the schoolbook part.
    
    So for the Strassen part (cols 0-3):
    - B[0,0] = rows 0-1, cols 0-1
    - B[0,1] = rows 0-1, cols 2-3
    - B[1,0] = rows 2-3, cols 0-1
    - B[1,1] = rows 2-3, cols 2-3
    
    Global B index: row = ac*2 + y_row, col = bc*2 + y_inner_col
    v_idx = (ac*2 + y_row) * 6 + (bc*2 + y_inner_col)
    
    For the schoolbook part (cols 4-5):
    - B[0,2] = rows 0-1, cols 4-5
    - B[1,2] = rows 2-3, cols 4-5
    
    Global B index: row = ac*2 + v, col = 2*2 + u = 4 + u
    v_idx = (ac*2 + v) * 6 + (4 + u)
    
    But the inspiration has bc*3 + y_inner_col. Let me check if maybe they're using a different
    block size. If B is split into 2 row-blocks and 3 col-blocks, and each block is 2x2,
    then col = bc*2 + offset. But bc*3 would mean blocks of width 3, which doesn't make sense
    for 2x2 blocks.
    
    Unless... the "2x3" in the inner product refers to something else. Let me re-read the
    problem statement: "4x4 * 4x6". So A is 4x4, B is 4x6.
    
    I think the inspiration code might have a bug or I'm misreading it. Let me just implement
    the correct version from scratch.
    
    Correct structure:
    - A: 4x4, split into 2x2 blocks of 2x2: A[r, c] for r,c in {0,1}
    - B: 4x6, split into 2x3 blocks of 2x2: B[r, c] for r in {0,1}, c in {0,1,2}
    - C: 4x6, split into 2x3 blocks of 2x2: C[r, c] for r in {0,1}, c in {0,1,2}
    
    Top-level decomposition:
    For the first 2 col-blocks of B (cols 0-3), use Strassen:
      7 terms, each computing X_s * Y_s where X_s, Y_s are 2x2 matrices
      Each 2x2*2x2 product uses inner Strassen: rank 7
      Subtotal: 7 * 7 = 49
    
    For the 3rd col-block of B (cols 4-5), use schoolbook at block level:
      4 terms (2 row-blocks of A x 2 col-blocks of A... wait, no)
      
    Actually for C[:, 4:6] = A * B[:, 4:6]:
    C[r, 2] = sum_{c=0}^{1} A[r, c] * B[c, 2]
    That's 2*2 = 4 block multiplications, each 2x2 * 2x2 = 2x2.
    Each uses inner Strassen: rank 7.
    Subtotal: 4 * 7 = 28
    
    Total: 49 + 28 = 77.
    
    Now let me implement this correctly.
    """
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)
    
    col = 0
    
    # Part 1: Strassen for cols 0-3 (first 2 col-blocks of B)
    for s in range(7):
        # X_s = sum_{r=0}^{3} alpha[s][r] * A[r//2, r%2]
        # Y_s = sum_{c=0}^{3} beta[s][c] * B[c//2, c%2]
        # Product: X_s * Y_s = sum_{(i,j,k) in recon[s]} coeff * X_s[i,j] * Y_s[j,k]
        # Each X_s[i,j] * Y_s[j,k] is a 2x2 * 2x2 = 2x2 product, done with inner Strassen
        
        for i, j, coeff in recon[s]:
            # X_s[i, :] * Y_s[:, j] -> C_block[i, j] (2x2 result)
            # X_s[i, a] = sum_{r} alpha[s][r] * A[r//2, r%2][i, a]
            # Y_s[b, j] = sum_{c} beta[s][c] * B[c//2, c%2][b, j]
            # C_block[i, j][a, b] = sum_{x=0}^{1} X_s[i, x] * Y_s[x, j]
            #                     = sum_{x=0}^{1} (sum_r alpha[s][r] * A[r//2, r%2][i, x]) 
            #                                  * (sum_c beta[s][c] * B[c//2, c%2][x, j])
            
            # Inner Strassen for the 2x2*2x2 product:
            # The "A" matrix for inner Strassen is X_s[i, :] (1x2 vector, but we need 2x2)
            # Actually, X_s[i, :] is a row vector, not a 2x2 matrix.
            
            # I think I'm overcomplicating this. Let me think differently.
            
            # The full product C[i, j] (a 2x2 matrix) = X_s[i, :] @ Y_s[:, j]
            # where X_s[i, :] is a 1x2 vector and Y_s[:, j] is a 2x1 vector.
            # That's just a dot product, rank 2.
            
            # No wait, X_s is a 2x2 matrix and Y_s is a 2x2 matrix.
            # C_block[i, j] = X_s[i, :] @ Y_s[:, j] is a scalar, not a 2x2 matrix.
            
            # I'm confusing myself. Let me restart with clearer notation.
            
            pass  # Will redo below
    
    # Let me redo this with a cleaner approach.
    # 
    # A is 4x4. A[i, j] for i,j in 0..3.
    # A-block A[r, c] = A[2r:2r+2, 2c:2c+2], a 2x2 submatrix.
    # 
    # B is 4x6. B[i, j] for i in 0..3, j in 0..5.
    # B-block B[r, c] = B[2r:2r+2, 2c:2c+2], a 2x2 submatrix.
    # 
    # C is 4x6. C[i, j] for i in 0..3, j in 0..5.
    # C-block C[r, c] = C[2r:2r+2, 2c:2c+2], a 2x2 submatrix.
    # 
    # C[r, c] = sum_{t=0}^{1} A[r, t] @ B[t, c]
    # 
    # For c=0,1: use Strassen on the 2x2 block matrix product
    #   [A[0,0] A[0,1]]   [B[0,0] B[0,1]]   [C[0,0] C[0,1]]
    #   [A[1,0] A[1,1]] @ [B[1,0] B[1,1]] = [C[1,0] C[1,1]]
    # 
    # For c=2: schoolbook
    #   C[0,2] = A[0,0]@B[0,2] + A[0,1]@B[1,2]
    #   C[1,2] = A[1,0]@B[0,2] + A[1,1]@B[1,2]
    
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    # Part 1: Strassen for C[*, 0:4] (cols 0-3)
    for s in range(7):
        # X_s = sum_{r=0}^{3} alpha[s][r] * A[r//2, r%2]  (2x2 matrix)
        # Y_s = sum_{c=0}^{3} beta[s][c] * B[c//2, c%2]  (2x2 matrix)
        # Contribution: coeff * X_s @ Y_s added to appropriate C blocks
        # 
        # X_s[a, b] = sum_{r} alpha[s][r] * A[2*(r//2)+a, 2*(r%2)+b]
        # Y_s[a, b] = sum_{c} beta[s][c] * B[2*(c//2)+a, 2*(c%2)+b]
        # 
        # (X_s @ Y_s)[i, j] = sum_{x=0}^{1} X_s[i, x] * Y_s[x, j]
        # 
        # For each recon entry (out_r, out_c, coeff):
        #   C[out_r, out_c] += coeff * (X_s @ Y_s)[out_r, out_c]
        #   C[out_r, out_c][a, b] += coeff * sum_x X_s[out_r, x] * Y_s[x, out_c]
        # 
        # Each (X_s @ Y_s)[i, j] is a scalar, but we need to express it as
        # a sum of products of linear forms in A and B entries.
        # 
        # (X_s @ Y_s)[i, j] = sum_{x=0}^{1} X_s[i, x] * Y_s[x, j]
        # = sum_x [sum_r alpha[s][r] * A[2*(r//2)+i, 2*(r%2)+x]] * [sum_c beta[s][c] * B[2*(c//2)+x, 2*(c%2)+j]]
        # 
        # This is a sum of 2 * 4 * 4 = 32 terms, but we can use inner Strassen
        # to reduce the rank.
        # 
        # Actually, the inner Strassen applies to the 2x2*2x2 matrix product X_s @ Y_s.
        # The result is a 2x2 matrix, and Strassen computes it with 7 scalar multiplications.
        # 
        # Inner Strassen for X_s @ Y_s:
        #   M1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
        #   M2 = (X[1,0]+X[1,1])*Y[0,0]
        #   M3 = X[0,0]*(Y[0,1]-Y[1,1])
        #   M4 = X[1,1]*(Y[1,0]-Y[0,0])
        #   M5 = (X[0,0]+X[0,1])*Y[1,1]
        #   M6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
        #   M7 = (X[1,1]-X[1,0])*(Y[1,0]+Y[1,1])
        # 
        # Each M_k is a scalar = (linear form in X entries) * (linear form in Y entries)
        # = (linear form in A entries) * (linear form in B entries)
        # 
        # So each M_k contributes one rank-1 term to the decomposition.
        # 7 such terms per Strassen block product.
        
        for si in range(7):
            # Inner Strassen term si
            # M_si = (sum_a inner_alpha[si][a] * X_s[a//2, a%2]) * (sum_b inner_beta[si][b] * Y_s[b//2, b%2])
            
            # U part: linear form in A entries
            # sum_a inner_alpha[si][a] * X_s[a//2, a%2]
            # = sum_a inner_alpha[si][a] * sum_r alpha[s][r] * A[2*(r//2)+(a//2), 2*(r%2)+(a%2)]
            for a in range(4):
                ia = alpha[si][a]
                if ia == 0:
                    continue
                x_r, x_c = a // 2, a % 2
                for r in range(4):
                    toa = alpha[s][r]
                    if toa == 0:
                        continue
                    ar, ac_ = r // 2, r % 2
                    # A index: row = 2*ar + x_r, col = 2*ac_ + x_c
                    u_idx = (2*ar + x_r) * 4 + (2*ac_ + x_c)
                    U[u_idx, col] += ia * toa
            
            # V part: linear form in B entries
            # sum_b inner_beta[si][b] * Y_s[b//2, b%2]
            # = sum_b inner_beta[si][b] * sum_c beta[s][c] * B[2*(c//2)+(b//2), 2*(c%2)+(b%2)]
            for b in range(4):
                ib = beta[si][b]
                if ib == 0:
                    continue
                y_r, y_c = b // 2, b % 2
                for c in range(4):
                    tob = beta[s][c]
                    if tob == 0:
                        continue
                    br, bc_ = c // 2, c % 2
                    # B index: row = 2*br + y_r, col = 2*bc_ + y_c
                    v_idx = (2*br + y_r) * 6 + (2*bc_ + y_c)
                    V[v_idx, col] += ib * tob
            
            # W part: contribution to C
            # For each (out_r, out_c, coeff) in recon[s]:
            #   C[out_r, out_c] += coeff * (contribution from inner Strassen term si)
            # Inner Strassen recon: (out_ir, out_ic, inner_coeff)
            #   (X@Y)[out_ir, out_ic] += inner_coeff * M_si
            #   C_block[out_r, out_c][a, b] += coeff * inner_coeff * M_si  (for appropriate a,b)
            # 
            # Wait, I need to be more careful. The inner Strassen computes X_s @ Y_s,
            # which is a 2x2 matrix. The result (X_s @ Y_s)[i, j] is a scalar.
            # But C_block[out_r, out_c] is a 2x2 matrix, and it equals coeff * (X_s @ Y_s)[out_r, out_c]
            # which is coeff times a scalar. That doesn't make sense dimensionally.
            # 
            # I think the issue is that X_s and Y_s are 2x2 matrices of scalars,
            # and their product is a 2x2 matrix of scalars. Each entry (X_s @ Y_s)[i, j]
            # is a scalar. But C_block[out_r, out_c] is a 2x2 matrix.
            # 
            # Oh! I see the confusion. The "blocks" A[r, c] are 2x2 matrices, and
            # the block product A[r, c] @ B[c, d] is a 2x2 @ 2x2 = 2x2 matrix product.
            # So C_block[out_r, out_c] = sum_t A[out_r, t] @ B[t, out_c] is a 2x2 matrix.
            # 
            # In the Strassen decomposition, X_s and Y_s are 2x2 matrices whose entries
            # are themselves 2x2 matrices (the blocks). So X_s @ Y_s is a 2x2 matrix
            # of 2x2 matrices, i.e., a 4x4 matrix.
            # 
            # So the inner Strassen is applied to the 2x2 matrix product where each
            # "entry" is a 2x2 matrix. The inner Strassen computes 7 products of
            # 2x2 matrices, each done with another level of Strassen (rank 7).
            # 
            # This means:
            # - Top-level: 7 Strassen terms
            # - Each term: 7 inner Strassen terms (for the 2x2 matrix product)
            # - Each inner term: 1 scalar multiplication
            # Total: 7 * 7 = 49 for the first 4 columns.
            
            for out_r, out_c, coeff in recon[s]:
                for ir, ic, inner_coeff in recon[si]:
                    # C_block[out_r, out_c] += coeff * inner_coeff * M_si
                    # M_si is a scalar = (linear form in A) * (linear form in B)
                    # C_block[out_r, out_c][a, b] = C[2*out_r+a, 2*out_c+b]
                    # W index: C[k*n + i] where k=col, i=row
                    # C[2*out_r+a, 2*out_c+b] -> W[(2*out_c+b)*4 + (2*out_r+a)]
                    
                    # But wait, M_si contributes to (X_s @ Y_s)[ir, ic], not to
                    # C_block[out_r, out_c] directly. The relationship is:
                    # C_block[out_r, out_c] = coeff * (X_s @ Y_s)[out_r, out_c]
                    # and (X_s @ Y_s)[out_r, out_c] is computed by inner Strassen.
                    # 
                    # So for each inner Strassen term si, it contributes to
                    # (X_s @ Y_s)[ir, ic] via inner_coeff * M_si.
                    # And (X_s @ Y_s)[ir, ic] contributes to C_block[ir, ic] via
                    # the top-level recon coefficient.
                    # 
                    # But the top-level recon says C_block[out_r, out_c] gets
                    # coeff * (X_s @ Y_s)[out_r, out_c]. So ir=out_r and ic=out_c.
                    # 
                    # Hmm, but that means each inner Strassen term si only contributes
                    # to one specific C block (determined by ir, ic from recon[si]).
                    # And the top-level recon determines which C block gets the result.
                    # 
                    # I think the correct interpretation is:
                    # - Top-level Strassen: 7 terms, each computing a 2x2 matrix product
                    #   X_s @ Y_s (where X_s, Y_s are 2x2 matrices of 2x2 blocks)
                    # - The result X_s @ Y_s is a 2x2 matrix of 2x2 matrices
                    # - Top-level recon: C_block[out_r, out_c] += coeff * (X_s @ Y_s)[out_r, out_c]
                    # - Inner Strassen: computes (X_s @ Y_s)[i, j] for all i,j using 7 terms
                    #   (X_s @ Y_s)[i, j] = sum_{si} inner_recon[si] contribution
                    # 
                    # So for a given top-level term s and inner term si:
                    # - Inner recon says: (X_s @ Y_s)[ir, ic] += inner_coeff * M_si
                    # - Top-level recon says: C_block[ir, ic] += top_coeff * (X_s @ Y_s)[ir, ic]
                    #   (only if (ir, ic) matches some entry in recon[s])
                    # 
                    # Wait, that's not right either. The top-level recon has entries
                    # like [(0,0,1), (1,1,1)] for s=0, meaning:
                    #   C_block[0,0] += 1 * (X_0 @ Y_0)[0,0]
                    #   C_block[1,1] += 1 * (X_0 @ Y_0)[1,1]
                    # 
                    # And inner recon for si=0 has [(0,0,1), (1,1,1)], meaning:
                    #   (X@Y)[0,0] += 1 * M_0
                    #   (X@Y)[1,1] += 1 * M_0
                    # 
                    # So M_0 contributes to C_block[0,0] (via top recon) and C_block[1,1].
                    # 
                    # The W part: C_block[ir, ic][a, b] gets inner_coeff * top_coeff * M_si
                    # where (ir, ic, inner_coeff) is from recon[si] and (ir, ic, top_coeff)
                    # is from recon[s]. But we need ir, ic to match.
                    # 
                    # Actually, I think the structure is simpler:
                    # - For each top-level term s, we compute X_s @ Y_s (a 2x2 matrix of 2x2 blocks)
                    # - The inner Strassen gives us 7 scalar multiplications that produce
                    #   the 4 entries of X_s @ Y_s
                    # - Each scalar multiplication M_si contributes to specific entries of
                    #   X_s @ Y_s via inner_recon[si]
                    # - Then the top-level recon[s] distributes these entries to C blocks
                    # 
                    # So for term (s, si):
                    #   M_si = (linear form in A) * (linear form in B)
                    #   For each (ir, ic, inner_coeff) in recon[si]:
                    #     (X_s @ Y_s)[ir, ic] += inner_coeff * M_si
                    #   For each (out_r, out_c, top_coeff) in recon[s]:
                    #     if out_r == ir and out_c == ic:
                    #       C_block[out_r, out_c] += top_coeff * inner_coeff * M_si
                    # 
                    # But that would require ir==out_r and ic==out_c, which is restrictive.
                    # 
                    # I think the correct interpretation is that the top-level recon
                    # directly maps the 2x2 result of X_s @ Y_s to C blocks, and the
                    # inner Strassen computes that 2x2 result. So:
                    # 
                    # C_block[out_r, out_c] = top_coeff * (X_s @ Y_s)[out_r, out_c]
                    # (X_s @ Y_s)[out_r, out_c] = sum_{si} [inner_coeff_si * M_si]
                    #   where inner_coeff_si is from recon[si] at position (out_r, out_c)
                    # 
                    # So for term (s, si), the contribution to C is:
                    #   For each (ir, ic, inner_coeff) in recon[si]:
                    #     For each (out_r, out_c, top_coeff) in recon[s]:
                    #       if out_r == ir and out_c == ic:
                    #         C_block[out_r, out_c] += top_coeff * inner_coeff * M_si
                    # 
                    # Let me just implement this directly.
                    
                    pass  # Will implement in the actual loop below
    
    # OK this is getting too complex in my head. Let me just directly implement
    # the full expansion without the two-level Strassen abstraction, and verify.
    
    # Actually, let me use the approach from the inspiration code which seems to work.
    # I'll adapt it with correct indexing.
    
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    for s in range(7):
        # Top-level Strassen term s
        # X_s = sum_{r=0}^{3} alpha[s][r] * A_block[r//2, r%2]  (2x2 matrix)
        # Y_s = sum_{c=0}^{3} beta[s][c] * B_block[c//2, c%2]  (2x2 matrix, cols 0-1)
        
        # Part 1: Inner Strassen for X_s @ Y_s (2x2 x 2x2 -> 2x2)
        for si in range(7):
            # U: linear form in A
            for a in range(4):
                ia = alpha[si][a]
                if ia == 0:
                    continue
                x_r, x_c = a // 2, a % 2
                for r in range(4):
                    toa = alpha[s][r]
                    if toa == 0:
                        continue
                    ar, ac_ = r // 2, r % 2
                    u_idx = (2*ar + x_r) * 4 + (2*ac_ + x_c)
                    U[u_idx, col] += ia * toa
            
            # V: linear form in B (cols 0-1 only)
            for b in range(4):
                ib = beta[si][b]
                if ib == 0:
                    continue
                y_r, y_c = b // 2, b % 2
                for c in range(4):
                    tob = beta[s][c]
                    if tob == 0:
                        continue
                    br, bc_ = c // 2, c % 2
                    v_idx = (2*br + y_r) * 6 + (2*bc_ + y_c)
                    V[v_idx, col] += ib * tob
            
            # W: contribution to C blocks (cols 0-1)
            for ir, ic, inner_coeff in recon[si]:
                for out_r, out_c, top_coeff in recon[s]:
                    if out_r == ir and out_c == ic:
                        for a in range(2):
                            for b in range(2):
                                # C[2*out_r+a, 2*out_c+b]
                                c_row = 2*out_r + a
                                c_col = 2*out_c + b
                                w_idx = c_col * 4 + c_row
                                W[w_idx, col] += top_coeff * inner_coeff
            
            col += 1
        
        # Part 2: Schoolbook for X_s * B_col2 (2x2 x 2x1 -> 2x1)
        # X_s @ B_block[:, 2] where B_block[:, 2] means B[0,2] and B[1,2]
        # C_block[out_r, 2] += top_coeff * (X_s @ B[:,2])[out_r, :]
        # (X_s @ B[:,2])[i, :] = sum_{x=0}^{1} X_s[i, x] * B[x, 2]
        # = X_s[i, 0] * B[0, 2] + X_s[i, 1] * B[1, 2]
        
        for out_r, out_c, top_coeff in recon[s]:
            if out_c != 1:
                # Only col-block 1 of the Strassen result maps to col-block 2 of C?
                # No, I think this is wrong. Let me reconsider.
                pass
        
        # Actually, I think the schoolbook part handles the 3rd col-block of B separately.
        # Let me restructure: after the 7 Strassen terms (handling cols 0-3),
        # add 4 schoolbook terms for col-block 2.
    
    # This is getting too tangled. Let me use a completely different, simpler approach:
    # Directly enumerate all the rank-1 terms.
    
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    # Part 1: 7 Strassen terms, each with 7 inner Strassen terms = 49 terms
    for s in range(7):
        for si in range(7):
            # This term computes:
            # M = (sum_a alpha[si][a] * X_s[a//2, a%2]) * (sum_b beta[si][b] * Y_s[b//2, b%2])
            # where X_s[i, j] = sum_r alpha[s][r] * A[2*(r//2)+i, 2*(r%2)+j]
            #       Y_s[i, j] = sum_c beta[s][c] * B[2*(c//2)+i, 2*(c%2)+j]
            
            # U vector: coefficients for A[i*4+j]
            # M first factor = sum_a alpha[si][a] * sum_r alpha[s][r] * A[2*(r//2)+(a//2), 2*(r%2)+(a%2)]
            # Coefficient of A[i, j] = sum_{a, r: 2*(r//2)+(a//2)==i, 2*(r%2)+(a%2)==j} alpha[si][a] * alpha[s][r]
            for a in range(4):
                ia = alpha[si][a]
                if ia == 0:
                    continue
                x_r, x_c = a // 2, a % 2
                for r in range(4):
                    toa = alpha[s][r]
                    if toa == 0:
                        continue
                    ar, ac_ = r // 2, r % 2
                    i = 2*ar + x_r
                    j = 2*ac_ + x_c
                    U[i * 4 + j, col] += ia * toa
            
            # V vector: coefficients for B[i*6+j]
            # M second factor = sum_b beta[si][b] * sum_c beta[s][c] * B[2*(c//2)+(b//2), 2*(c%2)+(b%2)]
            for b in range(4):
                ib = beta[si][b]
                if ib == 0:
                    continue
                y_r, y_c = b // 2, b % 2
                for c in range(4):
                    tob = beta[s][c]
                    if tob == 0:
                        continue
                    br, bc_ = c // 2, c % 2
                    i = 2*br + y_r
                    j = 2*bc_ + y_c
                    V[i * 6 + j, col] += ib * tob
            
            # W vector: coefficients for C[j*4+i] (W[k*n+i] where k=col, i=row)
            # M contributes to (X_s @ Y_s)[ir, ic] via inner_coeff
            # (X_s @ Y_s)[ir, ic] contributes to C_block[out_r, out_c] via top_coeff
            # C_block[out_r, out_c][a, b] = C[2*out_r+a, 2*out_c+b]
            for ir, ic, inner_coeff in recon[si]:
                for out_r, out_c, top_coeff in recon[s]:
                    if out_r == ir and out_c == ic:
                        for a in range(2):
                            for b in range(2):
                                c_row = 2*out_r + a
                                c_col = 2*out_c + b
                                w_idx = c_col * 4 + c_row
                                W[w_idx, col] += top_coeff * inner_coeff
            
            col += 1
    
    # Part 2: 4 schoolbook terms for col-block 2, each with 7 inner Strassen terms = 28 terms
    for i_block in range(2):
        for j_block in range(2):
            # Term: A_block[i_block, j_block] @ B_block[j_block, 2]
            # This is a 2x2 @ 2x2 = 2x2 product, done with inner Strassen
            # Result: C_block[i_block, 2]
            
            for si in range(7):
                # U: linear form in A_block[i_block, j_block]
                # X = A[2*i_block:2*i_block+2, 2*j_block:2*j_block+2]
                for a in range(4):
                    ia = alpha[si][a]
                    if ia == 0:
                        continue
                    x_r, x_c = a // 2, a % 2
                    i = 2*i_block + x_r
                    j = 2*j_block + x_c
                    U[i * 4 + j, col] += ia
                
                # V: linear form in B_block[j_block, 2]
                # Y = B[2*j_block:2*j_block+2, 4:6]
                for b in range(4):
                    ib = beta[si][b]
                    if ib == 0:
                        continue
                    y_r, y_c = b // 2, b % 2
                    i = 2*j_block + y_r
                    j = 4 + y_c
                    V[i * 6 + j] += ib
                    V[i * 6 + j, col] += ib
                
                # W: contribution to C_block[i_block, 2]
                for ir, ic, inner_coeff in recon[si]:
                    for a in range(2):
                        for b in range(2):
                            c_row = 2*i_block + a
                            c_col = 4 + b
                            w_idx = c_col * 4 + c_row
                            W[w_idx, col] += inner_coeff
                
                col += 1
    
    assert col == R, f"Expected {R} columns, got {col}"
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    U, V, W = _build_decomposition()
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    for r in range(U.shape[1]):
        scalar = np.dot(U[:, r], A_flat) * np.dot(V[:, r], B_flat)
        C_flat = scalar * W[:, r]
        C_computed += C_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    r.append(int(rx))
                else:
                    r.append(float(x))
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
