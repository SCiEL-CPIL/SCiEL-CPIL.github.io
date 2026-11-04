import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2x2
    # Each component is a list of (index, value)
    # U index: i * 2 + j
    # V index: j * 2 + k
    # W index: k * 2 + i
    
    sU = [
        [(0, 1), (3, 1)],      # 0
        [(2, 1), (3, 1)],      # 1
        [(0, 1)],              # 2
        [(3, 1)],              # 3
        [(0, 1), (1, 1)],      # 4
        [(0, 1), (2, -1)],     # 5
        [(1, 1), (3, -1)],     # 6
    ]
    sV = [
        [(0, 1), (3, 1)],      # 0
        [(0, 1)],              # 1
        [(1, 1), (3, -1)],     # 2
        [(2, 1), (0, -1)],     # 3
        [(3, 1)],              # 4
        [(0, 1), (1, 1)],      # 5
        [(2, 1), (3, 1)],      # 6
    ]
    sW = [
        [(0, 1), (3, 1)],      # 0
        [(1, 1), (3, -1)],     # 1
        [(2, 1), (3, 1)],      # 2
        [(0, 1), (1, 1)],      # 3
        [(0, -1), (2, 1)],     # 4
        [(3, 1)],              # 5
        [(0, 1)],              # 6
    ]
    
    # Block offsets for the 8 blocks
    # bI in {0,1}, bJ in {0,1,2,3}, bK in {0,1}
    # A is 2x4, B is 4x8, C is 2x8? No C is 2x8?
    # A (2x4) * B (4x8) -> C (2x8).
    # Wait, C is 2x8.
    # My previous logic:
    # A0 (2x2) * Bt0 (2x2) -> C0 (2x2).
    # Bt0 is cols 0,1 of B_top.
    # B_top is rows 0,1 of B.
    # So Bt0 is B[0:2, 0:2].
    # Bt1 is B[0:2, 2:4].
    # Bt2 is B[0:2, 4:6].
    # Bt3 is B[0:2, 6:8].
    # Bb0 is B[2:4, 0:2].
    # Bb1 is B[2:4, 2:4].
    # Bb2 is B[2:4, 4:6].
    # Bb3 is B[2:4, 6:8].
    
    # Block (bi, bj) where bi is A-row-block (0,1), bj is B-col-block (0..3) ?
    # No, A has 2 col-blocks. B has 2 row-blocks.
    # A0 pairs with B_top. A1 pairs with B_bottom.
    # So bi determines A block and B row block.
    # bj determines B col block.
    
    # bi=0: A0 (rows 0,1 of A? No, A is 2x4. A0 is cols 0,1? No.
    # A = [A0 | A1]. A0 is 2x2 (cols 0,1). A1 is 2x2 (cols 2,3).
    # B = [B_top; B_bottom]. B_top is rows 0,1. B_bottom is rows 2,3.
    # A0 multiplies B_top. A1 multiplies B_bottom.
    
    # So for A0 (i_base=0, j_base=0 in A indices? No.
    # A index (i, j). i is row 0,1. j is col 0..3.
    # A0 uses j=0,1. A1 uses j=2,3.
    # B index (j, k). j is row 0..3. k is col 0..7.
    # B_top uses j=0,1. B_bottom uses j=2,3.
    # So A0 (j=0,1) multiplies B_top (j=0,1).
    # A1 (j=2,3) multiplies B_bottom (j=2,3).
    
    # The "j" index in A must match the "j" index in B for the multiplication to be valid?
    # Yes, A[i,j] * B[j,k].
    # So for A0, j is 0,1. B must use rows 0,1.
    # For A1, j is 2,3. B must use rows 2,3.
    
    # Now partition B into col blocks of size 2.
    # k=0,1; k=2,3; k=4,5; k=6,7.
    # There are 4 such blocks.
    # So we have 2 (A blocks) * 4 (B col blocks) = 8 total 2x2x2 multiplications.
    
    # Block id b from 0 to 7.
    # b = a_block * 4 + b_col_block.
    # a_block 0: A j_base=0, B j_base=0.
    # a_block 1: A j_base=2, B j_base=2.
    # b_col_block 0: B k_base=0.
    # b_col_block 1: B k_base=2.
    # b_col_block 2: B k_base=4.
    # b_col_block 3: B k_base=6.
    
    for b in range(8):
        a_block = b // 4
        b_col = b % 4
        
        if a_block == 0:
            i_base = 0
            j_base = 0
        else:
            i_base = 0 # A rows are always 0,1
            j_base = 2
        
        k_base = b_col * 2
        
        # Add Strassen components
        for s in range(7):
            r = b * 7 + s
            
            # Update U
            for idx, val in sU[s]:
                # idx is local 2x2 index. i = idx // 2, j = idx % 2
                li = idx // 2
                lj = idx % 2
                gi = i_base + li # i_base is 0
                gj = j_base + lj
                U[gi * 4 + gj][r] += val
                
            # Update V
            for idx, val in sV[s]:
                # idx is local 2x2 index. j = idx // 2, k = idx % 2
                lj = idx // 2
                lk = idx % 2
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][r] += val
                
            # Update W
            for idx, val in sW[s]:
                # idx is local 2x2 index. k = idx // 2, i = idx % 2
                lk = idx // 2
                li = idx % 2
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
