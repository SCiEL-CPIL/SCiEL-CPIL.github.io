"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 3, 4, 8

    # We'll use a hybrid approach:
    # 1. Use Strassen-like rank-7 decomposition for 2x2x2 blocks where possible
    # 2. Use schoolbook for the rest
    # 
    # For 3x4 * 4x8:
    # We can partition the 4x4 middle dimension into 2x2 blocks
    # The input A is 3x4, B is 4x8, output C is 3x8
    #
    # Let's think about this differently.
    # A is n x m = 3 x 4
    # B is m x p = 4 x 8
    # C is n x p = 3 x 8
    #
    # We can split m=4 into two groups of 2: m1=2, m2=2
    # A = [A1 | A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8
    # C = A1*B1 + A2*B2
    #
    # For each 3x2 * 2x8 product, we can further decompose.
    # Actually, let's use the known result that rank(3,4,8) can be improved.
    #
    # A better approach: use the fact that we can apply Strassen to 2x2x2 subproblems.
    # 
    # Let's try a different partitioning:
    # Split p=8 into four groups of 2: p1=2, p2=2, p3=2, p4=2
    # Split m=4 into two groups of 2: m1=2, m2=2
    # 
    # C = A * B where A is 3x4, B is 4x8
    # 
    # Let's use the following strategy:
    # For each pair of 2-column blocks of B (there are 4 such blocks),
    # and for each of the 2 row-blocks of A (splitting m=4 into 2+2),
    # we have a 3x2 * 2x2 subproblem.
    # 
    # Actually, let me reconsider. The tensor T_{i,j,k} = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Known lower bounds: rank(3,4,8) >= 24 (trivially n*p = 24)
    # Schoolbook gives 3*4*8 = 96
    # 
    # Let's use a simple improvement: exploit the structure.
    # For fixed i (row of A, 0..2) and k (col of C, 0..7), 
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # One approach: group terms.
    # (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #
    # For the first group, we can use a rank-2 decomposition of 1x2*2x1 type,
    # but that's just 2 terms. No savings.
    #
    # Let me try using Strassen for 2x2x2 blocks.
    # If we split m=4 into 2+2 and p=8 into 2+2+2+2:
    # 
    # A = [A1 A2] (3x2 each)
    # B = [B1; B2; B3; B4] (2x2 each, 4 blocks)
    # 
    # C = A1*B1 + A2*B1 + A1*B2 + A2*B2 + A1*B3 + A2*B3 + A1*B4 + A2*B4
    #
    # Each term like A1*B1 is 3x2 * 2x2 = 3x2. 
    # This can be done in rank 3*2*2/2... no.
    # Actually 3x2 * 2x2: the tensor has dimensions 3,2,2. Rank is at most 3*2*2=12, 
    # but we can do better. Actually for 3x2*2x2, the rank is 6 (since min(3*2, 2*2, 3*2)=... 
    # no, the rank of matrix multiplication tensor [a,b,c] is the tensor rank.
    # For [3,2,2], the rank is 6? Let me think. 
    # Actually [3,2,2] can be done in rank 6 by schoolbook, and I believe that's optimal.
    #
    # Hmm, this is getting complicated. Let me just implement a known good decomposition.
    #
    # Actually, the simplest improvement: use the fact that for the 4x4 middle,
    # we can use rank-7 Strassen-like for 2x2x2 blocks.
    #
    # Let me try: partition m=4 into 2+2, and for each of the n=3 rows and p=8 cols,
    # we need to compute sum over 4 middle indices.
    #
    # Alternative: just use a greedy/optimization approach to find a lower rank.
    # But that's hard.
    #
    # Let me just implement the Strassen-based approach:
    # Split m into 2+2. For each i in 0..2, k in 0..7:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #        = (A1 row i) dot (B1 col k) + (A2 row i) dot (B2 col k)
    #
    # where A1 = A[:, 0:2], A2 = A[:, 2:4], B1 = B[0:2, :], B2 = B[2:4, :]
    #
    # Each inner product is of length 2, so it's just 2 multiplications.
    # Total: 3 * 8 * 2 * 2 = 96. Same as schoolbook. No improvement.
    #
    # To get improvement, we need to use Strassen on 2x2x2 blocks.
    # Strassen computes 2x2 * 2x2 in rank 7 instead of 8.
    #
    # So let's partition:
    # n=3, m=4, p=8
    # Split m=4 into 2+2
    # Split p=8 into 2+2+2+2
    # 
    # Now C = A * B
    # A is 3x4 = [A_left (3x2) | A_right (3x2)]
    # B is 4x8 = [B_top (2x8); B_bottom (2x8)]
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # Now A_left * B_top is 3x2 * 2x8.
    # Split B_top into 4 blocks of 2x2: B_top = [Bt1 | Bt2 | Bt3 | Bt4]
    # A_left * B_top = A_left * Bt1 + A_left * Bt2 + A_left * Bt3 + A_left * Bt4
    # 
    # Each A_left * Btj is 3x2 * 2x2. 
    # For 3x2 * 2x2, we can't directly use Strassen (which is for 2x2*2x2).
    # But we can split the 3 rows into 2+1:
    # A_left = [A_top (2x2); A_bottom (1x2)]
    # A_left * Btj = A_top * Btj + A_bottom * Btj
    # A_top * Btj is 2x2 * 2x2 -> rank 7 (Strassen)
    # A_bottom * Btj is 1x2 * 2x2 -> rank 2 (schoolbook: 1*2*2=4? No, 1x2*2x2: 
    #   result is 1x2, sum over 2 middle: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k], rank 2)
    # 
    # So for each of the 4 Btj blocks:
    # A_top * Btj: rank 7
    # A_bottom * Btj: rank 2
    # Total per Btj: 9
    # For 4 Btj: 36
    # 
    # Similarly for A_right * B_bottom: another 36
    # Total: 72
    #
    # That's an improvement from 96 to 72!
    #
    # Let me implement this.
    
    # Strassen 2x2*2x2 decomposition (rank 7)
    # For matrices X (2x2), Y (2x2), Z = X*Y
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    def strassen_decomp():
        """Return U,V,W for 2x2*2x2 Strassen, rank 7.
        U has shape (2*2, 7) = (4,7), V has shape (2*2, 7) = (4,7), W has shape (2*2, 7) = (4,7)
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (X00+X11)*(Y00+Y11)
        # X00 is U[0], X11 is U[3], Y00 is V[0], Y11 is V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # Z00 += M1, Z11 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (X10+X11)*Y00
        # X10 is U[2], X11 is U[3], Y00 is V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # Z10 += M2
        W[2][1] = 1
        
        # M3 = X00*(Y01-Y11)
        # X00 is U[0], Y01 is V[1], Y11 is V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # Z01 += M3, Z11 += M3
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = X11*(Y10-Y00)
        # X11 is U[3], Y10 is V[2], Y00 is V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # Z00 += M4, Z10 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (X00+X01)*Y11
        # X00 is U[0], X01 is U[1], Y11 is V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # Z00 -= M5
        W[0][4] = -1
        
        # M6 = (X10-X00)*(Y00+Y01)
        # X10 is U[2], X00 is U[0], Y00 is V[0], Y01 is V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # Z11 += M6
        W[3][5] = 1
        
        # M7 = (X01-X11)*(Y10+Y11)
        # X01 is U[1], X11 is U[3], Y10 is V[2], Y11 is V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # Z00 += M7
        W[0][6] = 1
        
        return U, V, W
    
    # Build the full decomposition
    # Partition:
    # n=3, m=4, p=8
    # Split m into m1=2 (indices 0,1) and m2=2 (indices 2,3)
    # Split n into n1=2 (indices 0,1) and n2=1 (index 2)
    # Split p into 4 blocks of 2: p1=2 (0,1), p2=2 (2,3), p3=2 (4,5), p4=2 (6,7)
    
    # For A_left (n x m1 = 3x2) * B_top (m1 x p = 2x8):
    #   Split A_left into A_top (2x2) and A_bottom (1x2)
    #   Split B_top into 4 blocks of 2x2
    #   For each block: A_top * Btj (2x2*2x2, rank 7) + A_bottom * Btj (1x2*2x2, rank 2)
    # Similarly for A_right (3x2) * B_bottom (2x8)
    
    # Total rank: 2 * 4 * (7 + 2) = 2 * 4 * 9 = 72
    
    rank = 72
    U = [[0]*rank for _ in range(n*m)]  # 12 x 72
    V = [[0]*rank for _ in range(m*p)]  # 32 x 72
    W = [[0]*rank for _ in range(p*n)]  # 24 x 72
    
    col = 0
    
    # Helper to add a 1x2 * 2x2 schoolbook block
    # A is 1x2 (row i of A, cols j1,j2), B is 2x2 (rows j1,j2, cols k1,k2)
    # Result is 1x2: C[i,k1] = A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
    # C[i,k2] = A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    def add_1x2_2x2_block(i, j1, j2, k1, k2, col):
        """Add 2 rank-1 terms for 1x2 * 2x2."""
        nonlocal col
        # Term 1: A[i,j1] * B[j1,k1] -> C[i,k1]
        U[i*m + j1][col] = 1
        V[j1*p + k1][col] = 1
        W[k1*n + i][col] = 1
        col += 1
        # Term 2: A[i,j2] * B[j2,k2] -> C[i,k2]
        U[i*m + j2][col] = 1
        V[j2*p + k2][col] = 1
        W[k2*n + i][col] = 1
        col += 1
        return col
    
    # Helper to add Strassen 2x2*2x2 block
    def add_strassen_block(i1, i2, j1, j2, k1, k2, col):
        """Add 7 rank-1 terms for 2x2 * 2x2 using Strassen.
        i1,i2 are row indices, j1,j2 are col indices of A (and row indices of B),
        k1,k2 are col indices of B (and result).
        """
        nonlocal col
        sU, sV, sW = strassen_decomp()
        for c in range(7):
            # U entries
            for ui in range(4):
                if sU[ui][c] != 0:
                    # ui = 0: (i1,j1), ui=1: (i1,j2), ui=2: (i2,j1), ui=3: (i2,j2)
                    if ui == 0:
                        U[i1*m + j1][col + c] += sU[ui][c]
                    elif ui == 1:
                        U[i1*m + j2][col + c] += sU[ui][c]
                    elif ui == 2:
                        U[i2*m + j1][col + c] += sU[ui][c]
                    elif ui == 3:
                        U[i2*m + j2][col + c] += sU[ui][c]
            # V entries
            for vi in range(4):
                if sV[vi][c] != 0:
                    # vi = 0: (j1,k1), vi=1: (j1,k2), vi=2: (j2,k1), vi=3: (j2,k2)
                    if vi == 0:
                        V[j1*p + k1][col + c] += sV[vi][c]
                    elif vi == 1:
                        V[j1*p + k2][col + c] += sV[vi][c]
                    elif vi == 2:
                        V[j2*p + k1][col + c] += sV[vi][c]
                    elif vi == 3:
                        V[j2*p + k2][col + c] += sV[vi][c]
            # W entries
            for wi in range(4):
                if sW[wi][c] != 0:
                    # wi = 0: (k1,i1), wi=1: (k1,i2), wi=2: (k2,i1), wi=3: (k2,i2)
                    if wi == 0:
                        W[k1*n + i1][col + c] += sW[wi][c]
                    elif wi == 1:
                        W[k1*n + i2][col + c] += sW[wi][c]
                    elif wi == 2:
                        W[k2*n + i1][col + c] += sW[wi][c]
                    elif wi == 3:
                        W[k2*n + i2][col + c] += sW[wi][c]
        col += 7
        return col
    
    # Process A_left * B_top:
    # A_left: rows 0-2, cols 0-1 (m indices 0,1)
    # B_top: rows 0-1, cols 0-7 (m indices 0,1; p indices 0-7)
    # Split A_left into A_top (rows 0-1) and A_bottom (row 2)
    # Split B_top into 4 blocks of 2 cols each
    
    for block in range(4):
        k1 = block * 2
        k2 = block * 2 + 1
        # A_top (2x2) * Bt_block (2x2): Strassen, rank 7
        col = add_strassen_block(0, 1, 0, 1, k1, k2, col)
        # A_bottom (1x2) * Bt_block (2x2): schoolbook, rank 2
        col = add_1x2_2x2_block(2, 0, 1, k1, k2, col)
    
    # Process A_right * B_bottom:
    # A_right: rows 0-2, cols 2-3 (m indices 2,3)
    # B_bottom: rows 2-3, cols 0-7 (m indices 2,3; p indices 0-7)
    # Split A_right into A_top (rows 0-1) and A_bottom (row 2)
    # Split B_bottom into 4 blocks of 2 cols each
    
    for block in range(4):
        k1 = block * 2
        k2 = block * 2 + 1
        # A_top (2x2) * Bb_block (2x2): Strassen, rank 7
        col = add_strassen_block(0, 1, 2, 3, k1, k2, col)
        # A_bottom (1x2) * Bb_block (2x2): schoolbook, rank 2
        col = add_1x2_2x2_block(2, 2, 3, k1, k2, col)
    
    # Verify col == rank
    # col should be 72
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
