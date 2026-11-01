"""Matrix multiplication 2x5x6 rank-minimal exact decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # We'll build a decomposition with rank R.
    # Strategy: Use Strassen on 2x2x2 sub-blocks where possible.
    # 
    # The 2x5x6 multiplication can be viewed as:
    # - Rows of A: 2 (i=0,1)
    # - Cols of A / rows of B: 5 (j=0,1,2,3,4)
    # - Cols of B: 6 (k=0,1,2,3,4,5)
    #
    # We can partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}
    # For the 2x2x2 blocks (j-pair, k-pair), we use Strassen (rank 7 each)
    # For the remaining 2x1x2 blocks (j=4, k-pair), we use schoolbook (rank 4 each)
    #
    # Number of 2x2x2 blocks: 2 j-pairs * 3 k-pairs = 6 blocks, each rank 7 = 42
    # Number of 2x1x2 blocks: 1 j-single * 3 k-pairs = 3 blocks, each rank 4 = 12
    # Total rank: 42 + 12 = 54
    
    rank = 54
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 54
    
    col = 0
    
    # Strassen 2x2x2 decomposition for sub-blocks
    # A_sub = [[a, b], [c, d]] where a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
    # B_sub = [[e, f], [g, h]] where e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
    # C_sub = A_sub * B_sub = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # Strassen products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1, start_col):
        """Add a Strassen 2x2x2 block to the decomposition."""
        # p1 = (a+d)(e+h)
        # a = A[i0,j0], d = A[i1,j1], e = B[j0,k0], h = B[j1,k1]
        U[i0 * m + j0][start_col] = 1.0
        U[i1 * m + j1][start_col] = 1.0
        V[j0 * p + k0][start_col] = 1.0
        V[j1 * p + k1][start_col] = 1.0
        # C[k0*n + i0] gets +1, C[k1*n + i1] gets +1 (from p1)
        W[k0 * n + i0][start_col] = 1.0
        W[k1 * n + i1][start_col] = 1.0
        col_local = start_col + 1
        
        # p2 = (c+d)e
        # c = A[i1,j0], d = A[i1,j1], e = B[j0,k0]
        U[i1 * m + j0][col_local] = 1.0
        U[i1 * m + j1][col_local] = 1.0
        V[j0 * p + k0][col_local] = 1.0
        # C[k0*n + i1] gets +1
        W[k0 * n + i1][col_local] = 1.0
        col_local += 1
        
        # p3 = a(f-h)
        # a = A[i0,j0], f = B[j0,k1], h = B[j1,k1]
        U[i0 * m + j0][col_local] = 1.0
        V[j0 * p + k1][col_local] = 1.0
        V[j1 * p + k1][col_local] = -1.0
        # C[k1*n + i0] gets +1
        W[k1 * n + i0][col_local] = 1.0
        col_local += 1
        
        # p4 = d(g-e)
        # d = A[i1,j1], g = B[j1,k0], e = B[j0,k0]
        U[i1 * m + j1][col_local] = 1.0
        V[j1 * p + k0][col_local] = 1.0
        V[j0 * p + k0][col_local] = -1.0
        # C[k0*n + i1] gets +1, C[k1*n + i0] gets -1 (wait, let me recheck)
        # Actually p4 contributes to C[0,0] and C[1,0]
        # C[0,0] = p1 + p4 - p5 + p7, so p4 adds to C[0,0] which is C[k0*n+i0]
        # C[1,0] = p2 + p4, so p4 adds to C[1,0] which is C[k0*n+i1]
        W[k0 * n + i0][col_local] = 1.0
        W[k0 * n + i1][col_local] = 1.0
        col_local += 1
        
        # p5 = (a+b)h
        # a = A[i0,j0], b = A[i0,j1], h = B[j1,k1]
        U[i0 * m + j0][col_local] = 1.0
        U[i0 * m + j1][col_local] = 1.0
        V[j1 * p + k1][col_local] = 1.0
        # C[0,0] = ... - p5 ..., so C[k0*n+i0] gets -1
        # C[0,1] = p3 + p5, so C[k1*n+i0] gets +1
        W[k0 * n + i0][col_local] = -1.0
        W[k1 * n + i0][col_local] = 1.0
        col_local += 1
        
        # p6 = (c-a)(e+f)
        # c = A[i1,j0], a = A[i0,j0], e = B[j0,k0], f = B[j0,k1]
        U[i1 * m + j0][col_local] = 1.0
        U[i0 * m + j0][col_local] = -1.0
        V[j0 * p + k0][col_local] = 1.0
        V[j0 * p + k1][col_local] = 1.0
        # C[1,1] = p1 - p2 + p3 + p6, so C[k1*n+i1] gets +1
        W[k1 * n + i1][col_local] = 1.0
        col_local += 1
        
        # p7 = (b-d)(g+h)
        # b = A[i0,j1], d = A[i1,j1], g = B[j1,k0], h = B[j1,k1]
        U[i0 * m + j1][col_local] = 1.0
        U[i1 * m + j1][col_local] = -1.0
        V[j1 * p + k0][col_local] = 1.0
        V[j1 * p + k1][col_local] = 1.0
        # C[0,0] = ... + p7 ..., so C[k0*n+i0] gets +1
        W[k0 * n + i0][col_local] = 1.0
        col_local += 1
        
        return col_local
    
    # Add 6 Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i0, i1 in [(0, 1)]:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                col = add_strassen_block(i0, i1, j0, j1, k0, k1, col)
    
    # Now add schoolbook for j=4, all k=0..5
    # Each term A[*,4] * B[4,k] contributes to C[k,*]
    # For i in {0,1}, k in {0,1,2,3,4,5}: 12 terms
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] = 1.0
            V[4 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
