DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll decompose using block structure:
    # m = 3 = 2 + 1
    # p = 8 = 4 + 4
    # 
    # The multiplication 2x3 * 3x8 can be seen as:
    # [A1 A2] where A1 is 2x2, A2 is 2x1 (splitting columns of first matrix)
    # [B1 B2] where B1 is 3x4, B2 is 3x4 (splitting columns of second matrix)
    # But we need to split rows of second matrix and columns of first matrix consistently.
    #
    # Actually, the contraction is over the middle dimension m=3.
    # Let's split m into m1=2 and m2=1.
    # Split p into p1=4 and p2=4.
    #
    # Then the product is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j in 1..2} A[i,j] * B[j,k] + sum_{j in 3..3} A[i,j] * B[j,k]
    #
    # For k in 1..4: C[i,k] = (A1[i,:] dot B1[:,k]) + A2[i,3] * B2[3,k]
    # For k in 5..8: C[i,k] = (A1[i,:] dot B1[:,k]) + A2[i,3] * B2[3,k]
    #
    # So we have 4 independent sub-problems:
    # 1. 2x2 * 2x4 (for i=1,2; j=1,2; k=1..4)
    # 2. 2x2 * 2x4 (for i=1,2; j=1,2; k=5..8)  
    # 3. 2x1 * 1x4 (for i=1,2; j=3; k=1..4)
    # 4. 2x1 * 1x4 (for i=1,2; j=3; k=5..8)
    
    # Rank of 2x2 * 2x4: use tensor product of Strassen (rank 7 for 2x2*2x2) and identity (rank 2 for 1x2*2x2... wait)
    # Actually, 2x2 * 2x4 can be done by applying a 2x2*2x2 algorithm to each pair of columns.
    # So rank = 7 * 2 = 14? No, that's not right either.
    #
    # Better: R(2,2,4) <= R(2,2,2) * R(1,2,2) = 7 * 2 = 14
    # Where R(1,2,2) = 2 (just scaling)
    #
    # The tensor product construction: if we have a rank-r decomposition for n1xm1 * m1xp1,
    # and rank-s for n2xm2 * m2xp2, then we have rank r*s for (n1*n2)x(m1*m2) * (m1*m2)x(p1*p2).
    #
    # For 2x2 * 2x4: write as (2x1 * 1x2) tensor (1x2 * 2x2)
    # R(2,1,2) = 2, R(1,2,2) = 2, so R(2,2,4) <= 2*2 = 4? That seems too good.
    # Wait, that's not the right factorization.
    #
    # Let me think again. 2x2 * 2x4:
    # We can view this as two separate 2x2 * 2x2 multiplications (columns 1-2 and 3-4).
    # Each has rank 7, so total rank 14.
    #
    # Alternatively, use the fact that 2x2 * 2x4 = (2x2 * 2x2) block-diagonal with (2x2 * 2x2)
    # So rank = 7 + 7 = 14.
    
    # For 2x1 * 1x4: rank = 2*4 = 8 (no savings possible)
    
    # Total rank = 2 * 14 + 2 * 8 = 28 + 16 = 44
    
    # Now I need to construct the actual factor matrices.
    # This is complex. Let me use a simpler approach: just use schoolbook but see if I can find
    # a known low-rank algorithm.
    
    # Actually, let me just implement the schoolbook for now and verify it works,
    # then try to optimize.
    
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
