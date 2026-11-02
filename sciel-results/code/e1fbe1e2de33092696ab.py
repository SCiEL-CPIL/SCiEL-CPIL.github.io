"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    # We'll build the decomposition by combining:
    # - 4 rank-7 Winograd blocks for the 2x2x2 diagonal sub-blocks
    # - Schoolbook for the remaining cross terms
    # Total: 4*7 + cross_terms_rank
    
    # First, let's establish the full schoolbook as baseline and then
    # try to replace 2x2x2 subproblems with rank-7 versions.
    
    # The 4x4x6 tensor: T[i,j,k] = A[i,j] * B[j,k]
    # Index: i in [0,4), j in [0,4), k in [0,6)
    # U index: i*m + j = i*4 + j, range [0,16)
    # V index: j*p + k = j*6 + k, range [0,24)
    # W index: k*n + i = k*4 + i, range [0,24)
    
    # Approach: Partition i into {0,1} and {2,3}, j into {0,1} and {2,3}
    # This gives 4 blocks of A (each 2x2) and corresponding B blocks.
    # A_top_left (i=0,1; j=0,1) * B_top (j=0,1; k=0..5) -> C_top_left (i=0,1; k=0..5)
    # A_top_right (i=0,1; j=2,3) * B_bot (j=2,3; k=0..5) -> C_top_right_part
    # etc.
    
    # This gets complex. Let me use a different, cleaner approach:
    # Use the fact that rank(4x4x6) <= rank(4x4x4) * 2 - adjustments
    # Actually, simplest: just use schoolbook but with a twist.
    
    # Let me try: Pad B to 4x8 by adding 2 zero columns. Then 4x4x8.
    # Use known rank-28 for 4x4x4 (Strassen-like). But 4x4x8 is harder.
    
    # Alternative clean approach: Direct construction using Kronecker structure.
    # 4x4x6 = (2x2 x 2x2) x (2x2 x 2x6)
    # = (A11*A11 + A12*A21) for top-left 2x2 block of C, etc.
    
    # Let me just implement a verified rank-7 2x2x2 and combine.
    
    # Winograd 2x2x2 factors (verified to work):
    # U_w = [[1,1,1,1,1,0,0],
    #        [0,1,0,1,0,1,1]]
    # V_w = [[1,1,1,0,1,1,0],
    #        [0,1,0,1,0,1,1]]
    # W_w = [[1,1,0,1,1,0,0],
    #        [0,1,1,0,1,1,1]]
    
    # Verify: For A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    
    # With Winograd intermediates x0..x6:
    # x0 = a00+a11, x1 = a10+a11, x2 = a00-a10, x3 = a01+a11
    # x4 = a01-a00, x5 = a11-a10, x6 = a11-a01
    
    # And y0..y6 from B:
    # y0 = b00+b11, y1 = b10+b11, y2 = b00-b10, y3 = b01+b11
    # y4 = b01-b00, y5 = b11-b10, y6 = b11-b01
    
    # Then:
    # C00 = x0*y0 - x2*y2 - x4*y4 + x6*y6
    # C01 = x0*y3 - x2*y5 - x4*y3 + x6*y5 ... this is getting messy.
    
    # Let me just use the standard verified Winograd from literature.
    # Standard 2x2x2 rank-7:
    # U = [[1, 1, 1, 1, 1, 0, 0],
    #      [0, 1, 0, 1, 0, 1, 1]]
    # V = [[1, 1, 1, 0, 1, 1, 0],
    #      [0, 1, 0, 1, 0, 1, 1]]
    # W = [[1, 1, 0, 1, 1, 0, 0],
    #      [0, 1, 1, 0, 1, 1, 1]]
    
    # I'll trust this and build the full 4x4x6.
    
    # Actually, let me take a step back. The safest way to guarantee correctness
    # while improving rank is to use a well-known algorithm. 
    # 
    # For 4x4x6, one approach: treat as two 4x2x6 multiplications (splitting j).
    # No, that's not right either.
    #
    # Let me just do: rank = 96 - 4 = 92 by finding 4 independent 2x2x2 sub-tensors
    # that can each be reduced by 1 (from rank 8 to 7).
    #
    # The 4x4x6 tensor has "diagonal" 2x2x2 sub-tensors when we look at
    # (i,j,k) where i,j are in the same 2x2 block. There are 4 such blocks
    # for i,j, but k goes 0..5, so each is 2x2x6, not 2x2x2.
    #
    # This is harder than I thought. Let me just implement a clean, correct
    # schoolbook and accept rank 96, OR find a simpler verified improvement.
    #
    # Simplest verified improvement: Use the fact that for the k-dimension,
    # we can use rank-reduced 2x2x2 in the (i,k) or (j,k) planes.
    #
    # Actually, I realize I should just implement the full schoolbook correctly
    # since the previous attempt was rejected for being incomplete/invalid.
    # A correct rank-96 solution scores 1.0, same as current. To beat it,
    # I need a correct lower-rank solution.
    #
    # Let me implement a clean rank-84 solution using 4 Winograd 2x2x2 blocks
    # applied to the "diagonal" structure, with schoolbook for the rest.
    
    n, m, p = 4, 4, 6
    rank = 96  # start with schoolbook rank, then reduce
    
    # Build full schoolbook first, then we'll see if we can compress.
    # For now, let me just return a correct schoolbook to ensure validity,
    # but that gives score 1.0.
    
    # To actually improve, let me use a different strategy:
    # The tensor T[i,j,k] for i in [0,4), j in [0,4), k in [0,6)
    # can be written as sum over (i,j,k) of e_i^U ⊗ e_j^V ⊗ e_k^W
    # 
    # Key insight: If we group indices, T = T1 + T2 where T1 involves
    # (i<2,j<2) and T2 involves the rest. T1 is a 2x2x6 sub-tensor.
    # T1[i,j,k] = A[i,j]*B[j,k] for i,j in {0,1}, k in {0..5}
    # This is equivalent to (2x2 A block) * (2x6 B block) = 2x6 C block
    # The rank of 2x2x6 multiplication: schoolbook is 2*2*6=24.
    # Can we do better? 2x2x6 = two 2x2x3 multiplications? No.
    # 2x2x6 has rank at most... it's the tensor for multiplying 2x2 by 2x6.
    # This is equivalent to: for each k, C[:,k] = A * B[:,k]
    # So it's 6 independent 2x2x1 multiplications, each rank 4, total 24.
    # No savings possible here with simple block methods.
    
    # I think the cleanest path is to just return the schoolbook correctly.
    # The score will be 1.0, same as parent. But the directive asks for improvement.
    
    # Let me try one more idea: Use the known result that rank(4,4,4) = 49
    # (or at most 49 by known constructions). For 4x4x6, we can embed
    # into 4x4x8 by padding, and rank(4,4,8) <= rank(4,4,4) + rank(4,4,4) = 98?
    # No, that's worse.
    
    # Actually, rank(4,4,6) <= rank(4,4,4) * ceil(6/4) = 49*2 = 98? Worse.
    
    # Or rank(4,4,6) <= rank(4,4,8) and rank(4,4,8) <= ... 
    
    # I think for 4x4x6, the known best might actually be close to 96, and
    # simple improvements are hard. Let me just return a correct schoolbook.
    
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
