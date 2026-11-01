import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split 12 columns into two groups of 6
    # For each group of 6 columns, use rank-11 decomposition of 2x2 x 2x6
    # Total rank = 2 * 11 = 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 4 x 22
    V = [[0] * rank for _ in range(m * p)]  # 24 x 22
    W = [[0] * rank for _ in range(p * n)]  # 24 x 22
    
    # For each group of 6 columns, we decompose 2x2 x 2x6
    # Strategy: Use Strassen (rank 7) for first 2 columns + schoolbook (rank 4) for next 2 + schoolbook (rank 4) for last 2? 
    # No, that's 15. We need rank 11 for 2x2x2x6.
    # 
    # Actually, for 2x2 x 2x6:
    # - Strassen on 2x2x2x2 gives rank 7 for first 2 columns
    # - For remaining 4 columns, we can use a rank-4 scheme? No, schoolbook is 4x4=16.
    # 
    # Let me think differently. The rank of 2x2 x 2xp:
    # For p=2: rank 7 (Strassen)
    # For p=3: rank 11 (known)
    # For p=4: rank 14? or 15?
    # For p=6: rank 11? That seems too low.
    #
    # Actually, I recall that for 2x2 x 2xp, the rank is:
    # p=1: 4 (schoolbook)
    # p=2: 7 (Strassen)
    # p=3: 11
    # p=4: 14
    # p=5: 17
    # p=6: 20
    #
    # Hmm, that doesn't match the strategy. Let me reconsider.
    #
    # Wait, the strategy says "known rank-11 decomposition for 2x2x6". Let me verify.
    # Actually, I think the rank might be lower. Let me try a different approach.
    #
    # For 2x2 x 2xp, we can use:
    # - 4 "diagonal" terms: A00*B0k, A01*B1k, A10*B0k, A11*B1k for each k
    # - But Strassen reduces this.
    #
    # Let me just try rank 22 and see if I can construct it.
    # Actually, let me try a simpler approach: use Strassen for pairs of columns.
    # 12 columns = 6 pairs, each pair uses rank 7, total = 42. That's what we have now.
    #
    # To do better, we need to share computation across more than 2 columns.
    #
    # Key insight: For 2x2 x 2xp, we can write:
    # C[:,k] = A @ B[:,k]
    # 
    # If we let S1 = A00 + A11, S2 = A00 - A11, S3 = A10 + A11, S4 = A01 - A11,
    # S5 = A00 + A01, S6 = A10 - A00, S7 = A01 - A11
    # Then for each pair of columns (2t, 2t+1), we use 7 Strassen terms.
    #
    # But can we share some of these across multiple pairs? 
    # The S terms depend only on A, not on which column pair we're processing.
    # So S1, S2, ..., S7 are computed once!
    #
    # Wait, but in the tensor decomposition, each "term" c has specific U, V, W values.
    # If we use the same S1 for multiple column pairs, we'd need separate V and W for each.
    # So we can't directly share.
    #
    # Let me try a different decomposition. Consider:
    # For all k, C0k = A00*B0k + A01*B1k
    # For all k, C1k = A10*B0k + A11*B1k
    #
    # We can write:
    # C0k = (A00+B1k*B0k_inv...) no, that doesn't work.
    #
    # Let me try: introduce auxiliary variables.
    # Let D0 = A00 + A10, D1 = A01 + A11, D2 = A00 - A10, D3 = A01 - A11
    # Then:
    # C0k = A00*B0k + A01*B1k
    # C1k = A10*B0k + A11*B1k
    #
    # C0k + C1k = (A00+A10)*B0k + (A01+A11)*B1k = D0*B0k + D1*B1k
    # C0k - C1k = (A00-A10)*B0k + (A01-A11)*B1k = D2*B0k + D3*B1k
    #
    # So if we compute D0*B0k and D1*B1k for all k, and D2*B0k and D3*B1k for all k,
    # we can recover C0k and C1k.
    #
    # D0*B0k for all k: this is a rank-1 matrix times a row vector, so it's 12 scalar products.
    # But in tensor rank terms, D0*B0k for a specific k is 1 term.
    # For all k, we need 12 terms for D0*B0k, 12 for D1*B1k, 12 for D2*B0k, 12 for D3*B1k.
    # Total = 48, which is schoolbook.
    #
    # Hmm, that's not helping. Let me think about this more carefully.
    #
    # Actually, I think the best we can do is 42 with Strassen pairs, or possibly lower.
    # Let me check if rank 24 is achievable.
    #
    # For 2x2 x 2x12, the border rank is at least 2*12 = 24 (since each output column is a 2x1 vector
    # that's a linear combination of 2 input vectors, and we need to distinguish all 12 columns).
    # Actually, the rank is at least 2p = 24 for this type of problem? No, that's not right either.
    #
    # Let me just verify the current rank-42 solution works and try to find a better one through
    # a different construction.
    #
    # Alternative: use the fact that 2x2 x 2x12 can be seen as computing 24 dot products
    # (2 rows of A with 12 columns of B, but actually it's 2x2 times 2x12).
    #
    # Let me try a rank-24 construction:
    # For each k, use 2 terms:
    # Term 2k: U[A00]=1, U[A10]=1, V[B0k]=1, W[C0k]=1, W[C1k]=1
    #   This gives (A00+A10)*B0k added to both C0k and C1k
    # Term 2k+1: U[A01]=1, U[A11]=1, V[B1k]=1, W[C0k]=1, W[C1k]=1
    #   This gives (A01+A11)*B1k added to both C0k and C1k
    #
    # But this gives C0k = (A00+A10)*B0k + (A01+A11)*B1k and C1k = same, which is wrong.
    #
    # Let me try:
    # Term 2k: U[A00]=1, V[B0k]=1, W[C0k]=1  -> A00*B0k to C0k
    # Term 2k+1: U[A01]=1, V[B1k]=1, W[C0k]=1  -> A01*B1k to C0k
    # Term 2k+2: U[A10]=1, V[B0k]=1, W[C1k]=1  -> A10*B0k to C1k  
    # Term 2k+3: U[A11]=1, V[B1k]=1, W[C1k]=1  -> A11*B1k to C1k
    # That's 4 per column, 48 total. Schoolbook.
    #
    # I think 42 is hard to beat without a clever construction. Let me just verify the
    # current solution and keep it, but try a few alternative constructions to see if
    # any give rank < 42.
    
    # Let me try the Strassen-based construction again, but verify it carefully.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    num_pairs = p // 2  # 6
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00 + A11) * (B0_{2t} + B1_{2t+1})
                U[0][c] = 1  # A00
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B0, col 2t
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = 1  # C0, col 2t
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 1:
                # M2 = (A10 + A11) * B0_{2t}
                U[2][c] = 1  # A10
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B0, col 2t
                W[2*t*n + 1][c] = 1  # C1, col 2t
                W[(2*t+1)*n + 1][c] = -1  # C1, col 2t+1
            elif s == 2:
                # M3 = A00 * (B1_{2t+1} - B0_{2t+1})
                U[0][c] = 1  # A00
                V[0*p + 2*t+1][c] = 1  # B0, col 2t+1
                V[1*p + 2*t+1][c] = -1  # B1, col 2t+1
                W[(2*t+1)*n + 0][c] = 1  # C0, col 2t+1
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 3:
                # M4 = A11 * (B1_{2t} - B0_{2t})
                U[3][c] = 1  # A11
                V[1*p + 2*t][c] = 1  # B1, col 2t
                V[0*p + 2*t][c] = -1  # B0, col 2t
                W[2*t*n + 0][c] = 1  # C0, col 2t
                W[2*t*n + 1][c] = 1  # C1, col 2t
            elif s == 4:
                # M5 = (A00 + A01) * B1_{2t+1}
                U[0][c] = 1  # A00
                U[1][c] = 1  # A01
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = -1  # C0, col 2t
                W[(2*t+1)*n + 0][c] = 1  # C0, col 2t+1
            elif s == 5:
                # M6 = (A10 - A00) * (B0_{2t} + B0_{2t+1})
                U[2][c] = 1  # A10
                U[0][c] = -1  # A00
                V[0*p + 2*t][c] = 1  # B0, col 2t
                V[0*p + 2*t+1][c] = 1  # B0, col 2t+1
                W[(2*t+1)*n + 1][c] = 1  # C1, col 2t+1
            elif s == 6:
                # M7 = (A01 - A11) * (B1_{2t} + B1_{2t+1})
                U[1][c] = 1  # A01
                U[3][c] = -1  # A11
                V[1*p + 2*t][c] = 1  # B1, col 2t
                V[1*p + 2*t+1][c] = 1  # B1, col 2t+1
                W[2*t*n + 0][c] = 1  # C0, col 2t
    
    # Verify the decomposition
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
