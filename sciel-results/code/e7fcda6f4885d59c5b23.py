"""Matrix multiplication 3,4,4 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build a decomposition of rank = 3 * (m + p - 1) = 3 * 7 = 21
    # Each block handles one row i of A, using the 1xm x pxp structure
    # For 1xm x pxp, rank is m+p-1 (classical result)
    
    # The 1xm x pxp tensor: T[j, (j',k'), k''] = 1 iff j==j' and k==k''
    # Rank m+p-1 construction using "prefix" sums
    
    sub_rank = m + p - 1  # 7
    total_rank = n * sub_rank  # 21
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n):
        for r in range(sub_rank):
            col = i * sub_rank + r
            
            # For the 1xm x pxp block at row i, use construction:
            # Term r: 
            #   U part: e_j for j where j <= min(r, m-1) ... 
            # Let's use a valid explicit construction.
            
            # Construction for 1xm x pxp, term r (0-indexed, r=0..m+p-2):
            # If r < m: 
            #   U_r[j] = 1 if j == r else 0
            #   V_r[j',k'] = 1 if j' == r and k' <= r else 0  -- NO this isn't right
            
            # Let me use a different verified construction:
            # For 1xm x pxp, use terms corresponding to pairs (a,b) where a in [0,m-1], b in [0,p-1], a+b < m+p-1
            # Actually, simplest valid construction:
            
            # Term r: split into "diagonal" part
            # a = min(r, m-1), b = r - a if r >= m else 0... 
            
            # I'll use the following known-correct construction:
            # For r in 0..m+p-2:
            #   a = max(0, min(r, m-1))
            #   b = max(0, min(p-1, r - (m-1)))
            #   This doesn't quite work either.
            
            # Let me just use: for each r, define
            #   U_r[j] = 1 if j <= r and r < m, else (1 if j == m-1 and r >= m else 0)
            #   This is getting too complex without verification.
            
            # SAFE FALLBACK: use schoolbook within each 1xm x pxp block but that gives m*p = 16 per block, total 48.
            # Instead, let me use a verified small-rank construction.
            
            # VERIFIED construction for 1xm x pxp with rank m+p-1:
            # For r = 0 to m+p-2:
            #   Let a = r if r < m else m - 1
            #   Let b = max(0, r - (m - 1))
            #   U_r[j] = 1 if j == a else 0
            #   V_r[j',k'] = 1 if j' == a and k' == b else 0  -- this gives rank 1 terms that only cover diagonal
            #   That's not right.
            
            pass
    
    # Since the explicit construction is error-prone, let me use a numerically-verified approach:
    # Use the fact that 1xm x pxp can be embedded, and use a rank-(m+p-1) construction
    # that I can verify in code.
    
    # Actually, let me use a simpler but valid improvement:
    # Use the standard result that rank(3x4x4) <= 3 * rank(1x4x4)
    # And use a computationally-verified rank-7 construction for 1x4x4.
    
    # Here's a valid rank-7 construction for 1x4x4 (verified by direct expansion):
    # Terms indexed by r=0..6:
    # r=0: U=[1,0,0,0], V has V[0,0]=1, W=[1,0,0,0]
    # r=1: U=[1,1,0,0], V has V[0,1]=1,V[1,0]=1, W=[0,1,0,0]  -- NO
    # 
    # I give up on the explicit formula. Let me just return the schoolbook (rank 48) to be safe,
    # but that won't improve the score.
    
    # Better: use a block-diagonal structure where I share V factors across rows.
    # If I use the same 4x4 matmul decomposition for all 3 rows, I can potentially share.
    
    # Safest valid improvement: none that I can guarantee without extensive verification.
    # Return schoolbook to be correct.
    
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
