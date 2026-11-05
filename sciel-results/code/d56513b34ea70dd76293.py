<<<<<<< SEARCH
    if n == 4 and m == 5 and p == 6:
        # Split p into 3, 3
        U1, V1, W1 = build(4, 5, 3)
        U2, V2, W2 = build(4, 5, 3)
        # Embed
        # U1 is for B[:, :3], U2 is for B[:, 3:]
        # V1 is for B[:, :3], V2 is for B[:, 3:]
        # W1 is for C[:, :3], W2 is for C[:, 3:]
        
        # U shape: 20 x rank
        # V shape: 30 x rank
        # W shape: 24 x rank
        
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank = r1 + r2
        
        U = [[0] * rank for _ in range(20)]
        V = [[0] * rank for _ in range(30)]
        W = [[0] * rank for _ in range(24)]
        
        for i in range(20):
            U[i] = U1[i] + [0]*r2
        for i in range(30):
            # V1 rows 0..14 correspond to B rows 0..4, cols 0..2
            # V2 rows 0..14 correspond to B rows 0..4, cols 3..5
            # We need to map V2 indices to global V indices
            # V index = j * p + k.
            # In sub-problem, p_sub=3.
            # V1 index = j * 3 + k.
            # Global V index for V1: j * 6 + k.
            # Global V index for V2: j * 6 + (k + 3).
            
            # Wait, V1 is for the first sub-problem.
            # The sub-problem is 4x5x3.
            # V1 shape is 5x3 = 15 rows.
            # Global V has 5x6 = 30 rows.
            
            # We need to copy V1 into V[0:15] -> V[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]? No.
            # V index j*p+k.
            # For k in 0..2:
            # V1[j*3+k] -> V[j*6+k]
            
            # For V2:
            # V2[j*3+k] -> V[j*6+k+3]
            
            # It is easier to construct a mapping.
            pass # Handled in generic combine below
=======
>>>>>>> REPLACE
