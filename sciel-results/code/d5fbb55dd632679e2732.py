<<<<<<< SEARCH
    # Build final matrices
    rank = len(U_cols)
    U = [U_cols[c] for c in range(rank)]
    V = [V_cols[c] for c in range(rank)]
    W = [W_cols[c] for c in range(rank)]
    
    return {"U": U, "V": V, "W": W}
=======
    # Build final matrices
    rank = len(U_cols)
    
    # U_cols is a list of 'rank' vectors, each of length n*m.
    # We need U to be a matrix of shape (n*m, rank).
    # So U[i][c] should be U_cols[c][i].
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for j in range(m * p):
            V[j][c] = V_cols[c][j]
        for k in range(p * n):
            W[k][c] = W_cols[c][k]
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
