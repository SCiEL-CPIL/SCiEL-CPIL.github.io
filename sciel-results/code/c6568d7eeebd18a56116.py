<<<<<<< SEARCH
    # Schoolbook for A_right (2x1) @ B_bottom (1x10): 2*10 = 20 terms
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
=======
    # Rank-1 decomposition for A_right (2x1) @ B_bottom (1x10)
    # T[i, 2, k] = 1 for all i in {0,1}, k in {0..9}
    # U[i, r] = 1 for i in {2, 5} (indices for A[0,2] and A[1,2])
    # V[20+k, r] = 1 for all k (indices for B[2, k])
    # W[2k+i, r] = 1 for all k, i (indices for C[k, i])
    
    U[2][r] = 1  # A[0, 2]
    U[5][r] = 1  # A[1, 2]
    
    for k in range(10):
        V[20 + k][r] = 1  # B[2, k]
        
    for k in range(10):
        for i in range(2):
            W[k * 2 + i][r] = 1  # C[k, i]
            
    r += 1
>>>>>>> REPLACE
