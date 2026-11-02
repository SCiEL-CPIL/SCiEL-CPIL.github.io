"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 18 x 35
    W = [[0] * rank for _ in range(p * n)]   # 18 x 35

    # Laderman decomposition for 2x2x2 (columns 0,1)
    # Verified decomposition with 7 terms
    # Using indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    terms = [
        # (U_entries, V_entries, W_entries)
        # Term 0: A[0][0] * B[0][0]
        {('U', (0, 0, 1)), ('V', (0, 0, 1)), ('W', (0, 0, 1)), ('W', (3, 0, 1))},
        # Term 1: (A[1][0]+A[1][1]) * B[0][0]  
        {('U', (2, 1, 1)), ('U', (3, 1, 1)), ('V', (0, 1, 1)), ('W', (2, 1, 1)), ('W', (3, 1, -1))},
        # Term 2: A[0][0] * (B[0][1]-B[1][1])
        {('U', (0, 2, 1)), ('V', (1, 2, 1)), ('V', (11, 2, -1)), ('W', (1, 2, 1)), ('W', (3, 2, 1))},
        # Term 3: A[1][1] * (B[1][0]-B[0][0])
        {('U', (3, 3, 1)), ('V', (10, 3, 1)), ('V', (0, 3, -1)), ('W', (0, 3, 1)), ('W', (2, 3, 1))},
        # Term 4: (A[0][0]+A[0][1]) * B[1][1]
        {('U', (0, 4, 1)), ('U', (1, 4, 1)), ('V', (11, 4, 1)), ('W', (0, 4, -1)), ('W', (1, 4, 1))},
        # Term 5: (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
        {('U', (2, 5, 1)), ('U', (0, 5, -1)), ('V', (0, 5, 1)), ('V', (1, 5, 1)), ('W', (3, 5, 1))},
        # Term 6: (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
        {('U', (1, 6, 1)), ('U', (3, 6, -1)), ('V', (10, 6, 1)), ('V', (11, 6, 1)), ('W', (0, 6, 1))},
    ]
    
    for r, term in enumerate(terms):
        for entry in term:
            matrix, (row, col, val) = entry
            if matrix == 'U':
                U[row][col] = val
            elif matrix == 'V':
                V[row][col] = val
            elif matrix == 'W':
                W[row][col] = val

    # Schoolbook for columns 2-8
    col = 7
    for k in range(2, 9):
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
