    # Block 3: k={4}, rank 1
    for i in range(2):
        for j in range(3):
            U[i * 3 + j][col] = 1
    for j in range(3):
        V[j * 5 + 4][col] = 1
    for i in range(2):
        W[4 * 2 + i][col] = 1
    col += 1
