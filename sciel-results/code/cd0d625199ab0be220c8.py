    for k_block in range(3):
        for j_block in range(2):
            col1 = add_strassen(1, j_block * 2, k_block * 2, U1, V1, W1, col1)
