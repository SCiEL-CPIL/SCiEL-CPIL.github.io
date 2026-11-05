<<<<<<< SEARCH
    for i in range(2):
        for k in range(8):
            U[2 + i][col] = 1          # A[i, 2]
            V[16 + k][col] = 1         # B[2, k]
            W[k * 2 + i][col] = 1      # C[k, i]
            col += 1
=======
    for i in range(2):
        for k in range(8):
            # A[i, 2] global index is i*m + 2 = 3*i + 2
            U[3 * i + 2][col] = 1      
            V[16 + k][col] = 1         # B[2, k]
            W[k * 2 + i][col] = 1      # C[k, i]
            col += 1
>>>>>>> REPLACE
