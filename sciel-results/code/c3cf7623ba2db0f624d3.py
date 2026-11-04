<<<<<<< SEARCH
    # Block 3: column 8 (schoolbook, rank 16)
    offset3 = 98
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J, offset3+idx] = 1
        V[J*9+8, offset3+idx] = 1
        W[8*4+I, offset3+idx] = 1
=======
    # Block 3: column 8 (rank 4 decomposition)
    offset3 = 98
    for r in range(4):
        for i in range(4):
            for j in range(4):
                if j == r:
                    U[i*4+j, offset3+r] = 1
        for j in range(4):
            if j == r:
                V[j*9+8, offset3+r] = 1
        for i in range(4):
            W[8*4+i, offset3+r] = 1
>>>>>>> REPLACE
