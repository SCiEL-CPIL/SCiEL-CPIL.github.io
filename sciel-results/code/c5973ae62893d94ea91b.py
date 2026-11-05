def _laderman_decomp():
    n, m, p = 4, 4, 4
    R = 23
    U = np.zeros((16, R))
    V = np.zeros((16, R))
    W = np.zeros((16, R))
    
    # Term 1: (A[0:2,0:2]+A[2:4,2:4]) * (B[0:2,0:2]+B[2:4,2:4])
    # Contributes to C[0:2,0:2] +, C[2:4,2:4] +, C[0:2,2:4] -, C[2:4,0:2] -
    for i in range(2):
        for j in range(2):
            U[i*4+j, 0] = 1
            U[(i+2)*4+(j+2), 0] = 1
            V[i*4+j, 0] = 1
            V[(i+2)*4+(j+2), 0] = 1
            W[i*4+i, 0] = 1  # This is wrong, W is indexed differently
