def add_block_mult(U, V, W, rank, 
                   i_start, j_start, k_start, 
                   i_end, j_end, k_end, 
                   coeff=1):
    # Multiplies A[i_start:i_end, j_start:j_end] by B[j_start:j_end, k_start:k_end]
    # and adds to C[i_start:i_end, k_start:k_end]
    # This is for the case where the block is small enough to use schoolbook
    # Here we assume the block is 2x2 or 2x1
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                U[i*4 + j][rank] += coeff
                V[j*7 + k][rank] += coeff
                W[k*4 + i][rank] += coeff
