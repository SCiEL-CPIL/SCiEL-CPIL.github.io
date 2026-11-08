"""
Matrix multiplication 2x3x6 rank-minimal decomposition.
Uses block decomposition: 2x1x6 (schoolbook, rank 12) + 2x2x6 (Strassen per 2x2x2 block, rank 21).
Total rank = 33.
"""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    total_rank = 33
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * total_rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Block 1: 2x1x6 (j=0, k=0..5), schoolbook, rank 12
    for i in range(2):
        for k in range(6):
            U[i * 3 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Block 2: 2x2x6 (j=1,2; k=0..5), Strassen per 2x2x2 block
    # Split k into 3 blocks of size 2: k=0,1; k=2,3; k=4,5
    # For each block, use Strassen on 2x2x2 with i in {0,1}, j in {1,2}, k' in {0,1}
    
    # Strassen formulas for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # In our context: A is 2x2 (i in {0,1}, j in {1,2}), B is 2x2 (j in {1,2}, k' in {0,1})
    # A[i,j] corresponds to U[i*3 + j, c]
    # B[j,k'] corresponds to V[j*6 + k, c] where k = 2*block + k'
    # Output C[i,k] = W[k*2 + i, c]
    
    # Strassen U, V, W coefficients for each of the 7 products:
    # For M1: A00+A11 times B00+B11
    #   U: A00 -> 1, A11 -> 1  (i=0,j=1 and i=1,j=2)
    #   V: B00 -> 1, B11 -> 1  (j=1,k'=0 and j=2,k'=1)
    #   W: C00 -> 1, C01 -> -1, C10 -> -1, C11 -> 1
    # Wait, let me be more careful.
    # 
    # Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # 
    # c00 = M1 - M4 + M5 + M6
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    # 
    # In our indexing:
    # A[i,j]: i in {0,1}, j in {1,2}  -> U[i*3 + j, c]
    #   a00 = A[0,1] = U[0*3+1] = U[1]
    #   a01 = A[0,2] = U[0*3+2] = U[2]
    #   a10 = A[1,1] = U[1*3+1] = U[4]
    #   a11 = A[1,2] = U[1*3+2] = U[5]
    # 
    # B[j,k']: j in {1,2}, k' in {0,1}  -> V[j*6 + k, c] where k = 2*block + k'
    #   b00 = B[1,0] = V[1*6 + k]  (k'=0)
    #   b01 = B[1,1] = V[1*6 + k+1]  (k'=1)
    #   b10 = B[2,0] = V[2*6 + k]  (k'=0)
    #   b11 = B[2,1] = V[2*6 + k+1]  (k'=1)
    # 
    # C[i,k] = W[k*2 + i, c]
    #   c00 = C[0, k] = W[k*2 + 0]
    #   c01 = C[0, k+1] = W[(k+1)*2 + 0]
    #   c10 = C[1, k] = W[k*2 + 1]
    #   c11 = C[1, k+1] = W[(k+1)*2 + 1]
    
    # Strassen coefficients:
    # M1: U: {1:1, 5:1}, V: {b00:1, b11:1}, W: {c00:1, c01:-1, c10:-1, c11:1}
    # M2: U: {4:1, 5:1}, V: {b00:1}, W: {c10:1, c11:-1}
    # M3: U: {1:1}, V: {b01:1, b11:-1}, W: {c01:1, c11:1}
    # M4: U: {5:1}, V: {b10:1, b00:-1}, W: {c00:-1, c10:1}
    # M5: U: {1:1, 2:1}, V: {b11:1}, W: {c00:1, c01:1}
    # M6: U: {4:1, 1:-1}, V: {b00:1, b01:1}, W: {c00:1}
    # M7: U: {2:1, 5:-1}, V: {b10:1, b11:1}, W: {c11:1}
    
    strassen_blocks = [
        # (U_coefs, V_coefs, W_coefs)
        # U_coefs: dict from U_index to coeff
        # V_coefs: dict from (j, kprime) to coeff
        # W_coefs: dict from (i, kprime) to coeff
        # M1
        ({1: 1, 5: 1}, {(1,0): 1, (2,1): 1}, {(0,0): 1, (0,1): -1, (1,0): -1, (1,1): 1}),
        # M2
        ({4: 1, 5: 1}, {(1,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({1: 1}, {(1,1): 1, (2,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({5: 1}, {(2,0): 1, (1,0): -1}, {(0,0): -1, (1,0): 1}),
        # M5
        ({1: 1, 2: 1}, {(2,1): 1}, {(0,0): 1, (0,1): 1}),
        # M6
        ({4: 1, 1: -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # M7
        ({2: 1, 5: -1}, {(2,0): 1, (2,1): 1}, {(1,1): 1}),
    ]
    
    for block in range(3):
        base_k = 2 * block
        for u_coefs, v_coefs, w_coefs in strassen_blocks:
            # U part
            for u_idx, coeff in u_coefs.items():
                if coeff != 0:
                    U[u_idx][col] += coeff
            # V part
            for (j, kp), coeff in v_coefs.items():
                if coeff != 0:
                    v_idx = j * 6 + (base_k + kp)
                    V[v_idx][col] += coeff
            # W part
            for (i, kp), coeff in w_coefs.items():
                if coeff != 0:
                    w_idx = (base_k + kp) * 2 + i
                    W[w_idx][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
