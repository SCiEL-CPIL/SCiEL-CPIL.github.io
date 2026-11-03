"""Matrix multiplication 2x4 * 4x8 via rank-56 Strassen decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56
    
    # Strassen factors for 2x2 * 2x2 (local flat index i*2+j)
    strassen_U = [
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, -1, 0],  # M7: A12 - A22
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [-1, 0, 1, 0],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1],   # M7: B21 + B22
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11 + C22
        [0, 0, 1, -1],  # M2: C21 - C22
        [0, 1, 0, 1],   # M3: C12 + C22
        [1, 0, 1, 0],   # M4: C11 + C21
        [-1, 1, 0, 0],  # M5: -C11 + C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    for b in range(8):
        a_block = b % 2
        b_block = (b // 2) % 2
        c_block = b // 4
        
        for s in range(7):
            col = b * 7 + s
            u_vec = strassen_U[s]
            v_vec = strassen_V[s]
            w_vec = strassen_W[s]
            
            for i in range(2):
                for j in range(2):
                    # U: A[i, 2*a_block + j], flat = i*4 + 2*a_block + j
                    U[i * 4 + 2 * a_block + j][col] += u_vec[i * 2 + j]
                    
                    # V: B[2*b_block + i, 2*c_block + j], flat = (2*b_block+i)*8 + (2*c_block+j)
                    V[(2 * b_block + i) * 8 + (2 * c_block + j)][col] += v_vec[i * 2 + j]
                    
                    # W: C[2*c_block + j, i], flat = (2*c_block+j)*2 + i
                    W[(2 * c_block + j) * 2 + i][col] += w_vec[i * 2 + j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
