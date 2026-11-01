import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit

    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # A vector: [A11, A12, A21, A22]
    # B vector: [B11, B12, B21, B22]
    # C vector: [C11, C21, C12, C22] (row-major within 2x2 block)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, indices [i*m+j] = [i*2+j]
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10, for block k, columns 2k and 2k+1
            # B[j, 2k] -> V[j*p + 2k] = V[j*10 + 2k]
            # B[j, 2k+1] -> V[j*p + 2k+1] = V[j*10 + 2k+1]
            V[0 * p + 2 * k][col] = b_vec[0]      # B[0, 2k]
            V[0 * p + 2 * k + 1][col] = b_vec[1]  # B[0, 2k+1]
            V[1 * p + 2 * k][col] = b_vec[2]      # B[1, 2k]
            V[1 * p + 2 * k + 1][col] = b_vec[3]  # B[1, 2k+1]
            
            # W: C is 2x10, convention W[k'*n + i] where k' is col, i is row
            # C[0, 2k]   -> W[2k * n + 0] = W[4k]
            # C[1, 2k]   -> W[2k * n + 1] = W[4k+1]
            # C[0, 2k+1] -> W[(2k+1)*n + 0] = W[4k+2]
            # C[1, 2k+1] -> W[(2k+1)*n + 1] = W[4k+3]
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
