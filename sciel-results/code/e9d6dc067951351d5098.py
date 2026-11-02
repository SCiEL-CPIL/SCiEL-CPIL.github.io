import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen terms for 2x2 multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Vector order: [A00, A01, A10, A11]
    # Vector order: [B00, B01, B10, B11]
    # Vector order: [C00, C01, C10, C11]
    
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (A10+A11)(B00)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # P3 = A00*(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # P4 = A11*(B10+B01)
        ([0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]),
        # P5 = (A00+A01)*(B11)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # P6 = (A10-A00)*(B01+B10)
        ([-1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]),
        # P7 = (A01-A10)*(B00+B10)
        ([0, 1, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, indices [i*m+j] = [i*2+j]
            # U has shape (4, 35)
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10. Block k uses columns 2k, 2k+1.
            # B index: row j, col c -> j*p + c
            # B[0, 2k]   -> V[0*10 + 2k]
            # B[0, 2k+1] -> V[0*10 + 2k+1]
            # B[1, 2k]   -> V[1*10 + 2k]
            # B[1, 2k+1] -> V[1*10 + 2k+1]
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            
            # W: C is 2x10. Convention W[k'*n + i] where k' is col, i is row.
            # C[0, 2k]   -> W[2k * 2 + 0]
            # C[1, 2k]   -> W[2k * 2 + 1]
            # C[0, 2k+1] -> W[(2k+1) * 2 + 0]
            # C[1, 2k+1] -> W[(2k+1) * 2 + 1]
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
