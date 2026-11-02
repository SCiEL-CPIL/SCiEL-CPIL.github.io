import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # rank = 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A coeffs: [a00, a01, a10, a11]
    A_coeffs = [
        [1, 0, 0, 1], # r=1
        [0, 0, 1, 1], # r=2
        [1, 0, 0, 0], # r=3
        [0, 0, 0, 1], # r=4
        [1, 1, 0, 0], # r=5
        [-1, 0, 1, 0],# r=6
        [0, 1, 0, -1] # r=7
    ]
    
    # B coeffs for a 2x2 block: [b00, b01, b10, b11]
    # Note: b00 is B[0, 2k], b01 is B[0, 2k+1], b10 is B[1, 2k], b11 is B[1, 2k+1]
    B_coeffs = [
        [1, 0, 0, 1], # r=1
        [1, 0, 0, 0], # r=2
        [0, 1, 0, -1],# r=3
        [0, 0, 1, -1],# r=4
        [0, 0, 0, 1], # r=5
        [1, 1, 0, 0], # r=6
        [0, 0, 1, 1]  # r=7
    ]
    
    # C coeffs for a 2x2 block: [c00, c01, c10, c11]
    # c00 is C[2k, 0], c01 is C[2k+1, 0], c10 is C[2k, 1], c11 is C[2k+1, 1]
    # Wait, W index is k*n + i.
    # For block k, columns are 2k, 2k+1.
    # C[2k, 0] -> index 2k*2 + 0
    # C[2k+1, 0] -> index (2k+1)*2 + 0
    # C[2k, 1] -> index 2k*2 + 1
    # C[2k+1, 1] -> index (2k+1)*2 + 1
    # The order in my C_coeffs list was [c00, c01, c10, c11] corresponding to matrix entries.
    # Let's map carefully.
    C_coeffs = [
        [1, 0, 0, 1], # r=1 -> c00, c11
        [0, 0, 1, -1],# r=2 -> c10, c11
        [0, 1, 0, 1], # r=3 -> c01, c11
        [1, 0, 1, 0], # r=4 -> c00, c10
        [-1, 1, 0, 0],# r=5 -> c00, c01
        [0, 0, 0, 1], # r=6 -> c11
        [1, 0, 0, 1]  # r=7 -> c00, c11
    ]
    
    num_blocks = p // 2 # 8
    
    for k in range(num_blocks):
        for r in range(7):
            col = k * 7 + r
            
            # Fill U
            # U index: i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = A_coeffs[r][i * m + j]
                    
            # Fill V
            # V index: j*p + col_idx
            # B block k has cols 2k, 2k+1
            # b00 -> B[0, 2k] -> V[0*p + 2k]
            # b01 -> B[0, 2k+1] -> V[0*p + 2k+1]
            # b10 -> B[1, 2k] -> V[1*p + 2k]
            # b11 -> B[1, 2k+1] -> V[1*p + 2k+1]
            V[0 * p + 2 * k][col] = B_coeffs[r][0]
            V[0 * p + 2 * k + 1][col] = B_coeffs[r][1]
            V[1 * p + 2 * k][col] = B_coeffs[r][2]
            V[1 * p + 2 * k + 1][col] = B_coeffs[r][3]
            
            # Fill W
            # W index: col_idx * n + i
            # C block k has cols 2k, 2k+1
            # c00 -> C[2k, 0] -> W[2k*n + 0]
            # c01 -> C[2k+1, 0] -> W[(2k+1)*n + 0]
            # c10 -> C[2k, 1] -> W[2k*n + 1]
            # c11 -> C[2k+1, 1] -> W[(2k+1)*n + 1]
            W[2 * k * n + 0][col] = C_coeffs[r][0]
            W[(2 * k + 1) * n + 0][col] = C_coeffs[r][1]
            W[2 * k * n + 1][col] = C_coeffs[r][2]
            W[(2 * k + 1) * n + 1][col] = C_coeffs[r][3]
            
    return {"U": U, "V": V, "W": W}
