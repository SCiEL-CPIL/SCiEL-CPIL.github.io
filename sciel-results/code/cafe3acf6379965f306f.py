"""Matrix multiplication 6,6,6 - rank 189 via Strassen (7) on 2x2 blocks of 3x3."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    X_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    Y_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for s in range(7):
        for t in range(27):
            col = s * 27 + t
            i = t // 9
            j = (t // 3) % 3
            k = t % 3
            
            for br, bc, coeff in X_coeffs[s]:
                row_u = (br * 3 + i) * 6 + (bc * 3 + j)
                U[row_u][col] += coeff
            
            for br, bc, coeff in Y_coeffs[s]:
                row_v = (br * 3 + j) * 6 + (bc * 3 + k)
                V[row_v][col] += coeff
            
            for br, bc, coeff in W_coeffs[s]:
                row_w = (bc * 3 + k) * 6 + (br * 3 + i)
                W[row_w][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
