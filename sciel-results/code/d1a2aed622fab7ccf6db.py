"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # Each term is a list of (index_u, index_v, coeff) for U
    # and (index_v, index_w, coeff) for V
    # and (index_w, index_u, coeff) for W
    # Indices are local 0,1
    
    terms = [
        # M1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # M3
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # M4
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # M5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # M6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # M7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    col = 0
    for b_i in range(2):
        for b_j in range(2):
            for b_k in range(3):
                for t in terms:
                    # Update U
                    for u, v, coeff in t['U']:
                        i = 2 * b_i + u
                        j = 2 * b_j + v
                        U[i * m + j][col] += coeff
                    # Update V
                    for v, w, coeff in t['V']:
                        j = 2 * b_j + v
                        k = 2 * b_k + w
                        V[j * p + k][col] += coeff
                    # Update W
                    for w, u, coeff in t['W']:
                        k = 2 * b_k + w
                        i = 2 * b_i + u
                        W[k * n + i][col] += coeff
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
