import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 decomposition vectors
    # u, v, w are lists of 7 lists of 4 floats/ints
    u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for l in range(7):
        for m_idx in range(7):
            col = l * 7 + m_idx
            ul = u[l]
            vm = v[m_idx]
            wl = w[l]
            wm = w[m_idx]
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = I * 2 + i_prime
                            j = J * 2 + j_prime
                            U[i * m + j][col] = ul[i_prime] * ul[j_prime]
                            
            # Fill V
            for J in range(2):
                for K in range(2):
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = J * 2 + j_prime
                            k = K * 2 + k_prime
                            V[j * p + k][col] = vm[j_prime] * vm[k_prime]
                            
            # Fill W
            for K in range(2):
                for I in range(2):
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = K * 2 + k_prime
                            i = I * 2 + i_prime
                            W[k * n + i][col] = wl[k_prime] * wm[i_prime]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
