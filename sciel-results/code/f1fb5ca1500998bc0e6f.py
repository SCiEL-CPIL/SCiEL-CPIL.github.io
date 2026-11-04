"""Rewritten program for 6x6 matrix multiplication with rank 189 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 block matrices (each block is 3x3)
    # Term t: alpha[t][(I,J)] = coeff, beta[t][(Jp,K)] = coeff, gamma[t][(I,K)] = coeff

    alphas = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    betas = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    gammas = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    for t in range(7):
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    c = t * 27 + ip * 9 + jp * 3 + kp

                    for (I, J), val in alphas[t].items():
                        i_global = I * 3 + ip
                        j_global = J * 3 + jp
                        U[i_global * 6 + j_global][c] = val

                    for (Jp, K), val in betas[t].items():
                        j_global = Jp * 3 + jp
                        k_global = K * 3 + kp
                        V[j_global * 6 + k_global][c] = val

                    for (I, K), val in gammas[t].items():
                        k_global = K * 3 + kp
                        i_global = I * 3 + ip
                        W[k_global * 6 + i_global][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
