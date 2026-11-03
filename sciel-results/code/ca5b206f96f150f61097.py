"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size is 3
    bs = 3
    # Strassen rank is 7 * (3*3*3) = 189
    rank = 7 * bs * bs * bs
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple (lambda_dict, mu_dict, nu_dict)
    # lambda_dict: {(br, bc): coeff} for A blocks
    # mu_dict: {(bc, bn): coeff} for B blocks
    # nu_dict: {(br, bn): coeff} for C blocks
    
    terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for lam, mu, nu in terms:
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    # Update U
                    for (br, bc), coeff in lam.items():
                        r_A = br * bs + i
                        c_A = bc * bs + j
                        U[r_A * m + c_A][col] += coeff
                    
                    # Update V
                    for (bc, bn), coeff in mu.items():
                        r_B = bc * bs + j
                        c_B = bn * bs + k
                        V[r_B * p + c_B][col] += coeff
                    
                    # Update W
                    for (br, bn), coeff in nu.items():
                        r_C = bn * bs + k
                        c_C = br * bs + i
                        W[r_C * n + c_C][col] += coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
