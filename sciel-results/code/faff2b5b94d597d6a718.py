"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_vals, v_vals, w_vals):
        for (i, j), val in u_vals.items():
            U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            V[j * p + k][col] += val
        for (k, i), val in w_vals.items():
            W[k * n + i][col] += val

    # Block 1: k in {0, 1}
    add_term(0, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    add_term(1, {(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
    add_term(2, {(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    add_term(3, {(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
    add_term(4, {(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1})
    add_term(5, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    add_term(6, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    
    # Block 2: k in {2, 3}
    add_term(7, {(0,0): 1, (1,1): 1}, {(0,2): 1, (1,3): 1}, {(2,0): 1, (3,1): 1})
    add_term(8, {(1,0): 1, (1,1): 1}, {(0,2): 1}, {(2,1): 1, (3,1): -1})
    add_term(9, {(0,0): 1}, {(0,3): 1, (1,3): -1}, {(3,0): 1, (3,1): 1})
    add_term(10, {(1,1): 1}, {(1,2): 1, (0,2): -1}, {(2,0): 1, (2,1): 1})
    add_term(11, {(0,0): 1, (0,1): 1}, {(1,3): 1}, {(2,0): -1, (3,0): 1})
    add_term(12, {(1,0): 1, (0,0): -1}, {(0,2): 1, (0,3): 1}, {(3,1): 1})
    add_term(13, {(0,1): 1, (1,1): -1}, {(1,2): 1, (1,3): 1}, {(2,0): 1, (3,1): 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
