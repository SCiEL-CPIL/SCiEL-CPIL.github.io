"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication using
    two independent rank-49 Strassen decompositions for the 4x4*4x4
    subproblems (left and right halves of B).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=8

    # Precompute the 49 rank-1 terms for a single 4x4*4x4 Strassen product.
    inner_terms = [
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]

    top_terms = [
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]

    terms_4x4 = []
    for t in range(7):
        a_blocks, b_blocks, c_blocks = top_terms[t]
        for s in range(7):
            a_inner, b_inner, c_inner = inner_terms[s]

            u_vec = {}
            v_vec = {}
            w_vec = {}

            for (abr, abc), ca in a_blocks.items():
                for (air, aic), ci in a_inner.items():
                    a_row = abr * 2 + air
                    a_col = abc * 2 + aic
                    u_idx = a_row * 4 + a_col
                    u_vec[u_idx] = u_vec.get(u_idx, 0) + ca * ci

            for (bbr, bbc), cb in b_blocks.items():
                for (bir, bic), ci in b_inner.items():
                    b_row = bbr * 2 + bir
                    b_col = bbc * 2 + bic
                    v_idx = b_row * 4 + b_col
                    v_vec[v_idx] = v_vec.get(v_idx, 0) + cb * ci

            for (cbr, cbc), cc in c_blocks.items():
                for (cir, cic), ci in c_inner.items():
                    c_row = cbr * 2 + cir
                    c_col = cbc * 2 + cic
                    w_idx = c_col * 4 + c_row
                    w_vec[w_idx] = w_vec.get(w_idx, 0) + cc * ci

            terms_4x4.append((u_vec, v_vec, w_vec))

    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]

    col = 0
    for half in range(2):
        for (u_vec, v_vec, w_vec) in terms_4x4:
            for u_idx, coeff in u_vec.items():
                U[u_idx][col] += coeff

            for v_idx, coeff in v_vec.items():
                b_row = v_idx // 4
                b_col = v_idx % 4
                actual_v_idx = b_row * 8 + b_col + half * 4
                V[actual_v_idx][col] += coeff

            for w_idx, coeff in w_vec.items():
                c_col = w_idx // 4
                c_row = w_idx % 4
                actual_w_idx = (c_col + half * 4) * 4 + c_row
                W[actual_w_idx][col] += coeff

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
