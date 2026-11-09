"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (6, R)
    # V shape: (18, R)
    # W shape: (12, R)
    
    terms = []
    
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals.items():
            vec[idx] = val
        return vec

    # Part 1: A_12 * B_2 (Outer product, rank 12)
    for i in range(2):
        for k in range(6):
            u_idx = i * 3 + 2
            v_idx = 2 * 6 + k
            w_idx = k * 2 + i
            
            u_vec = make_vec(6, {u_idx: 1})
            v_vec = make_vec(18, {v_idx: 1})
            w_vec = make_vec(12, {w_idx: 1})
            terms.append((u_vec, v_vec, w_vec))
            
    # Part 2: A_11 * B_1 (Strassen, rank 21)
    for j in range(3):
        b00_idx = 2*j
        b01_idx = 2*j + 1
        b10_idx = 6 + 2*j
        b11_idx = 6 + 2*j + 1
        
        c00_idx = 4*j
        c01_idx = 4*j + 2
        c10_idx = 4*j + 1
        c11_idx = 4*j + 3
        
        terms.append((make_vec(6, {0: 1, 4: 1}), make_vec(18, {b00_idx: 1, b11_idx: 1}), make_vec(12, {c00_idx: 1, c11_idx: 1})))
        terms.append((make_vec(6, {3: 1, 4: 1}), make_vec(18, {b00_idx: 1}), make_vec(12, {c10_idx: 1, c11_idx: -1})))
        terms.append((make_vec(6, {0: 1}), make_vec(18, {b01_idx: 1, b11_idx: -1}), make_vec(12, {c01_idx: 1, c11_idx: 1})))
        terms.append((make_vec(6, {4: 1}), make_vec(18, {b10_idx: 1, b00_idx: -1}), make_vec(12, {c00_idx: 1, c10_idx: 1})))
        terms.append((make_vec(6, {0: 1, 1: 1}), make_vec(18, {b11_idx: 1}), make_vec(12, {c00_idx: -1, c01_idx: 1})))
        terms.append((make_vec(6, {3: 1, 0: -1}), make_vec(18, {b00_idx: 1, b01_idx: 1}), make_vec(12, {c11_idx: 1})))
        terms.append((make_vec(6, {1: 1, 4: -1}), make_vec(18, {b10_idx: 1, b11_idx: 1}), make_vec(12, {c00_idx: 1})))
        
    rank = len(terms)
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(12)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(6):
            U[i][c] = u[i]
        for i in range(18):
            V[i][c] = v[i]
        for i in range(12):
            W[i][c] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
