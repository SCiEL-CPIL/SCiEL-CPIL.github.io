import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1.0, val_v=1.0, val_w=1.0):
        nonlocal col
        if u_idx is not None:
            U[u_idx[0] * m + u_idx[1]][col] += val_u
        if v_idx is not None:
            V[v_idx[0] * p + v_idx[1]][col] += val_v
        if w_idx is not None:
            W[w_idx[0] * n + w_idx[1]][col] += val_w
        col += 1

    # m blocks: 0, 1
    # p blocks: 0, 1, 2
    for b in range(2):
        j1 = 2 * b
        j2 = 2 * b + 1
        for a in range(3):
            k1 = 2 * a
            k2 = 2 * a + 1
            
            # Part 1: i=0 (1x2x2, rank 4)
            # Terms:
            # 1. M[0, j1] * K[j1, k1] -> N[0, k1]
            add_term((0, j1), (j1, k1), (k1, 0))
            # 2. M[0, j2] * K[j2, k1] -> N[0, k1]
            add_term((0, j2), (j2, k1), (k1, 0))
            # 3. M[0, j1] * K[j1, k2] -> N[0, k2]
            add_term((0, j1), (j1, k2), (k2, 0))
            # 4. M[0, j2] * K[j2, k2] -> N[0, k2]
            add_term((0, j2), (j2, k2), (k2, 0))
            
            # Part 2: i=1,2 (2x2x2, rank 7, Strassen)
            # M = [[M[1, j1], M[1, j2]], [M[2, j1], M[2, j2]]]
            # K = [[K[j1, k1], K[j1, k2]], [K[j2, k1], K[j2, k2]]]
            # N = [[N[1, k1], N[1, k2]], [N[2, k1], N[2, k2]]]
            
            # z1 = M[1, j1] * (K[j1, k1] + K[j2, k2])
            # Contributes to N[1, k1] and N[2, k2]
            add_term((1, j1), (j1, k1), (k1, 1))
            add_term((1, j1), (j2, k2), (k2, 2))
            # Wait, U, V, W are vectors for the term.
            # Term r: sum U[i,j] V[j,k] W[k,i]
            # For z1:
            # U[1, j1] = 1
            # V[j1, k1] = 1, V[j2, k2] = 1
            # W[k1, 1] = 1, W[k2, 2] = 1
            # This creates 4 non-zero entries in U, V, W?
            # No, the term is one column in U, V, W.
            # U[1, j1] = 1
            # V[j1, k1] = 1, V[j2, k2] = 1
            # W[k1, 1] = 1, W[k2, 2] = 1
            # This is ONE term in the sum.
            # But my add_term function assumes single indices.
            # I need to allow multiple indices for V and W.
            
            # Let's redefine add_term to handle lists of indices.
            pass

    # Redefine add_term to handle sparse updates
    # I will just manually add to the arrays.
    
    # Reset
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    def add_sparse_term(u_indices, v_indices, w_indices):
        nonlocal col
        for i, j in u_indices:
            U[i * m + j][col] += 1.0
        for j, k in v_indices:
            V[j * p + k][col] += 1.0
        for k, i in w_indices:
            W[k * n + i][col] += 1.0
        col += 1

    for b in range(2):
        j1 = 2 * b
        j2 = 2 * b + 1
        for a in range(3):
            k1 = 2 * a
            k2 = 2 * a + 1
            
            # Part 1: i=0
            add_sparse_term([(0, j1)], [(j1, k1)], [(k1, 0)])
            add_sparse_term([(0, j2)], [(j2, k1)], [(k1, 0)])
            add_sparse_term([(0, j1)], [(j1, k2)], [(k2, 0)])
            add_sparse_term([(0, j2)], [(j2, k2)], [(k2, 0)])
            
            # Part 2: i=1,2 (Strassen)
            # z1 = M[1, j1] * (K[j1, k1] + K[j2, k2])
            # N[1, k1] += z1, N[2, k2] += z1
            add_sparse_term([(1, j1)], [(j1, k1), (j2, k2)], [(k1, 1), (k2, 2)])
            
            # z2 = (M[2, j1] + M[2, j2]) * (K[j1, k1] + K[j1, k2])
            # N[2, k1] += z2, N[2, k2] += z2
            add_sparse_term([(2, j1), (2, j2)], [(j1, k1), (j1, k2)], [(k1, 2), (k2, 2)])
            
            # z3 = (M[1, j1] + M[1, j2]) * (K[j2, k1] + K[j2, k2])
            # N[1, k1] += z3, N[1, k2] += z3
            add_sparse_term([(1, j1), (1, j2)], [(j2, k1), (j2, k2)], [(k1, 1), (k2, 1)])
            
            # z4 = M[2, j2] * (K[j1, k2] - K[j2, k2])
            # N[1, k1] += z4, N[2, k2] -= z4
            add_sparse_term([(2, j2)], [(j1, k2), (j2, k2)], [(k1, 1), (k2, 2)])
            # Need to subtract for W[2, k2]
            W[k2 * n + 2][col-1] -= 1.0
            V[j2 * p + k2][col-1] -= 1.0
            
            # z5 = M[1, j1] * (K[j1, k2] - K[j2, k1])
            # N[1, k1] -= z5, N[1, k2] += z5
            add_sparse_term([(1, j1)], [(j1, k2), (j2, k1)], [(k1, 1), (k2, 1)])
            W[k1 * n + 1][col-1] -= 1.0
            V[j2 * p + k1][col-1] -= 1.0
            
            # z6 = (M[2, j1] - M[1, j1]) * (K[j1, k1] + K[j2, k2])
            # N[2, k2] += z6
            add_sparse_term([(2, j1), (1, j1)], [(j1, k1), (j2, k2)], [(k2, 2)])
            U[1 * m + j1][col-1] -= 1.0
            
            # z7 = (M[1, j2] - M[2, j2]) * (K[j2, k1] + K[j2, k2])
            # N[1, k1] += z7
            add_sparse_term([(1, j2), (2, j2)], [(j2, k1), (j2, k2)], [(k1, 1)])
            U[2 * m + j2][col-1] -= 1.0

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
