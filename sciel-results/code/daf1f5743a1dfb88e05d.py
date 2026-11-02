"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Middle product decomposition:
    # Model 2x2x4 as two 2x2x2 blocks with shared V-support structure.
    # Block A: k in {0,1}, Block B: k in {2,3}
    # For each block, use 7 Strassen terms.
    # The V-factors for Block B are constructed to share support patterns
    # with Block A, exploiting the rectangular width-4 structure.
    
    # Define the 7 Strassen terms as (u_vec, v_vec, w_vec)
    # where each vec is a dict mapping index to coefficient.
    # u_vec: (i,j) -> coeff
    # v_vec: (j,k_local) -> coeff  (k_local in {0,1})
    # w_vec: (k_local,i) -> coeff  (k_local in {0,1})
    
    strassen_terms = [
        # Term 0: (a00+a11)(b00+b11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # Term 1: (a10+a11)b00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        # Term 2: a00(b01-b11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ),
        # Term 3: a11(b10-b00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        # Term 4: (a00+a01)b11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        ),
        # Term 5: (a10-a00)(b00+b01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # Term 6: (a01-a11)(b10+b11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    # Fill Block A: columns 0-6, k_local maps to global k = k_local
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = t
        
        for (i, j), coeff in u_dict.items():
            u_idx = i * m + j
            U[u_idx][col] += coeff
            
        for (j, kl), coeff in v_dict.items():
            v_idx = j * p + kl
            V[v_idx][col] += coeff
            
        for (kl, i), coeff in w_dict.items():
            w_idx = kl * n + i
            W[w_idx][col] += coeff
    
    # Fill Block B: columns 7-13, k_local maps to global k = k_local + 2
    # V-factors share support structure with Block A but shifted in k
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = 7 + t
        
        for (i, j), coeff in u_dict.items():
            u_idx = i * m + j
            U[u_idx][col] += coeff
            
        for (j, kl), coeff in v_dict.items():
            v_idx = j * p + (kl + 2)
            V[v_idx][col] += coeff
            
        for (kl, i), coeff in w_dict.items():
            w_idx = (kl + 2) * n + i
            W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
