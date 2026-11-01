"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen decomposition for 2x2x2
    # A: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    # B: 0:(1,k0), 1:(1,k0+1), 2:(2,k0), 3:(2,k0+1)
    # C: 0:(1,k0), 1:(1,k0+1), 2:(2,k0), 3:(2,k0+1)
    
    col = 0
    for block_idx in range(5):
        k0 = block_idx * 2
        
        # B indices
        b_x = 0 * p + k0      # (1, k0)
        b_y = 0 * p + (k0+1)  # (1, k0+1)
        b_z = 1 * p + k0      # (2, k0)
        b_w = 1 * p + (k0+1)  # (2, k0+1)
        
        # C indices
        c_11 = k0 * n + 0     # (k0, 1) -> row k0, col 1? 
        # Wait, W shape is (p*n, rank). Index is k*n + i.
        # C[i,k] corresponds to W[k*n + i].
        # So C[0, k0] is W[k0*n + 0].
        # C[0, k0+1] is W[(k0+1)*n + 0].
        # C[1, k0] is W[k0*n + 1].
        # C[1, k0+1] is W[(k0+1)*n + 1].
        
        c_11 = k0 * n + 0
        c_12 = (k0+1) * n + 0
        c_21 = k0 * n + 1
        c_22 = (k0+1) * n + 1
        
        # M1: (a+d)(x+w)
        add_term(0, 1, b_x, 1, c_11, 1, col)
        add_term(3, 1, b_x, 1, c_11, 1, col)
        add_term(0, 1, b_y, 1, c_11, 1, col)
        add_term(3, 1, b_y, 1, c_11, 1, col)
        
        # M1 also contributes to C22
        add_term(0, 1, b_x, 1, c_22, 1, col)
        add_term(3, 1, b_x, 1, c_22, 1, col)
        add_term(0, 1, b_y, 1, c_22, 1, col)
        add_term(3, 1, b_y, 1, c_22, 1, col)
        
        col += 1
        
        # M2: (c+d)x
        add_term(2, 1, b_x, 1, c_21, 1, col)
        add_term(3, 1, b_x, 1, c_21, 1, col)
        
        # M2 also contributes to C22 with -1
        add_term(2, -1, b_x, 1, c_22, 1, col)
        add_term(3, -1, b_x, 1, c_22, 1, col)
        
        col += 1
        
        # M3: a(y-w)
        add_term(0, 1, b_y, 1, c_12, 1, col)
        add_term(0, -1, b_w, 1, c_12, 1, col)
        
        # M3 also contributes to C22 with 1
        add_term(0, 1, b_y, 1, c_22, 1, col)
        add_term(0, -1, b_w, 1, c_22, 1, col)
        
        col += 1
        
        # M4: d(z-x)
        add_term(3, 1, b_z, 1, c_11, 1, col)
        add_term(3, -1, b_x, 1, c_11, 1, col)
        
        # M4 also contributes to C21 with 1
        add_term(3, 1, b_z, 1, c_21, 1, col)
        add_term(3, -1, b_x, 1, c_21, 1, col)
        
        col += 1
        
        # M5: (a+b)w
        add_term(0, -1, b_w, 1, c_11, 1, col)
        add_term(1, -1, b_w, 1, c_11, 1, col)
        
        # M5 also contributes to C12 with 1
        add_term(0, 1, b_w, 1, c_12, 1, col)
        add_term(1, 1, b_w, 1, c_12, 1, col)
        
        col += 1
        
        # M6: (c-a)(x+z)
        add_term(2, 1, b_x, 1, c_22, 1, col)
        add_term(2, 1, b_z, 1, c_22, 1, col)
        add_term(0, -1, b_x, 1, c_22, 1, col)
        add_term(0, -1, b_z, 1, c_22, 1, col)
        
        col += 1
        
        # M7: (b-d)(y+w)
        add_term(1, 1, b_y, 1, c_11, 1, col)
        add_term(1, 1, b_w, 1, c_11, 1, col)
        add_term(3, -1, b_y, 1, c_11, 1, col)
        add_term(3, -1, b_w, 1, c_11, 1, col)
        
        col += 1

    # Last block k=10 (0-indexed 10, so k=11 in 1-based)
    k0 = 10
    b_x = 0 * p + k0
    b_y = 1 * p + k0
    
    c_11 = k0 * n + 0
    c_21 = k0 * n + 1
    
    # M1: a x
    add_term(0, 1, b_x, 1, c_11, 1, col)
    col += 1
    
    # M2: b y
    add_term(1, 1, b_y, 1, c_11, 1, col)
    col += 1
    
    # M3: c x
    add_term(2, 1, b_x, 1, c_21, 1, col)
    col += 1
    
    # M4: d y
    add_term(3, 1, b_y, 1, c_21, 1, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
