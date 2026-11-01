"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)] # 4 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 16 x 28
    
    # Strassen terms for 2x2 multiplication
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # P1
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 1]
    w0 = [1, -1, 1, 1]
    
    # P2
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [0, 0, 1, -1]
    
    # P3
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [0, 1, 0, 1]
    
    # P4
    u3 = [0, 0, 0, 1]
    v3 = [0, -1, 1, 0]
    w3 = [1, -1, 1, 0]
    
    # P5
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 1]
    w4 = [0, 1, -1, 1]
    
    # P6
    u5 = [0, 1, 0, -1]
    v5 = [0, 0, 1, 1]
    w5 = [-1, 1, 0, 0]
    
    # P7
    u6 = [1, 0, -1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [1, 0, 1, 0]
    
    terms = [
        (u0, v0, w0),
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6)
    ]
    
    for q in range(4):
        base_col = q * 7
        for t in range(7):
            col = base_col + t
            u_vec, v_vec, w_vec = terms[t]
            
            # Fill U
            # U indices: i*m + j. Same for all blocks.
            for idx, val in enumerate(u_vec):
                U[idx][col] = val
            
            # Fill V
            # V indices: j*p + k
            # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global B indices for block q:
            # 0 -> j=0, k=2q -> 0*8 + 2q = 2q
            # 1 -> j=0, k=2q+1 -> 0*8 + 2q+1 = 2q+1
            # 2 -> j=1, k=2q -> 1*8 + 2q = 8+2q
            # 3 -> j=1, k=2q+1 -> 1*8 + 2q+1 = 9+2q
            
            v_indices = [2*q, 2*q+1, 8+2*q, 9+2*q]
            for idx, val in enumerate(v_vec):
                V[v_indices[idx]][col] = val
            
            # Fill W
            # W indices: k*n + i
            # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global C indices for block q:
            # Local 0 (i=0, k=0 rel) -> k=2q, i=0 -> 2q*2 + 0 = 4q
            # Local 1 (i=0, k=1 rel) -> k=2q+1, i=0 -> (2q+1)*2 + 0 = 4q+2
            # Local 2 (i=1, k=0 rel) -> k=2q, i=1 -> 2q*2 + 1 = 4q+1
            # Local 3 (i=1, k=1 rel) -> k=2q+1, i=1 -> (2q+1)*2 + 1 = 4q+3
            
            w_indices = [4*q, 4*q+2, 4*q+1, 4*q+3]
            for idx, val in enumerate(w_vec):
                W[w_indices[idx]][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
