"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 35 based on 5 blocks of Laderman's rank-7 2x2x2 multiplication
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's algorithm coefficients for 2x2x2
    # U forms (A):
    # t=0: a+b
    # t=1: c+d
    # t=2: a-d
    # t=3: b+c
    # t=4: a-c
    # t=5: b+d
    # t=6: a+d
    # Indices in U (size 4): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 1, 0, 0], # a+b
        [0, 0, 1, 1], # c+d
        [1, 0, 0, -1], # a-d
        [0, 1, 1, 0], # b+c
        [1, 0, -1, 0], # a-c
        [0, 1, 0, 1], # b+d
        [1, 0, 0, 1]  # a+d
    ]
    
    # V forms (B):
    # t=0: e+h
    # t=1: e+h
    # t=2: e-g
    # t=3: g+h
    # t=4: e+g
    # t=5: f+h
    # t=6: f-h
    # Indices in V (size 20 for p=10, m=2):
    # B is 2x10. Block b uses cols 2b, 2b+1.
    # B[0, 2b] is index 2b.
    # B[0, 2b+1] is index 2b+1.
    # B[1, 2b] is index 10+2b.
    # B[1, 2b+1] is index 10+2b+1.
    
    # W forms (C):
    # C00 = P1 - P2 - P3 + P5
    # C01 = P3 + P6 - P7
    # C10 = P2 + P4 - P5
    # C11 = P1 - P4 + P7
    # Indices in W (size 20 for p=10, n=2):
    # W index is k*n + i.
    # Block b cols 2b, 2b+1.
    # C[0, 2b] is index 4b.
    # C[1, 2b] is index 4b+1.
    # C[0, 2b+1] is index 4b+2.
    # C[1, 2b+1] is index 4b+3.
    
    W_coeffs = [
        [1, 0, 0, 1],  # t=0: P1 -> C00:1, C10:0, C01:0, C11:1
        [-1, 1, 0, 0], # t=1: P2 -> C00:-1, C10:1, C01:0, C11:0
        [-1, 0, 1, 0], # t=2: P3 -> C00:-1, C10:0, C01:1, C11:0
        [0, 1, 0, -1], # t=3: P4 -> C00:0, C10:1, C01:0, C11:-1
        [1, -1, 0, 0], # t=4: P5 -> C00:1, C10:-1, C01:0, C11:0
        [0, 0, 1, 0],  # t=5: P6 -> C00:0, C10:0, C01:1, C11:0
        [0, 0, -1, 1]  # t=6: P7 -> C00:0, C10:0, C01:-1, C11:1
    ]
    
    # V coeffs for each t, given block b
    # Returns list of (index_offset, coeff) for the 4 relevant B entries
    # e=B[0, 2b], f=B[0, 2b+1], g=B[1, 2b], h=B[1, 2b+1]
    # Indices: e->2b, f->2b+1, g->10+2b, h->10+2b+1
    V_coeffs = [
        [(0, 1), (1, 1)],       # t=0: e+h -> idx 0, 1 relative? No, need absolute mapping
        [(0, 1), (1, 1)],       # t=1: e+h
        [(0, 1), (2, -1)],      # t=2: e-g -> e(0), g(2)
        [(2, 1), (3, 1)],       # t=3: g+h -> g(2), h(3)
        [(0, 1), (2, 1)],       # t=4: e+g -> e(0), g(2)
        [(1, 1), (3, 1)],       # t=5: f+h -> f(1), h(3)
        [(1, 1), (3, -1)]       # t=6: f-h -> f(1), h(3)
    ]
    # Mapping of relative index 0,1,2,3 to actual V index for block b
    # 0 -> 2b
    # 1 -> 2b+1
    # 2 -> 10+2b
    # 3 -> 10+2b+1
    
    for b in range(5):
        for t in range(7):
            col = b * 7 + t
            # Fill U
            for i in range(4):
                U[i][col] = U_forms[t][i]
            
            # Fill V
            # V_coeffs[t] contains (rel_idx, coeff)
            for rel_idx, coeff in V_coeffs[t]:
                if rel_idx == 0:
                    v_idx = 2 * b
                elif rel_idx == 1:
                    v_idx = 2 * b + 1
                elif rel_idx == 2:
                    v_idx = 10 + 2 * b
                elif rel_idx == 3:
                    v_idx = 10 + 2 * b + 1
                else:
                    continue
                V[v_idx][col] += coeff
            
            # Fill W
            # W_coeffs[t] is [C00, C10, C01, C11]
            # W indices:
            # C00 -> 4b
            # C10 -> 4b+1
            # C01 -> 4b+2
            # C11 -> 4b+3
            W[4 * b][col] = W_coeffs[t][0]
            W[4 * b + 1][col] = W_coeffs[t][1]
            W[4 * b + 2][col] = W_coeffs[t][2]
            W[4 * b + 3][col] = W_coeffs[t][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
