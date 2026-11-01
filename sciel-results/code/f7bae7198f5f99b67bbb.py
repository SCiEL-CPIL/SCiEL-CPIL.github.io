"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 by 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Term definitions (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    # We define 7 terms for the first block and 7 for the second.
    
    def add_term(term_idx, u_data, v_data, w_data):
        for idx, val in u_data:
            U[idx][term_idx] += val
        for idx, val in v_data:
            V[idx][term_idx] += val
        for idx, val in w_data:
            W[idx][term_idx] += val

    # Block 1: B cols 0,1 -> C cols 0,1
    # B indices for block 1: 0,1,2,3 (0:0,0 1:0,1 2:1,0 3:1,1)
    # C indices for block 1: 0,2,1,3 (0:0,0 1:0,1 2:1,0 3:1,1) -> Wait, W index is k*n+i.
    # C(0,0) -> 0*2+0=0
    # C(0,1) -> 1*2+0=2
    # C(1,0) -> 0*2+1=1
    # C(1,1) -> 1*2+1=3
    
    # M1 = (a+d)(e+g)
    add_term(0, 
             [(0,1), (3,1)], 
             [(0,1), (2,1)], 
             [(0,1), (3,-1)]) # C00=1, C11=-1? No, C11 is idx 3.
             
    # M2 = (c+d)(e+g)
    add_term(1, 
             [(2,1), (3,1)], 
             [(0,1), (2,1)], 
             [(0,-1), (1,1)]) # C00=-1, C10=1 (idx 1)
             
    # M3 = a(f-g)
    add_term(2, 
             [(0,1)], 
             [(1,1), (2,-1)], 
             [(2,1), (3,-1)]) # C01=1 (idx 2), C11=-1 (idx 3)
             
    # M4 = d(h-f)
    add_term(3, 
             [(3,1)], 
             [(3,1), (1,-1)], 
             [(0,1), (2,1)]) # C00=1, C01=1 (idx 2)
             
    # M5 = (a+b)(h+g)
    add_term(4, 
             [(0,1), (1,1)], 
             [(3,1), (2,1)], 
             [(3,1), (1,-1)]) # C11=1 (idx 3), C10=-1 (idx 1)
             
    # M6 = (c-a)(e+f)
    add_term(5, 
             [(2,1), (0,-1)], 
             [(0,1), (1,1)], 
             [(1,1), (2,-1)]) # C10=1 (idx 1), C01=-1 (idx 2)
             
    # M7 = (b-d)(g+h)
    add_term(6, 
             [(1,1), (3,-1)], 
             [(2,1), (3,1)], 
             [(3,1), (1,-1)]) # C11=1 (idx 3), C10=-1 (idx 1)

    # Block 2: B cols 2,3 -> C cols 2,3
    # B indices for block 2: 4,5,6,7 (4:0,2 5:0,3 6:1,2 7:1,3)
    # C indices for block 2: 4,6,5,7 (4:0,2 5:0,3 6:1,2 7:1,3) -> W index k*n+i.
    # C(0,2) -> 2*2+0=4
    # C(0,3) -> 3*2+0=6
    # C(1,2) -> 2*2+1=5
    # C(1,3) -> 3*2+1=7

    # N1 = (a+d)(g+k)
    add_term(7, 
             [(0,1), (3,1)], 
             [(4,1), (6,1)], 
             [(4,1), (7,-1)])
             
    # N2 = (c+d)(g+k)
    add_term(8, 
             [(2,1), (3,1)], 
             [(4,1), (6,1)], 
             [(4,-1), (5,1)])
             
    # N3 = a(h-k)
    add_term(9, 
             [(0,1)], 
             [(5,1), (6,-1)], 
             [(6,1), (7,-1)])
             
    # N4 = d(l-h)
    add_term(10, 
             [(3,1)], 
             [(7,1), (5,-1)], 
             [(4,1), (6,1)])
             
    # N5 = (a+b)(l+k)
    add_term(11, 
             [(0,1), (1,1)], 
             [(7,1), (6,1)], 
             [(7,1), (5,-1)])
             
    # N6 = (c-a)(g+h)
    add_term(12, 
             [(2,1), (0,-1)], 
             [(4,1), (5,1)], 
             [(5,1), (6,-1)])
             
    # N7 = (b-d)(k+l)
    add_term(13, 
             [(1,1), (3,-1)], 
             [(6,1), (7,1)], 
             [(7,1), (5,-1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
