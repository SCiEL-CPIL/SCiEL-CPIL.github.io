"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx: list of (index, coeff) for U
    # v_idx: list of (index, coeff) for V
    # w_idx: list of (index, coeff) for W
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for idx, coeff in u_list:
            U[idx][col] += coeff
        for idx, coeff in v_list:
            V[idx][col] += coeff
        for idx, coeff in w_list:
            W[idx][col] += coeff
        col += 1

    # Strassen 2x2x2 template
    # A indices: (i, j) -> i*m + j
    # B indices: (j, k) -> j*p + k
    # C indices: (i, k) -> k*n + i
    
    def strassen_block(i0, j0, k0):
        # i0, j0, k0 are the starting indices for the 2x2x2 block
        # A: i0..i0+1, j0..j0+1
        # B: j0..j0+1, k0..k0+1
        # C: i0..i0+1, k0..k0+1
        
        # 1. (A00+A11)(B00+B11) -> C00, C11
        u = [(i0*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
        v = [(j0*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
        w = [(k0*n + i0, 1), ((k0+1)*n + (i0+1), 1)]
        add_term(u, v, w)
        
        # 2. (A10+A11)B00 -> C10, -C11
        u = [((i0+1)*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
        v = [(j0*p + k0, 1)]
        w = [(k0*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        add_term(u, v, w)
        
        # 3. A00(B10-B11) -> C01, C10
        u = [(i0*m + j0, 1)]
        v = [((j0+1)*p + k0, 1), ((j0+1)*p + (k0+1), -1)]
        w = [(k0+1)*n + i0, 1], [(k0*n + (i0+1), 1)]
        add_term(u, v, w)
        
        # 4. A11(B01-B00) -> -C00, C01
        u = [((i0+1)*m + (j0+1), 1)]
        v = [(j0*p + (k0+1), 1), (j0*p + k0, -1)]
        w = [(k0*n + i0, -1), (k0+1)*n + i0, 1]
        add_term(u, v, w)
        
        # 5. (A00+A01)B11 -> C01, C11
        u = [(i0*m + j0, 1), (i0*m + (j0+1), 1)]
        v = [((j0+1)*p + (k0+1), 1)]
        w = [(k0+1)*n + i0, 1], [(k0+1)*n + (i0+1), 1]
        add_term(u, v, w)
        
        # 6. (A10-A00)(B00+B01) -> C00, C10
        u = [((i0+1)*m + j0, 1), (i0*m + j0, -1)]
        v = [(j0*p + k0, 1), (j0*p + (k0+1), 1)]
        w = [(k0*n + i0, 1), (k0*n + (i0+1), 1)]
        add_term(u, v, w)
        
        # 7. (A01-A11)(B10+B11) -> -C00, C10
        u = [(i0*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v = [((j0+1)*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
        w = [(k0*n + i0, -1), (k0*n + (i0+1), 1)]
        add_term(u, v, w)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    strassen_block(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    strassen_block(0, 0, 2)
    
    # Outer products
    # 3. a1 b_bot: A[0:2, 2] * B[2, 0:4]
    for i in range(2):
        for k in range(4):
            u = [(i*m + 2, 1)]
            v = [(2*p + k, 1)]
            w = [(k*n + i, 1)]
            add_term(u, v, w)
            
    # 4. a2 B_top: A[2, 0:2] * B[0:2, 0:4]
    for j in range(2):
        for k in range(4):
            u = [(2*m + j, 1)]
            v = [(j*p + k, 1)]
            w = [(k*n + 2, 1)]
            add_term(u, v, w)
            
    # 5. a3 b_bot: A[2, 2] * B[2, 0:4]
    for k in range(4):
        u = [(2*m + 2, 1)]
        v = [(2*p + k, 1)]
        w = [(k*n + 2, 1)]
        add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
