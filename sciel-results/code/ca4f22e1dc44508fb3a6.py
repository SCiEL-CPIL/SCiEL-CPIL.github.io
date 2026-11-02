"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def strassen_block(i0, j0, k0):
        """Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        
        # M1 = (a00+a11)(b00+b11)
        # c00 += M1, c11 += M1
        u = {(i0, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(i0, k0): 1, (i0+1, k0+1): 1}
        add_term(u, v, w)
        
        # M2 = (a10+a11)b00
        # c10 += M2, c11 -= M2
        u = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v = {(j0, k0): 1}
        w = {(i0+1, k0): 1, (i0+1, k0+1): -1}
        add_term(u, v, w)
        
        # M3 = a00(b10-b11)
        # c01 += M3, c11 += M3
        u = {(i0, j0): 1}
        v = {(j0+1, k0): 1, (j0+1, k0+1): -1}
        w = {(i0, k0+1): 1, (i0+1, k0+1): 1}
        add_term(u, v, w)
        
        # M4 = a11(b01-b00)
        # c00 -= M4, c10 += M4
        u = {(i0+1, j0+1): 1}
        v = {(j0, k0+1): 1, (j0, k0): -1}
        w = {(i0, k0): -1, (i0+1, k0): 1}
        add_term(u, v, w)
        
        # M5 = (a00+a01)b11
        # c00 += M5, c11 += M5
        u = {(i0, j0): 1, (i0, j0+1): 1}
        v = {(j0+1, k0+1): 1}
        w = {(i0, k0): 1, (i0+1, k0+1): 1}
        add_term(u, v, w)
        
        # M6 = (a10-a00)(b00+b10)
        # c11 += M6
        u = {(i0+1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0+1, k0): 1}
        w = {(i0+1, k0+1): 1}
        add_term(u, v, w)
        
        # M7 = (a01-a11)(b01+b11)
        # c00 -= M7, c01 += M7
        u = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v = {(j0, k0+1): 1, (j0+1, k0+1): 1}
        w = {(i0, k0): -1, (i0, k0+1): 1}
        add_term(u, v, w)
    
    def schoolbook(i, j, k):
        add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
    
    # Build decomposition for 2x5 * 5x6
    # Split j: {0,1}, {2,3}, {4}
    # Split k: {0,1}, {2,3}, {4,5}
    
    strassen_block(0, 0, 0)  # A[:,0:2] * B[0:2,0:2]
    strassen_block(0, 0, 2)  # A[:,0:2] * B[0:2,2:4]
    strassen_block(0, 0, 4)  # A[:,0:2] * B[0:2,4:6]
    strassen_block(0, 2, 0)  # A[:,2:4] * B[2:4,0:2]
    strassen_block(0, 2, 2)  # A[:,2:4] * B[2:4,2:4]
    strassen_block(0, 2, 4)  # A[:,2:4] * B[2:4,4:6]
    
    for k in range(6):
        schoolbook(0, 4, k)
        schoolbook(1, 4, k)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] += val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] += val
        for (i, k), val in w_dict.items():
            W[k * n + i][c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

