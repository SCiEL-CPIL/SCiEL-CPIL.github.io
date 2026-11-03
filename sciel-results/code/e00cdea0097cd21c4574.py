"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We split the middle dimension (m=4) into 2 groups of 2
    # We split the output dimension (p=6) into 3 groups of 2
    # For each of the 2*3=6 blocks, use Strassen (rank 7)
    # Total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    col = 0
    for g in range(2):  # j-group: 0,1
        for h in range(3):  # k-group: 0,1,2
            # A_g is 2x2: rows i=0,1, cols j=2g, 2g+1
            # B_{g,h} is 2x2: rows j=2g, 2g+1, cols k=2h, 2h+1
            
            # Strassen's 7 products for P*Q where P=A_g, Q=B_{g,h}
            # P = [[p00, p01], [p10, p11]]
            # Q = [[q00, q01], [q10, q11]]
            
            # m1 = (p00+p11)(q00+q11)
            # U entries: for p00 (i=0,j=2g): +1; for p11 (i=1,j=2g+1): +1
            # V entries: for q00 (j=2g,k=2h): +1; for q11 (j=2g+1,k=2h+1): +1
            # W entries: contributes to C00 (+1), C11 (+1)
            U[0 * m + (2*g)][col] = 1    # p00: i=0, j=2g
            U[1 * m + (2*g+1)][col] = 1  # p11: i=1, j=2g+1
            V[(2*g) * p + (2*h)][col] = 1      # q00: j=2g, k=2h
            V[(2*g+1) * p + (2*h+1)][col] = 1  # q11: j=2g+1, k=2h+1
            # C00: i=0, k=2h -> W index = k*n+i = 2h*2+0 = 4h
            # C11: i=1, k=2h+1 -> W index = k*n+i = (2h+1)*2+1 = 4h+3
            W[(2*h) * n + 0][col] = 1       # C00
            W[(2*h+1) * n + 1][col] = 1     # C11
            col += 1
            
            # m2 = (p10+p11)q00
            U[1 * m + (2*g)][col] = 1    # p10: i=1, j=2g
            U[1 * m + (2*g+1)][col] = 1  # p11: i=1, j=2g+1
            V[(2*g) * p + (2*h)][col] = 1      # q00: j=2g, k=2h
            # C10: i=1, k=2h -> W index = 2h*2+1 = 4h+1
            W[(2*h) * n + 1][col] = 1       # C10
            col += 1
            
            # m3 = p00(q01-q11)
            U[0 * m + (2*g)][col] = 1    # p00: i=0, j=2g
            V[(2*g) * p + (2*h+1)][col] = 1    # q01: j=2g, k=2h+1
            V[(2*g+1) * p + (2*h+1)][col] = -1 # q11: j=2g+1, k=2h+1
            # C01: i=0, k=2h+1 -> W index = (2h+1)*2+0 = 4h+2
            W[(2*h+1) * n + 0][col] = 1       # C01
            col += 1
            
            # m4 = p11(q10-q00)
            U[1 * m + (2*g+1)][col] = 1  # p11: i=1, j=2g+1
            V[(2*g+1) * p + (2*h)][col] = 1    # q10: j=2g+1, k=2h
            V[(2*g) * p + (2*h)][col] = -1     # q00: j=2g, k=2h
            # C00: i=0, k=2h -> W index = 4h
            # C10: i=1, k=2h -> W index = 4h+1
            W[(2*h) * n + 0][col] = 1       # C00
            W[(2*h) * n + 1][col] = 1       # C10
            col += 1
            
            # m5 = (p00+p01)q11
            U[0 * m + (2*g)][col] = 1    # p00: i=0, j=2g
            U[0 * m + (2*g+1)][col] = 1  # p01: i=0, j=2g+1
            V[(2*g+1) * p + (2*h+1)][col] = 1  # q11: j=2g+1, k=2h+1
            # C00: i=0, k=2h -> W index = 4h
            # C01: i=0, k=2h+1 -> W index = 4h+2
            W[(2*h) * n + 0][col] = -1       # C00 (with - sign)
            W[(2*h+1) * n + 0][col] = 1      # C01
            col += 1
            
            # m6 = (p10-p00)(q00+q01)
            U[1 * m + (2*g)][col] = 1    # p10: i=1, j=2g
            U[0 * m + (2*g)][col] = -1   # p00: i=0, j=2g
            V[(2*g) * p + (2*h)][col] = 1      # q00: j=2g, k=2h
            V[(2*g) * p + (2*h+1)][col] = 1    # q01: j=2g, k=2h+1
            # C11: i=1, k=2h+1 -> W index = (2h+1)*2+1 = 4h+3
            W[(2*h+1) * n + 1][col] = 1       # C11
            col += 1
            
            # m7 = (p01-p11)(q10+q11)
            U[0 * m + (2*g+1)][col] = 1  # p01: i=0, j=2g+1
            U[1 * m + (2*g+1)][col] = -1 # p11: i=1, j=2g+1
            V[(2*g+1) * p + (2*h)][col] = 1    # q10: j=2g+1, k=2h
            V[(2*g+1) * p + (2*h+1)][col] = 1  # q11: j=2g+1, k=2h+1
            # C00: i=0, k=2h -> W index = 4h
            W[(2*h) * n + 0][col] = 1       # C00
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
