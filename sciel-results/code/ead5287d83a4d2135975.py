"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 23 (Laderman) + 23 (Laderman) + 9 (Schoolbook for 3x3x1) = 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_laderman(offset_v, offset_w, start_col):
        # Laderman's 23-term algorithm for 3x3x3
        # Local indices i,j,k in 0..2
        # Global U index: 3*i + j
        # Global V index: 7*j + (k + offset_v)
        # Global W index: 3*(k + offset_w) + i
        
        terms = [
            # 1. a00 b00
            [(0,0,1), (0,0,1), (0,0,1)],
            # 2. a11 b11
            [(1,1,1), (1,1,1), (1,1,1)],
            # 3. a22 b22
            [(2,2,1), (2,2,1), (2,2,1)],
            # 4. (a00+a11)(b00+b11)
            [(0,0,1), (1,1,1), (0,0,1), (1,1,1), (0,0,1), (1,1,1)], # Wait, W needs to map to C.
            # This manual expansion is too hard to get right without a reference.
        ]
        # I will use a verified list of 23 triples (U_entries, V_entries, W_entries)
        # Each entry is (index, coeff)
        
        # Reference: Laderman (1976)
        # p1: U[0,0]=1, V[0,0]=1, W[0,0]=1
        # p2: U[1,1]=1, V[1,1]=1, W[1,1]=1
        # p3: U[2,2]=1, V[2,2]=1, W[2,2]=1
        # p4: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1, W[0,1]=1, W[1,0]=1
        # ...
        
        # To avoid errors, I will use a known valid decomposition if I can recall it.
        # If not, I will fallback to schoolbook.
        
        pass

    # Given the high risk of hallucinating Laderman coefficients,
    # and the fact that 63 is the current score,
    # I will attempt to use a simpler improvement if possible.
    # Is there a rank 56 algorithm?
    # 27 + 27 + 9 = 63.
    # If I use 23, it's 55.
    
    # I will implement Laderman carefully.
    
    laderman_terms = [
        # (U_entries, V_entries, W_entries)
        # U: (i,j,coeff), V: (j,k,coeff), W: (k,i,coeff)
        
        # 1
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # 2
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # 3
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # 4
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1), (0,1,1), (1,0,1)]),
        # 5
        ([(1,1,1), (2,2,1)], [(1,1,1), (2,2,1)], [(1,1,1), (2,2,1), (1,2,1), (2,1,1)]),
        # 6
        ([(2,2,1), (0,0,1)], [(2,2,1), (0,0,1)], [(2,2,1), (0,0,1), (2,0,1), (0,2,1)]),
        # 7
        ([(0,0,1), (0,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1), (0,1,1), (1,1,1)]), # Approx
        # 8
        ([(1,1,1), (1,2,1)], [(1,1,1), (2,1,1)], [(1,1,1), (2,1,1), (1,2,1), (2,2,1)]),
        # 9
        ([(2,2,1), (2,0,1)], [(2,2,1), (0,2,1)], [(2,2,1), (0,2,1), (2,0,1), (0,0,1)]),
        # 10
        ([(0,1,1), (0,2,1)], [(0,1,1), (2,1,1)], [(0,1,1), (2,1,1), (0,2,1), (2,2,1)]),
        # 11
        ([(1,0,1), (1,2,1)], [(1,0,1), (2,2,1)], [(1,0,1), (2,2,1), (1,2,1), (2,0,1)]),
        # 12
        ([(2,0,1), (2,1,1)], [(2,0,1), (1,2,1)], [(2,0,1), (1,2,1), (2,1,1), (1,1,1)]),
        # 13
        ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
        # 14
        ([(0,0,1), (1,1,1), (2,2,-1)], [(0,0,1), (1,1,1), (2,2,-1)], [(0,0,1), (1,1,1), (2,2,-1)]),
        # 15
        ([(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,-1), (2,2,1)]),
        # 16
        ([(0,0,1), (1,1,-1), (2,2,-1)], [(0,0,1), (1,1,-1), (2,2,-1)], [(0,0,1), (1,1,-1), (2,2,-1)]),
        # 17
        ([(0,1,1), (1,2,1), (2,0,1)], [(0,1,1), (1,2,1), (2,0,1)], [(0,1,1), (1,2,1), (2,0,1)]),
        # 18
        ([(0,2,1), (1,0,1), (2,1,1)], [(0,2,1), (1,0,1), (2,1,1)], [(0,2,1), (1,0,1), (2,1,1)]),
        # 19
        ([(0,0,1), (0,1,1), (0,2,1)], [(0,0,1), (1,0,1), (2,0,1)], [(0,0,1), (1,0,1), (2,0,1)]),
        # 20
        ([(1,0,1), (1,1,1), (1,2,1)], [(0,1,1), (1,1,1), (2,1,1)], [(0,1,1), (1,1,1), (2,1,1)]),
        # 21
        ([(2,0,1), (2,1,1), (2,2,1)], [(0,2,1), (1,2,1), (2,2,1)], [(0,2,1), (1,2,1), (2,2,1)]),
        # 22
        ([(0,0,1), (1,0,1), (2,0,1)], [(0,0,1), (0,1,1), (0,2,1)], [(0,0,1), (0,1,1), (0,2,1)]),
        # 23
        ([(0,1,1), (1,1,1), (2,1,1)], [(1,0,1), (1,1,1), (1,2,1)], [(1,0,1), (1,1,1), (1,2,1)]),
    ]
    
    # This list is GUESSED. It is likely incorrect.
    # I will fallback to Schoolbook to ensure validity.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
