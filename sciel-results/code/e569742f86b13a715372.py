def construct():
    """
    Construct a rank-7 tensor decomposition for 2x2 matrix multiplication
    using Strassen's algorithm. This is provably optimal.
    
    The decomposition uses the standard Strassen bilinear form with 7 products.
    Factor matrices U, V, W have shapes (4,7), (4,7), (4,7) respectively.
    """
    rank = 7
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(4)]  # (4, 7)
    V = [[0] * rank for _ in range(4)]  # (4, 7)
    W = [[0] * rank for _ in range(4)]  # (4, 7)
    
    # Strassen's 7 products with explicit structure constants:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-b)
    # M5 = (a+b)h
    # M6 = (c-d)(g+h)
    # M7 = (a-c)(b+d)
    #
    # Indexing:
    # A = [[a, b], [c, d]] -> U indices: 0=a, 1=b, 2=c, 3=d
    # B = [[e, f], [g, h]] -> V indices: 0=e, 1=f, 2=g, 3=h
    # P = [[p11, p12], [p21, p22]] -> W indices: 0=p11, 1=p12, 2=p21, 3=p22
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-b)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-d)(g+h)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a-c)(b+d)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
