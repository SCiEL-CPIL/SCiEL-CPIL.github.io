"""Matrix multiplication 4x4x6 - structured low-rank decomposition."""
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """Return Strassen's rank-7 decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0]]
    V = [[0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0]]
    W = [[0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0]]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # k=0,i=0 -> C[0,0]
    W[3][0] = 1  # k=1,i=1 -> C[1,1]
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # k=1,i=0 -> C[0,1]... wait
    # C[0,1] means row 0, col 1. W index = k*n+i = 1*2+0 = 2. Yes.
    W[3][1] = -1 # k=1,i=1 -> C[1,1]
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # k=0,i=1 -> C[1,0]... 
    # C[1,0] means row 1, col 0. W index = 0*2+1 = 1. Yes.
    W[3][2] = 1  # k=1,i=1 -> C[1,1]
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1 # k=0,i=0 -> C[0,0]
    W[2][3] = 1  # k=1,i=0 -> C[0,1]... 
    # C[0,1] means row 0, col 1. W index = 1*2+0 = 2. Yes.
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1  # k=0,i=0 -> C[0,0]
    W[1][4] = 1  # k=0,i=1 -> C[1,0]
    
    # M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1  # k=1,i=1 -> C[1,1]
    
    # M7 = (b-c)(e+g)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = -1 # k=0,i=0 -> C[0,0]
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    
    S_U, S_V, S_W = _strassen_2x2()
    
    rank = 84
    U = [[0]*rank for _ in range(16)]  # 4*4
    V = [[0]*rank for _ in range(24)]  # 4*6
    W = [[0]*rank for _ in range(24)]  # 6*4
    
    # 12 products of 2x2 matrices
    products = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 2, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 2, 1, 0, 1),
        (0, 0, 0, 2, 0, 2),
        (0, 1, 2, 2, 0, 2),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 2, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 2, 1, 1, 1),
        (1, 0, 0, 2, 1, 2),
        (1, 1, 2, 2, 1, 2),
    ]
    
    for prod_idx, (ar, ac, br, bc, cr, cc) in enumerate(products):
        base = prod_idx * 7
        for s in range(7):
            col = base + s
            for i in range(2):
                for j in range(2):
                    a_idx = (ar * 2 + i) * 4 + (ac * 2 + j)
                    U[a_idx][col] = S_U[i * 2 + j][s]
                    
                    b_idx = (br * 2 + i) * 6 + (bc * 2 + j)
                    V[b_idx][col] = S_V[i * 2 + j][s]
                    
                    # W index: k*n + i where k=col, i=row
                    c_idx = (cc * 2 + j) * 4 + (cr * 2 + i)
                    W[c_idx][col] = S_W[i * 2 + j][s]
    
    return {"U": U, "V": V, "W": W}
