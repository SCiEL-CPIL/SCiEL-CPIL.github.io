import numpy as np

def _laderman_2x2x2():
    """Return (U, V, W) of shape (4,7) each for 2x2 * 2x2 matrix mult."""
    # Laderman's rank-7 algorithm for 2x2x2 tensor
    # Indices: A is 2x2, B is 2x2, C = A*B
    # a00,a01,a10,a11; b00,b01,b10,b11
    # 7 rank-1 terms
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # term 0: (a00+a11)(b00+b11) -> C00
    U[0,0] = 1; U[1,0] = 1; U[2,0] = 0; U[3,0] = 1  # a00 + a11
    V[0,0] = 1; V[1,0] = 0; V[2,0] = 0; V[3,0] = 1  # b00 + b11
    W[0,0] = 1; W[1,0] = 0; W[2,0] = 0; W[3,0] = 0  # C00
    # term 1: (a00+a11)(b00-b11) -> C11
    U[:,1] = [1, 0, 0, 1]
    V[:,1] = [1, 0, 0, -1]
    W[:,1] = [0, 0, 0, 1]
    # term 2: (a10+a11)b01 -> C01
    U[:,2] = [0, 0, 1, 1]
    V[:,2] = [0, 1, 0, 0]
    W[:,2] = [0, 1, 0, 0]
    # term 3: (a00+a01)b10 -> C10
    U[:,3] = [1, 1, 0, 0]
    V[:,3] = [0, 0, 1, 0]
    W[:,3] = [0, 0, 1, 0]
    # term 4: (a10+a01)(b10-b01) -> correction
    U[:,4] = [0, 1, 1, 0]
    V[:,4] = [0, -1, 1, 0]
    W[:,4] = [0, 0, 0, 0]
    # term 5: (a10-a01)(b00+b01) -> C00 correction
    U[:,5] = [0, -1, 1, 0]
    V[:,5] = [1, 1, 0, 0]
    W[:,5] = [1, 0, 0, 0]
    # term 6: (a11-a10)(b10+b11) -> C11 correction
    U[:,6] = [0, 0, -1, 1]
    V[:,6] = [0, 0, 1, 1]
    W[:,6] = [0, 0, 0, 1]
    return U, V, W

def construct():
    n, m, p = 2, 3, 5
    
    # Schoolbook construction for 2x3 * 3x5
    # Rank is n*m*p = 30
    # U shape: (n*m, rank) = (6, 30)
    # V shape: (m*p, rank) = (15, 30)
    # W shape: (p*n, rank) = (10, 30)
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i (trace convention W[k*n+i, r])
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
