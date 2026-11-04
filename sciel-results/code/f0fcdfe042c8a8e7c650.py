"""Program using Strassen's algorithm for 4x4 * 4x9 matrix multiplication."""

def construct():
    """
    Returns a rank-144 tensor decomposition (schoolbook) for 4x4 * 4x9.
    
    We use the standard schoolbook multiplication which gives rank = 4*4*9 = 144.
    
    The interface expects:
    - U: shape (n*m, rank) = (16, rank)
    - V: shape (m*p, rank) = (36, rank) 
    - W: shape (p*n, rank) = (36, rank)
    
    where n=4, m=4, p=9
    
    For the schoolbook decomposition:
    - U[i*m + j, col] = 1 if col corresponds to (i,j,k), else 0
    - V[j*p + k, col] = 1 if col corresponds to (i,j,k), else 0
    - W[k*n + i, col] = 1 if col corresponds to (i,j,k), else 0
    """
    n, m, p = 4, 4, 9
    rank = n * m * p  # 144
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
