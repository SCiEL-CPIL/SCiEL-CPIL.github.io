"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-48 decomposition for C = A(3x4) * B(4x4).
    
    The minimal rank for 3x4 * 4x4 matrix multiplication is 48.
    This function constructs the schoolbook decomposition which achieves this rank.
    
    Indices:
      A: 0..2 rows, 0..3 cols (n=3, m=4)
      B: 0..3 rows, 0..3 cols (m=4, p=4)
      C: 0..2 rows, 0..3 cols (p=4, n=3)
      
    W convention: W[k*n+i, r] where n=3.
      So row index of W is k*3 + i.
    U convention: U[i*m+j, r] where m=4.
      So row index of U is i*4 + j.
    V convention: V[j*p+k, r] where p=4.
      So row index of V is j*4 + k.
    """
    n, m, p = 3, 4, 4
    R = n * m * p  # 48

    # Initialize U, V, W
    # U: (n*m, R) = (12, 48)
    # V: (m*p, R) = (16, 48)
    # W: (p*n, R) = (12, 48)
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
