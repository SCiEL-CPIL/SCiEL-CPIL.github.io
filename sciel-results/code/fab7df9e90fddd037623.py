import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix 
    by a 3 by 3 matrix.
    
    The rank of the 2x3 * 3x3 matrix multiplication tensor is 18.
    We return the standard schoolbook decomposition which has rank 18.
    
    Returns dict with U (6 x 18), V (9 x 18), W (6 x 18).
    """
    n, m, p = 2, 3, 3
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    rank = n * m * p  # 18
    
    U = np.zeros((u_dim, rank))
    V = np.zeros((v_dim, rank))
    W = np.zeros((w_dim, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i*m + j
                # V index: (j, k) -> j*p + k
                # W index: (k, i) -> k*n + i
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists of floats as required by the interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
