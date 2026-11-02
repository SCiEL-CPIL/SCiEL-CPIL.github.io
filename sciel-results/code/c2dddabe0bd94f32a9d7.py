"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Uses schoolbook decomposition with rank n*m*p = 32.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 16
    W_dim = p * n  # 8
    
    rank = n * m * p  # 32
    
    U_mat = np.zeros((U_dim, rank))
    V_mat = np.zeros((V_dim, rank))
    W_mat = np.zeros((W_dim, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i * m + j, col] = 1
                V_mat[j * p + k, col] = 1
                W_mat[k * n + i, col] = 1
                col += 1
    
    # Verify
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
    if not np.allclose(recon, T, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    # Convert to lists
    U_list = [[float(U_mat[i, r]) for r in range(rank)] for i in range(U_dim)]
    V_list = [[float(V_mat[i, r]) for r in range(rank)] for i in range(V_dim)]
    W_list = [[float(W_mat[i, r]) for r in range(rank)] for i in range(W_dim)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
