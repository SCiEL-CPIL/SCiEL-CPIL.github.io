import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition based on 4 blocks of Strassen's rank 7 algorithm
    rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry is a tuple (U_2x2, V_2x2, W_2x2)
    # U, V, W are 2x2 matrices
    strassen_coeffs = [
        # 1
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 1], [1, 1]])),
        # 2
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, -1]])),
        # 3
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]), np.array([[0, 1], [0, 0]])),
        # 4
        (np.array([[0, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[0, 0], [1, 0]])),
        # 5
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [-1, 0]])),
        # 6
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
        # 7
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 0]]))
    ]
    
    # There are 4 blocks of 2x2 matrices in the 2x8 matrix B
    num_blocks = p // 2 # 4 blocks
    
    for q in range(num_blocks):
        for s in range(7):
            r = q * 7 + s
            U_s, V_s, W_s = strassen_coeffs[s]
            
            # Fill U
            # U index: i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = U_s[i, j]
                    
            # Fill V
            # V index: j*p + k
            # k is local to block q, so global k = q*2 + k_local
            for j in range(m):
                for k_local in range(2):
                    k_global = q * 2 + k_local
                    V[j * p + k_global, r] = V_s[j, k_local]
                    
            # Fill W
            # W index: k*n + i
            # k is local to block q, so global k = q*2 + k_local
            for k_local in range(2):
                k_global = q * 2 + k_local
                for i in range(n):
                    W[k_global * n + i, r] = W_s[k_local, i]
                    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
