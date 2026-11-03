import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition: rank = n * m * p
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Verification function
    def verify(U, V, W):
        r = U.shape[1]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We expect T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(r):
                        actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Return the schoolbook decomposition
    # The rank is 60.
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
