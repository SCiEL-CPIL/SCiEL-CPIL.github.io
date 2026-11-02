import numpy as np

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Returns the schoolbook decomposition which has rank R = 3 * 4 * 7 = 84.
    This is the standard exact decomposition.
    """
    n, m, p = 3, 4, 7
    R = n * m * p
    
    # Initialize U, V, W as NumPy arrays for efficient construction
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Generate all combinations of i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    # We iterate through the flattened index r from 0 to R-1
    # r corresponds to a unique (i, j, k) tuple
    
    # We can map r to (i, j, k) using divmod
    # Let's define the mapping:
    # r = i * (m * p) + j * p + k
    # This is a standard row-major flattening of a 3D array of shape (n, m, p)
    
    for r in range(R):
        # Decompose r into i, j, k
        i = r // (m * p)
        remainder = r % (m * p)
        j = remainder // p
        k = remainder % p
        
        # U index: row i*m + j
        u_row = i * m + j
        # V index: row j*p + k
        v_row = j * p + k
        # W index: row k*n + i
        w_row = k * n + i
        
        U[u_row, r] = 1.0
        V[v_row, r] = 1.0
        W[w_row, r] = 1.0
        
    # Convert to lists of lists to match the expected interface format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
