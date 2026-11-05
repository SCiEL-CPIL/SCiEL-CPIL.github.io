import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize matrices as numpy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    # Convert to lists of lists as required by interface {'U', 'V', 'W'} with shapes (n*m, rank) etc.
    # The interface expects the values to be accessible. The example returned lists of lists.
    # I will return lists of lists to be safe, or numpy arrays if allowed. 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually, these evaluators accept numpy arrays or lists. The seed returned lists.
    # I will return lists to match the seed's type exactly, to avoid type errors.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
