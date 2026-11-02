import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Convert to lists as required by interface? 
    # The interface says "factor matrices of shapes...". 
    # The seed program returns lists of lists. 
    # The evaluator might accept numpy arrays, but lists are safer if not specified.
    # The seed uses lists. I will return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > deadline:
        # Should not happen for this small size, but guard just in case
        pass
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
