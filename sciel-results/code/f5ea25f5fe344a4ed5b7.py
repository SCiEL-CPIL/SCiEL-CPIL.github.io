import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill in the schoolbook decomposition
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c corresponds to (i, j, k)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface example 
    # (though numpy arrays might be accepted, the prompt example used lists)
    # The prompt says: "factor matrices of shapes ... with integer/half-integer entries"
    # And the seed used lists. I will return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check time
    if time.time() > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
