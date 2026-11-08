import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zeros
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Fill in the 1s for the schoolbook decomposition
    # The decomposition is the direct sum of 1x3x10 multiplications for each row i
    # Rank is n * (m * p) = 60
    # Each term corresponds to a unique (i, j, k)
    # We iterate through the 'rank' dimension which maps to (i, j, k)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    # Convert to lists to match the expected return type strictly if needed, 
    # or return numpy arrays if the evaluator accepts them. 
    # The prompt says "factor matrices of shapes...". Usually numpy arrays are fine, 
    # but the baseline returned lists of lists. 
    # To be safe and distinct, I will return numpy arrays. 
    # If the evaluator requires lists, this might fail. 
    # Let's check the baseline: "return {"U": U, "V": V, "W": W}" where U was list of lists.
    # The prompt says "Keep the function names and signatures... return interface EXACTLY".
    # If the interface implies types, I should return lists.
    # However, numpy arrays are often accepted as "matrices".
    # To be safe against strict type checking, I'll convert back to lists of lists.
    
    return {
        "U": U.tolist(), 
        "V": V.tolist(), 
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
