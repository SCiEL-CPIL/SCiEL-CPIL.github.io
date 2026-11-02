import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # The rank of M_{n x m x 1} is n*m.
    # The total tensor is a direct sum of p copies of M_{n x m x 1}.
    # Total rank = p * n * m.
    
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct the decomposition
    # For each output column k (0 to p-1)
    for k in range(p):
        # Base rank for this column is n*m
        # We assign n*m consecutive indices in the rank dimension
        base_idx = k * n * m
        
        # For each row i of A (0 to n-1)
        for i in range(n):
            # For each col j of A / row j of B (0 to m-1)
            for j in range(m):
                idx = base_idx + i * m + j
                
                # U index: row i*m + j
                U[i * m + j, idx] = 1
                
                # V index: row j*p + k
                V[j * p + k, idx] = 1
                
                # W index: row k*n + i (trace convention W[k*n+i, r])
                W[k * n + i, idx] = 1
                
    # Convert to lists if the evaluator expects lists, or keep as arrays?
    # The seed program returns lists. The prompt says "factor matrices".
    # Usually, numpy arrays are fine, but to be safe and match the seed interface exactly:
    # "Keep the function names and signatures ... do not rename functions or change what they return."
    # The seed returns a dict of lists. I should probably return a dict of lists to be safe,
    # or check if the evaluator accepts arrays.
    # Given "integer/half-integer entries", lists of ints are safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
