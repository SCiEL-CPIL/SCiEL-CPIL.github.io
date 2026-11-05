import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize arrays with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized construction
    # We need to map each (i, j, k) to a column index r
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Create index arrays
    # We can iterate over the rank dimension or use broadcasting.
    # Since rank is small (105), a loop over the rank is fine, but let's use numpy indexing.
    
    # Generate all combinations of i, j, k
    # i: 0..n-1, j: 0..m-1, k: 0..p-1
    # r = i * m * p + j * p + k
    
    # Flatten indices
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    r_vals = (i_vals * m * p + j_vals * p + k_vals).flatten()
    
    # Flatten i, j, k to match row indices of U, V, W
    # U rows: i * m + j
    # V rows: j * p + k
    # W rows: k * n + i
    
    u_rows = (i_vals * m + j_vals).flatten()
    v_rows = (j_vals * p + k_vals).flatten()
    w_rows = (k_vals * n + i_vals).flatten()
    
    # Assign 1s
    U[u_rows, r_vals] = 1.0
    V[v_rows, r_vals] = 1.0
    W[w_rows, r_vals] = 1.0
    
    # Convert to lists as required by the interface (or return arrays? 
    # The seed returns lists of lists. The evaluator might accept arrays.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually numpy arrays are accepted. If not, convert to list.
    # The seed uses lists. I will return numpy arrays as they are standard in this context, 
    # but to be safe and match the seed's type exactly, I'll convert to lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
