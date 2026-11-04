import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use vectorized index generation to determine positions
    # This avoids nested Python loops for the assignment, which is the "new approach"
    # in terms of computational strategy (vectorized scatter vs iterative assignment).
    
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    c_flat = (i_flat * m + j_flat) * p + k_flat
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Assign to the list structures
    # Since we are assigning to lists, we still need a loop or list comprehension over the flattened indices
    # But the *logic* for determining *which* index goes where is vectorized.
    # To truly avoid the loop, we could build numpy arrays and convert at the end.
    
    # Let's build numpy arrays and convert to lists at the end to ensure type safety and speed.
    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)
    
    U_np[u_rows, c_flat] = 1
    V_np[v_rows, c_flat] = 1
    W_np[w_rows, c_flat] = 1
    
    # Convert to lists of lists to match seed interface
    U_list = U_np.tolist()
    V_list = V_np.tolist()
    W_list = W_np.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
