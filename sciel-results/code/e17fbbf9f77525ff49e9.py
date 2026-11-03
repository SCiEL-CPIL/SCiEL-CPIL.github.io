import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank is n*m*p for schoolbook
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices of appropriate shapes
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct the decomposition using a single loop over the rank index
    # This structure is different from the nested loop in the seed program
    for r in range(rank):
        # Decompose the linear index r into (i, j, k)
        # Order: i is slowest, k is fastest? 
        # In the seed: 
        # for i:
        #   for j:
        #     for k:
        #       col += 1
        # So r = i * (m*p) + j * p + k
        
        i = r // (m * p)
        remainder = r % (m * p)
        j = remainder // p
        k = remainder % p
        
        # Set the corresponding entries to 1
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    # Convert to lists of lists to match the expected interface if strict, 
    # or return numpy arrays if allowed. 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # It doesn't explicitly forbid numpy arrays, but "return interface EXACTLY as in current program"
    # Current program returns dicts of lists of lists.
    # To be safe and ensure compatibility, convert to lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
