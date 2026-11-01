"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use rank 40 (improvement over 42)
    rank = 40
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen coefficients for 2x2x2
    # Each entry is a list of (local_idx, coeff) for U, V, W
    # U local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    col = 0
    # Iterate over blocks of size 2 in the m dimension
    for block in range(m // 2):
        # Iterate over blocks of size 2 in the p dimension
        for sub in range(p // 2):
            # Apply Strassen for the 2x2x2 subproblem
            for s_idx in range(7):
                term = strassen_terms[s_idx]
                
                # Update U
                for local_u, coeff in term['U']:
                    a = local_u // 2
                    l = local_u % 2
                    global_u = a * m + (2 * block + l)
                    U[global_u, col] += coeff
                
                # Update V
                for local_v, coeff in term['V']:
                    l = local_v // 2
                    c = local_v % 2
                    global_v = (2 * block + l) * p + (2 * sub + c)
                    V[global_v, col] += coeff
                
                # Update W
                for local_w, coeff in term['W']:
                    a = local_w // 2
                    c = local_w % 2
                    global_w = (2 * sub + c) * n + a
                    W[global_w, col] += coeff
                
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
