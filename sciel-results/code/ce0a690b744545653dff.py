import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    
    # Rank 35 decomposition using Strassen's algorithm on 5 blocks of 2 columns
    rank = 7 * num_blocks
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Indices in U: 0:a, 1:b, 2:c, 3:d
    # Indices in V (local): 0:e, 1:f, 2:g, 3:h
    # Indices in W (local): 0:C00, 1:C01, 2:C10, 3:C11
    
    # Define the 7 Strassen products
    # Each product is defined by:
    # u_coeffs: dict or list of (index, coeff) for A
    # v_coeffs: dict or list of (index, coeff) for B
    # w_coeffs: dict or list of (index, coeff) for C
    
    # P1 = (a + d)(e + h)
    # P2 = (c + d)e
    # P3 = a(f - h)
    # P4 = d(g - e)
    # P5 = (a + b)h
    # P6 = (c - a)(e + f)
    # P7 = (b - d)(g + h)
    
    strassen_products = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),      # P1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),            # P2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),            # P3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),            # P4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),            # P5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),            # P6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),            # P7
    ]
    
    for b in range(num_blocks):
        base_rank = b * 7
        
        # V index offset for this block:
        # e = B[0, 2b] -> V[0*10 + 2b]
        # f = B[0, 2b+1] -> V[0*10 + 2b+1]
        # g = B[1, 2b] -> V[1*10 + 2b]
        # h = B[1, 2b+1] -> V[1*10 + 2b+1]
        
        # W index offset for this block:
        # C[0, 2b] -> W[2b*2 + 0]
        # C[0, 2b+1] -> W[(2b+1)*2 + 0]
        # C[1, 2b] -> W[2b*2 + 1]
        # C[1, 2b+1] -> W[(2b+1)*2 + 1]
        
        for k in range(7):
            r = base_rank + k
            u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff = strassen_products[k]
            
            # Fill U
            for idx, coeff in zip(u_idx, u_coeff):
                U[idx, r] = coeff
                
            # Fill V
            # Map local V indices (0:e, 1:f, 2:g, 3:h) to global
            # e -> 2b, f -> 2b+1, g -> 10+2b, h -> 10+2b+1
            for idx, coeff in zip(v_idx, v_coeff):
                if idx == 0:
                    V[2*b, r] = coeff
                elif idx == 1:
                    V[2*b+1, r] = coeff
                elif idx == 2:
                    V[10+2*b, r] = coeff
                elif idx == 3:
                    V[10+2*b+1, r] = coeff
                    
            # Fill W
            # Map local W indices (0:C00, 1:C01, 2:C10, 3:C11) to global
            # C00 -> 4b, C01 -> 4b+2, C10 -> 4b+1, C11 -> 4b+3
            for idx, coeff in zip(w_idx, w_coeff):
                if idx == 0:
                    W[4*b, r] = coeff
                elif idx == 1:
                    W[4*b+2, r] = coeff
                elif idx == 2:
                    W[4*b+1, r] = coeff
                elif idx == 3:
                    W[4*b+3, r] = coeff
                    
    # Convert to lists as required by some evaluators, though numpy arrays are often accepted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning lists is safer for strict type checking if not specified as array.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
