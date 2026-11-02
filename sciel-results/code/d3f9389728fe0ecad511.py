import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    We use Strassen's algorithm for each pair of columns. Since p=12 is even,
    we have 6 pairs of columns, each requiring rank 7 (Strassen).
    Total rank = 6 * 7 = 42.
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    where the tensor encodes C[i,k] = sum_j A[i,j] * B[j,k].
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    # Initialize matrices with zeros
    # Using lists of lists to match the required interface format
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
    # Define Strassen coefficients for a single 2x2x2 block
    # Each term is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local A index (0..3) to coefficient
    # v_coeffs: dict mapping local B index (0..3) to coefficient
    # w_coeffs: dict mapping local C index (0..3) to coefficient
    #
    # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00 + A11) * (B00 + B11)
    # C00 += M1, C11 += M1
    strassen_terms = [
        # (U parts, V parts, W parts)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    for t in range(num_pairs):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            u_parts, v_parts, w_parts = strassen_terms[s]
            
            # Fill U
            for a_idx, val in u_parts.items():
                # Global U index: i*m + j
                # a_idx 0 -> i=0, j=0 -> 0
                # a_idx 1 -> i=0, j=1 -> 1
                # a_idx 2 -> i=1, j=0 -> 2
                # a_idx 3 -> i=1, j=1 -> 3
                U[a_idx][col] += val
            
            # Fill V
            for b_idx, val in v_parts.items():
                # Map local B index to global V index: j*p + k
                if b_idx == 0: # B[0, 2t]
                    global_v = 0 * p + (2 * t)
                elif b_idx == 1: # B[0, 2t+1]
                    global_v = 0 * p + (2 * t + 1)
                elif b_idx == 2: # B[1, 2t]
                    global_v = 1 * p + (2 * t)
                else: # b_idx == 3 # B[1, 2t+1]
                    global_v = 1 * p + (2 * t + 1)
                
                V[global_v][col] += val
            
            # Fill W
            for c_idx, val in w_parts.items():
                # Map local C index to global W index: k*n + i
                if c_idx == 0: # C[0, 2t] -> i=0, k=2t
                    global_w = (2 * t) * n + 0
                elif c_idx == 1: # C[0, 2t+1] -> i=0, k=2t+1
                    global_w = (2 * t + 1) * n + 0
                elif c_idx == 2: # C[1, 2t] -> i=1, k=2t
                    global_w = (2 * t) * n + 1
                else: # c_idx == 3 # C[1, 2t+1] -> i=1, k=2t+1
                    global_w = (2 * t + 1) * n + 1
                
                W[global_w][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
