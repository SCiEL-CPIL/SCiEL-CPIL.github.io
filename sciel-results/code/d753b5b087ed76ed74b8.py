import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Each term: (u_dict, v_dict, w_dict)
    # u: A_idx -> coeff, v: local_B_idx -> coeff, w: local_C_idx -> coeff
    strassen_terms = [
        # M1
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        # M2
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        # M3
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        # M4
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        # M5
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        # M6
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        # M7
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]
    
    # Build U, V, W as numpy arrays for efficiency
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b, col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val
    
    # Convert back to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
