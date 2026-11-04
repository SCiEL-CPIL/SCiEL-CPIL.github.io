import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a Strassen-based decomposition.
    The 4x4x8 multiplication is decomposed into two 4x4x4 multiplications.
    Each 4x4x4 uses a rank-49 Strassen decomposition.
    Total rank: 98.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # --- Step 1: Construct rank-49 decomposition for 4x4x4 ---
    
    # Strassen coefficients
    # Outer Strassen (4x4 blocks into 2x2 blocks of 2x2 matrices)
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3)
    # B blocks: B00, B01, B10, B11
    
    alpha = {
        1: {0: 1, 3: 1},  # A00 + A11
        2: {2: 1, 3: 1},  # A10 + A11
        3: {0: 1},        # A00
        4: {3: 1},        # A11
        5: {0: 1, 1: 1},  # A00 + A01
        6: {2: 1, 0: -1}, # A10 - A00
        7: {1: 1, 3: -1}  # A01 - A11
    }
    
    beta = {
        1: {0: 1, 3: 1},  # B00 + B11
        2: {0: 1},        # B00
        3: {1: 1, 3: -1}, # B01 - B11
        4: {2: 1, 0: -1}, # B10 - B00
        5: {3: 1},        # B11
        6: {0: 1, 1: 1},  # B00 + B01
        7: {2: 1, 3: 1}   # B10 + B11
    }
    
    # Inner Strassen (2x2 blocks)
    # m_s = u_s * v_s
    # u_s = sum_ab delta_u[s][ab] * A_ab
    # v_s = sum_ab delta_v[s][ab] * B_ab
    # M_ab = sum_s gamma[s][ab] * m_s
    
    delta_u = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    delta_v = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    gamma = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,1): 1, (0,0): -1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    # Outer Strassen output assembly
    # C_PQ = sum_t kappa[t][PQ] * M_t
    
    kappa = {
        1: {0: 1, 3: 1},  # C00 += M1, C11 += M1
        2: {2: 1, 3: -1}, # C10 += M2, C11 -= M2
        3: {1: 1, 3: 1},  # C01 += M3, C11 += M3
        4: {0: 1, 2: 1},  # C00 += M4, C10 += M4
        5: {0: -1, 1: 1}, # C00 -= M5, C01 += M5
        6: {3: 1},        # C11 += M6
        7: {0: 1}         # C00 += M7
    }
    
    # Initialize 4x4x4 factors
    # U444: 16 x 49 (i*4+j)
    # V444: 16 x 49 (j*4+k)
    # W444: 16 x 49 (k*4+i)
    
    U444 = np.zeros((16, 49), dtype=float)
    V444 = np.zeros((16, 49), dtype=float)
    W444 = np.zeros((16, 49), dtype=float)
    
    for t in range(1, 8):
        for s in range(1, 8):
            r = (t - 1) * 7 + (s - 1)
            
            # U444 coefficients
            for a in range(2):
                for b in range(2):
                    coeff_u = delta_u[s].get((a,b), 0)
                    if coeff_u == 0: continue
                    
                    for P, alpha_val in alpha[t].items():
                        coeff_total = coeff_u * alpha_val
                        if coeff_total == 0: continue
                        
                        P_row = P // 2
                        P_col = P % 2
                        i = 2 * P_row + a
                        j = 2 * P_col + b
                        u_idx = i * 4 + j
                        U444[u_idx, r] += coeff_total
            
            # V444 coefficients
            for a in range(2):
                for b in range(2):
                    coeff_v = delta_v[s].get((a,b), 0)
                    if coeff_v == 0: continue
                    
                    for Q, beta_val in beta[t].items():
                        coeff_total = coeff_v * beta_val
                        if coeff_total == 0: continue
                        
                        Q_row = Q // 2
                        Q_col = Q % 2
                        j = 2 * Q_row + a
                        k = 2 * Q_col + b
                        v_idx = j * 4 + k
                        V444[v_idx, r] += coeff_total
            
            # W444 coefficients
            for a in range(2):
                for b in range(2):
                    coeff_m = gamma[s].get((a,b), 0)
                    if coeff_m == 0: continue
                    
                    for PQ, kappa_val in kappa[t].items():
                        coeff_total = coeff_m * kappa_val
                        if coeff_total == 0: continue
                        
                        PQ_row = PQ // 2
                        PQ_col = PQ % 2
                        i = 2 * PQ_row + a
                        k = 2 * PQ_col + b
                        w_idx = k * 4 + i
                        W444[w_idx, r] += coeff_total

    # --- Step 2: Construct 4x4x8 factors ---
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # Copy 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            
    # Copy 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8*j + k, 49:98] = V444[4*j + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4*k + i, 49:98] = W444[4*(k-4) + i, :]

    # Deadline check
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
