import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 matrices
    # U_out[i, s], V_out[j, k, s], W_out[k, i, s]
    # i, j, k in {0, 1}
    # s in {0, ..., 6}
    
    U_out = np.zeros((2, 7), dtype=int)
    V_out = np.zeros((2, 2, 7), dtype=int)
    W_out = np.zeros((2, 2, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_out[0, 0] = 1
    U_out[1, 0] = 1
    V_out[0, 0, 0] = 1
    V_out[1, 1, 0] = 1
    W_out[0, 0, 0] = 1
    W_out[1, 1, 0] = 1
    
    # M2 = (A10+A11)B00
    U_out[0, 1] = 0
    U_out[1, 1] = 1
    V_out[0, 0, 1] = 1
    W_out[0, 1, 1] = 1
    W_out[1, 1, 1] = -1 # C11 = M1 - M2 ... wait.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify W coefficients.
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    # M1 (s=0)
    W_out[0, 0, 0] = 1 # C00
    W_out[0, 1, 0] = 0 # C01
    W_out[1, 0, 0] = 0 # C10
    W_out[1, 1, 0] = 1 # C11
    
    # M2 (s=1)
    W_out[0, 0, 1] = 0
    W_out[0, 1, 1] = 0
    W_out[1, 0, 1] = 1 # C10
    W_out[1, 1, 1] = -1 # C11
    
    # M3 (s=2)
    W_out[0, 0, 2] = 0
    W_out[0, 1, 2] = 1 # C01
    W_out[1, 0, 2] = 0
    W_out[1, 1, 2] = 1 # C11
    
    # M4 (s=3)
    W_out[0, 0, 3] = 1 # C00
    W_out[0, 1, 3] = 0
    W_out[1, 0, 3] = 1 # C10
    W_out[1, 1, 3] = 0
    
    # M5 (s=4)
    W_out[0, 0, 4] = -1 # C00
    W_out[0, 1, 4] = 1 # C01
    W_out[1, 0, 4] = 0
    W_out[1, 1, 4] = 0
    
    # M6 (s=5)
    W_out[0, 0, 5] = 0
    W_out[0, 1, 5] = 0
    W_out[1, 0, 5] = 0
    W_out[1, 1, 5] = 1 # C11
    
    # M7 (s=6)
    W_out[0, 0, 6] = 1 # C00
    W_out[0, 1, 6] = 0
    W_out[1, 0, 6] = 0
    W_out[1, 1, 6] = 0

    # Fill U_out, V_out completely
    # M1: A00+A11, B00+B11
    U_out[0, 0] = 1; U_out[1, 0] = 1
    V_out[0, 0, 0] = 1; V_out[1, 1, 0] = 1
    
    # M2: A10+A11, B00
    U_out[0, 1] = 0; U_out[1, 1] = 1
    V_out[0, 0, 1] = 1
    
    # M3: A00, B10-B11
    U_out[0, 2] = 1; U_out[1, 2] = 0
    V_out[1, 0, 2] = 1; V_out[1, 1, 2] = -1
    
    # M4: A11, B01-B00
    U_out[0, 3] = 0; U_out[1, 3] = 1
    V_out[0, 1, 3] = 1; V_out[0, 0, 3] = -1
    
    # M5: A00+A01, B11
    U_out[0, 4] = 1; U_out[1, 4] = 0
    V_out[1, 1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U_out[0, 5] = -1; U_out[1, 5] = 1
    V_out[0, 0, 5] = 1; V_out[0, 1, 5] = 1
    
    # M7: A01-A11, B10+B11
    U_out[0, 6] = 0; U_out[1, 6] = -1
    V_out[1, 0, 6] = 1; V_out[1, 1, 6] = 1

    # Inner 2x2x4
    # U_in[i0, t], V_in[j0, k_prime, t], W_in[k_prime, i0, t]
    # i0, j0 in {0, 1}
    # k_prime in {0, 1, 2, 3}
    # t in {0, ..., 13}
    
    U_in = np.zeros((2, 14), dtype=int)
    V_in = np.zeros((2, 4, 14), dtype=int)
    W_in = np.zeros((4, 2, 14), dtype=int)
    
    for t in range(14):
        s = t % 7
        side = t // 7
        
        # U_in is same as U_out for this s
        U_in[0, t] = U_out[0, s]
        U_in[1, t] = U_out[1, s]
        
        # V_in: j0, k_prime
        # V_out[j, k, s]
        # k_prime = 2*side + k_col
        # If side=0, k_prime in {0, 1}. k_col = k_prime.
        # If side=1, k_prime in {2, 3}. k_col = k_prime - 2.
        for j0 in range(2):
            for k_prime in range(4):
                k_col = k_prime % 2
                # Only non-zero if k_prime matches the side
                if k_prime // 2 == side:
                    V_in[j0, k_prime, t] = V_out[j0, k_col, s]
                else:
                    V_in[j0, k_prime, t] = 0
        
        # W_in: k_prime, i0
        # W_out[k, i, s]
        for k_prime in range(4):
            k_col = k_prime % 2
            if k_prime // 2 == side:
                W_in[k_prime, 0, t] = W_out[k_col, 0, s]
                W_in[k_prime, 1, t] = W_out[k_col, 1, s]
            else:
                W_in[k_prime, 0, t] = 0
                W_in[k_prime, 1, t] = 0

    # Construct full U, V, W
    rank = 7 * 14
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i1 in range(2):
        for i0 in range(2):
            i = 2 * i1 + i0
            for s in range(7):
                for t_in in range(14):
                    col = s * 14 + t_in
                    U[i, col] = U_out[i1, s] * U_in[i0, t_in]
                    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for kp in range(4):
                    j = 2 * j1 + j0
                    k = 4 * k1 + kp
                    for s in range(7):
                        for t_in in range(14):
                            col = s * 14 + t_in
                            V[j * p + k, col] = V_out[j1, k1, s] * V_in[j0, kp, t_in]
                            
    for k1 in range(2):
        for kp in range(4):
            for i1 in range(2):
                for i0 in range(2):
                    k = 4 * k1 + kp
                    i = 2 * i1 + i0
                    for s in range(7):
                        for t_in in range(14):
                            col = s * 14 + t_in
                            W[k * n + i, col] = W_out[k1, i1, s] * W_in[kp, i0, t_in]
                            
    return {"U": U, "V": V, "W": W}
