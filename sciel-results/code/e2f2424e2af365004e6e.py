import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = 4, 4, 8
    
    # Rank 98 decomposition using Block Strassen (Laderman)
    # 4x4x4 has rank 49 (7x7 Strassen blocks)
    # 4x4x8 is two 4x4x4 problems, so rank 98
    
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition coefficients
    # U_2, V_2, W_2 are 4x7 matrices (4 entries in, 7 products)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1: M2 = (a10+a11)b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2: M3 = a00(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3: M4 = a11(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4: M5 = (a00+a01)b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5: M6 = (a10-a00)(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6: M7 = (a01-a11)(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen coefficients for 4x4x4
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11
    # B is 4x4, split into 2x2 blocks B00, B01, B10, B11
    # C is 4x4, split into 2x2 blocks C00, C01, C10, C11
    
    # The 7 products P0..P6 are formed from linear combinations of A-blocks and B-blocks.
    # Then C-blocks are formed from linear combinations of P's.
    
    # A_specs: list of 7 lists. Each list contains (coeff, block_idx) for A.
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)]   # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)]    # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)]            # P6: C00
    ]
    
    # We have two independent 4x4x4 problems (for the two halves of the 4x8 matrix)
    # half = 0 corresponds to columns 0-3 of B and C
    # half = 1 corresponds to columns 4-7 of B and C
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                # A is 4x4. Index global_a_idx = r*4 + c
                # Block index bi, bj. bi, bj in {0,1}
                # Block (bi, bj) covers rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            # Effective index in 2x2 Strassen block
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # --- V (Input B) ---
                # B is 4x8. Index global_b_idx = r*8 + c
                # For a specific 'half', we only use columns 4*half .. 4*half+3
                # Block (bi, bj) covers rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1 (relative to half)
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            # Global column index in B
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # --- W (Output C) ---
                # C is 4x8. Index global_w_idx = k*4 + i (Trace convention W[k*n+i, r])
                # n=4. So W[k*4 + i, col]
                # Block (bi, bj) covers rows 2*bi..2*bi+1 (i) and cols 2*bj..2*bj+1 (k)
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2): # r corresponds to i (row of C)
                        for c in range(2): # c corresponds to k (col of C)
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
                
                if time.time() > deadline:
                    break
    
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
