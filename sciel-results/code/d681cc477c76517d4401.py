import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Matrix indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # P1: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # P2: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # P3: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # P4: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # P5: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Strassen 4x4x4 block specifications
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],       # P0: A00 + A11
        [(1, 2), (1, 3)],       # P1: A10 + A11
        [(1, 0)],               # P2: A00
        [(1, 3)],               # P3: A11
        [(1, 0), (1, 1)],       # P4: A00 + A01
        [(1, 2), (-1, 0)],      # P5: A10 - A00
        [(1, 1), (-1, 3)],      # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],       # P0: B00 + B11
        [(1, 0)],               # P1: B00
        [(1, 1), (-1, 3)],      # P2: B01 - B11
        [(1, 2), (-1, 0)],      # P3: B10 - B00
        [(1, 3)],               # P4: B11
        [(1, 0), (1, 1)],       # P5: B00 + B01
        [(1, 2), (1, 3)],       # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],       # P0: C00 + C11
        [(1, 2), (-1, 3)],      # P1: C10 - C11
        [(1, 1), (1, 3)],       # P2: C01 + C11
        [(1, 0), (1, 2)],       # P3: C00 + C10
        [(-1, 0), (1, 1)],      # P4: -C00 + C01
        [(1, 3)],               # P5: C11
        [(1, 0)],               # P6: C00
    ]
    
    # Precompute mappings for efficiency
    # For each block index, precompute the global (row, col) offsets
    block_offsets = {}
    for bi in range(2):
        for bj in range(2):
            block_idx = bi * 2 + bj
            offsets = []
            for r in range(2):
                for c in range(2):
                    offsets.append((bi * 2 + r, bj * 2 + c))
            block_offsets[block_idx] = offsets
    
    for half in range(2):
        if time.time() > deadline:
            break
        b_col_offset = half * 4
        
        for k_p in range(7):
            for m_s in range(7):
                if time.time() > deadline:
                    break
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A: 4x4, flattened row-major) ---
                for coeff_a, block_idx in A_specs[k_p]:
                    for (gr, gc) in block_offsets[block_idx]:
                        a_eff_idx = (gr % 2) * 2 + (gc % 2)
                        val = U_2[a_eff_idx, m_s]
                        if val != 0:
                            U[gr * 4 + gc, col] += coeff_a * val
                
                # --- V (Input B: 4x8, flattened row-major) ---
                for coeff_b, block_idx in B_specs[k_p]:
                    for (gr, gc) in block_offsets[block_idx]:
                        b_eff_idx = (gr % 2) * 2 + (gc % 2)
                        val = V_2[b_eff_idx, m_s]
                        if val != 0:
                            global_col = gc + b_col_offset
                            V[gr * 8 + global_col, col] += coeff_b * val
                
                # --- W (Output C: 4x8, trace convention W[k*n+i, r]) ---
                for coeff_c, c_block_idx in C_specs[k_p]:
                    for (gi, gj) in block_offsets[c_block_idx]:
                        p_eff_idx = (gi % 2) * 2 + (gj % 2)
                        val = W_2[p_eff_idx, m_s]
                        if val != 0:
                            global_col = gj + b_col_offset
                            W[global_col * 4 + gi, col] += coeff_c * val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
