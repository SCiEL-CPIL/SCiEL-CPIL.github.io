"""Rewritten program for 5x5x5 matrix multiplication rank minimization."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices 0..4 into 3 groups: [0,1], [2,3], [4]
    # Group 0: size 2, indices 0, 1
    # Group 1: size 2, indices 2, 3
    # Group 2: size 1, index 4
    partitions = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    U = []
    V = []
    W = []
    
    def add_column(u_vals, v_vals, w_vals):
        # u_vals: dict {global_idx: coeff}
        # v_vals: dict {global_idx: coeff}
        # w_vals: dict {global_idx: coeff}
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for idx, val in u_vals.items():
            i, j = idx
            u_col[i * m + j] = val
        for idx, val in v_vals.items():
            j, k = idx
            v_col[j * p + k] = val
        for idx, val in w_vals.items():
            k, i = idx
            w_col[k * n + i] = val
            
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)

    # Precompute Strassen coefficients for 2x2x2
    # Local indices 0,1 for A, B, C
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Format: list of (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts {local_idx: val}
    strassen_blocks = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    # Iterate over the 27 sub-tensors
    for i_g in range(3):
        for j_g in range(3):
            for k_g in range(3):
                i_indices = partitions[i_g]
                j_indices = partitions[j_g]
                k_indices = partitions[k_g]
                
                n_sub = len(i_indices)
                m_sub = len(j_indices)
                p_sub = len(k_indices)
                
                # Determine decomposition type
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen
                    for u_l, v_l, w_l in strassen_blocks:
                        # Map local indices to global indices
                        u_g = {}
                        for (li, lj), val in u_l.items():
                            gi = i_indices[li]
                            gj = j_indices[lj]
                            u_g[(gi, gj)] = val
                            
                        v_g = {}
                        for (lj, lk), val in v_l.items():
                            gj = j_indices[lj]
                            gk = k_indices[lk]
                            v_g[(gj, gk)] = val
                            
                        w_g = {}
                        for (lk, li), val in w_l.items():
                            gk = k_indices[lk]
                            gi = i_indices[li]
                            w_g[(gk, gi)] = val
                            
                        add_column(u_g, v_g, w_g)
                else:
                    # Use Schoolbook for smaller blocks
                    # Rank is n_sub * m_sub * p_sub
                    for li in range(n_sub):
                        for lj in range(m_sub):
                            for lk in range(p_sub):
                                gi = i_indices[li]
                                gj = j_indices[lj]
                                gk = k_indices[lk]
                                
                                u_g = {(gi, gj): 1}
                                v_g = {(gj, gk): 1}
                                w_g = {(gk, gi): 1}
                                
                                add_column(u_g, v_g, w_g)
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
