import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # 1. Define 2x2 Strassen Decomposition
    # T[i, j, k] = A[i, j] * B[j, k]
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We represent each rank-1 term as (u, v, w) where u, v, w are vectors of length 4 (flattened 2x2)
    # u corresponds to A, v to B, w to C.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3

    strassen_2x2 = []
    
    # M1
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    strassen_2x2.append((u, v, w))
    
    # M2
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0]
    strassen_2x2.append((u, v, w))
    
    # M3
    u = [1, 0, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 1, 0, 0]
    strassen_2x2.append((u, v, w))
    
    # M4
    u = [0, 0, 0, 1]
    v = [0, 0, 0, 1]
    w = [1, 0, 0, 1]
    strassen_2x2.append((u, v, w))
    
    # M5
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    strassen_2x2.append((u, v, w))
    
    # M6
    u = [0, 0, 1, -1]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    strassen_2x2.append((u, v, w))
    
    # M7
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    strassen_2x2.append((u, v, w))

    # 2. Construct 4x4 Core Decomposition (Rank 49)
    # The 4x4 matrix multiplication can be seen as 2x2 block multiplication.
    # Each block is 2x2.
    # The Strassen algorithm for 4x4 uses 7 block multiplications.
    # Each block multiplication is a 2x2 matrix multiplication, which has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # We need to map the 4x4 indices to block and element indices.
    # i = 2*ib + ie, j = 2*jb + je, k = 2*kb + ke
    # ib, jb, kb in {0, 1}, ie, je, ke in {0, 1}
    
    # The Strassen coefficients for the 4x4 case are the same as for 2x2, but applied to blocks.
    # M1_block = (A00_block + A11_block)(B00_block + B11_block)
    # etc.
    
    # We iterate over the 7 Strassen terms for the block level (r_block)
    # and the 7 Strassen terms for the element level (r_elem).
    
    U_core = np.zeros((16, 49))
    V_core = np.zeros((16, 49))
    W_core = np.zeros((16, 49))
    
    col = 0
    for r_block in range(7):
        u_b, v_b, w_b = strassen_2x2[r_block]
        for r_elem in range(7):
            u_e, v_e, w_e = strassen_2x2[r_elem]
            
            # u_b is a linear form on the 4 blocks of A.
            # u_b[ib, jb] is the coefficient for block A[ib, jb].
            # Each block A[ib, jb] is a 2x2 matrix with indices ie, je.
            # So the coefficient for A[2*ib+ie, 2*jb+je] is u_b[ib, jb] * u_e[ie, je].
            
            # Flatten indices:
            # A index: (2*ib + ie) * 4 + (2*jb + je) ? No, A is 4x4.
            # A[i, j] where i = 2*ib + ie, j = 2*jb + je.
            # Flattened A index: i * 4 + j.
            
            for ib in range(2):
                for jb in range(2):
                    if u_b[ib * 2 + jb] != 0: # u_b is flat [00, 01, 10, 11]
                        coeff_b = u_b[ib * 2 + jb]
                        for ie in range(2):
                            for je in range(2):
                                if u_e[ie * 2 + je] != 0:
                                    coeff_e = u_e[ie * 2 + je]
                                    i = 2 * ib + ie
                                    j = 2 * jb + je
                                    U_core[i * 4 + j, col] = coeff_b * coeff_e
                                    
            # Similarly for V and W
            for jb in range(2):
                for kb in range(2):
                    if v_b[jb * 2 + kb] != 0:
                        coeff_b = v_b[jb * 2 + kb]
                        for je in range(2):
                            for ke in range(2):
                                if v_e[je * 2 + ke] != 0:
                                    coeff_e = v_e[je * 2 + ke]
                                    j = 2 * jb + je
                                    k = 2 * kb + ke
                                    V_core[j * 4 + k, col] = coeff_b * coeff_e
                                    
            for kb in range(2):
                for ib in range(2):
                    if w_b[kb * 2 + ib] != 0:
                        coeff_b = w_b[kb * 2 + ib]
                        for ke in range(2):
                            for ie in range(2):
                                if w_e[ke * 2 + ie] != 0:
                                    coeff_e = w_e[ke * 2 + ie]
                                    k = 2 * kb + ke
                                    i = 2 * ib + ie
                                    W_core[k * 4 + i, col] = coeff_b * coeff_e
                                    
            col += 1
            
    # 3. Construct Tail 4x4x2 Decomposition (Rank 32)
    # A[0..3, 0..3] * B[0..3, 4..5] -> C[0..3, 4..5]
    # Schoolbook: Rank 4*4*2 = 32
    U_tail1 = np.zeros((16, 32))
    V_tail1 = np.zeros((16, 32)) # Only first 16 rows used? No, V is 30xR.
    # V index: j*6 + k. j in 0..3, k in 4,5.
    # W index: k*4 + i. k in 4,5, i in 0..3.
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_tail1[i * 4 + j, col] = 1
                V_tail1[j * 4 + (k-4), col] = 1 # Local index for tail
                # But we need to map to global V index.
                # We will assemble V later. Let's store local factors.
                col += 1
                
    # Let's create global V_tail1 and W_tail1
    V_tail1_global = np.zeros((30, 32))
    W_tail1_global = np.zeros((24, 32))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_tail1[i * 4 + j, col] = 1
                V_tail1_global[j * 6 + k, col] = 1
                W_tail1_global[k * 4 + i, col] = 1
                col += 1
                
    # 4. Construct Tail 4x1x6 Decomposition (Rank 24)
    # A[0..3, 4] * B[4, 0..5] -> C[0..3, 0..5]
    # Schoolbook: Rank 4*1*6 = 24
    U_tail2 = np.zeros((16, 24))
    V_tail2_global = np.zeros((30, 24))
    W_tail2_global = np.zeros((24, 24))
    
    col = 0
    for i in range(4):
        for k in range(6):
            U_tail2[i * 4 + 4, col] = 1 # j=4
            V_tail2_global[4 * 6 + k, col] = 1
            W_tail2_global[k * 4 + i, col] = 1
            col += 1

    # 5. Assemble Full Factors
    # Core: i in 0..3, j in 0..3, k in 0..3
    # U_core is 16x49. Global U is 20xR.
    # Rows 0..15 of U_core map to rows 0..15 of U_global?
    # U_global index i*5+j.
    # For core, i in 0..3, j in 0..3.
    # i*5+j = 5i+j.
    # U_core index i*4+j = 4i+j.
    # We need to map U_core rows to U_global rows.
    
    U_full = np.zeros((20, 49 + 32 + 24))
    V_full = np.zeros((30, 49 + 32 + 24))
    W_full = np.zeros((24, 49 + 32 + 24))
    
    # Core
    for i in range(4):
        for j in range(4):
            idx_u_global = i * 5 + j
            idx_u_core = i * 4 + j
            U_full[idx_u_global, 0:49] = U_core[idx_u_core, :]
            
            idx_v_global = j * 6 + k # Wait, k is not in U.
            # V_core index j*4+k.
            # We need to map V_core rows to V_full rows.
            # V_core is 16x49.
            # V_full is 30xR.
            # For core, j in 0..3, k in 0..3.
            # idx_v_global = j*6 + k.
            # idx_v_core = j*4 + k.
            
    # Let's do the mapping properly.
    U_full[:, 0:49] = 0
    V_full[:, 0:49] = 0
    W_full[:, 0:49] = 0
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # U
                idx_u_g = i * 5 + j
                idx_u_c = i * 4 + j
                # V
                idx_v_g = j * 6 + k
                idx_v_c = j * 4 + k
                # W
                idx_w_g = k * 4 + i
                idx_w_c = k * 4 + i
                
                # U_core[idx_u_c, :] contributes to U_full[idx_u_g, :]
                # But U_core is already summed over k? No, U_core is independent of k.
                # U_core[i, j, r] is the same for all k.
                # So we just copy the row.
                U_full[idx_u_g, 0:49] = U_core[idx_u_c, :]
                
                # V_core[j, k, r] is independent of i.
                # So we copy the row.
                V_full[idx_v_g, 0:49] = V_core[idx_v_c, :]
                
                # W_core[k, i, r] is independent of j.
                # So we copy the row.
                W_full[idx_w_g, 0:49] = W_core[idx_w_c, :]
                
    # Tail 1
    U_full[:, 49:49+32] = U_tail1
    V_full[:, 49:49+32] = V_tail1_global
    W_full[:, 49:49+32] = W_tail1_global
    
    # Tail 2
    U_full[:, 49+32:49+32+24] = U_tail2
    V_full[:, 49+32:49+32+24] = V_tail2_global
    W_full[:, 49+32:49+32+24] = W_tail2_global
    
    # Verify
    def verify(U, V, W):
        # Check nonzero positions
        for j in range(5):
            for i in range(4):
                for k in range(6):
                    a = i * 5 + j
                    b = j * 6 + k
                    c = k * 4 + i
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_full, V_full, W_full):
        # Fallback to schoolbook if construction failed
        U_sb = np.zeros((20, 120))
        V_sb = np.zeros((30, 120))
        W_sb = np.zeros((24, 120))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    U_sb[i * 5 + j, col] = 1
                    V_sb[j * 6 + k, col] = 1
                    W_sb[k * 4 + i, col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U_sb],
            "V": [[float(x) for x in row] for row in V_sb],
            "W": [[float(x) for x in row] for row in W_sb]
        }

    return {
        "U": [[float(x) for x in row] for row in U_full],
        "V": [[float(x) for x in row] for row in V_full],
        "W": [[float(x) for x in row] for row in W_full]
    }
# EVOLVE-BLOCK-END
