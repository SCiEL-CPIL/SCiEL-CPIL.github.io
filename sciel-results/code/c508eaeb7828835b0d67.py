"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 tensors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_strassen = np.zeros((4, 7), dtype=int)
    V_strassen = np.zeros((4, 7), dtype=int)
    W_strassen = np.zeros((4, 7), dtype=int)
    
    # P1: (A00+A11)(B00+B11) -> C00, C11
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # P2: (A10+A11)B00 -> C10, C11
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    
    # P3: A00(B10-B00) -> C10, C11
    U_strassen[0, 2] = 1
    V_strassen[2, 2] = 1; V_strassen[0, 2] = -1
    W_strassen[2, 2] = 1; W_strassen[3, 2] = 1
    
    # P4: A11(B01-B11) -> C01, C11
    U_strassen[3, 3] = 1
    V_strassen[1, 3] = 1; V_strassen[3, 3] = -1
    W_strassen[1, 3] = 1; W_strassen[3, 3] = 1
    
    # P5: (A00+A01)B11 -> C00, C01
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1
    
    # P6: (A10-A00)(B00+B01) -> C11
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1
    
    # P7: (A01-A11)(B10+B11) -> C00
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    rank_total = 0
    U = np.zeros((n*m, 42), dtype=int)
    V = np.zeros((m*p, 42), dtype=int)
    W = np.zeros((p*n, 42), dtype=int)
    
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset, rank_offset):
        """
        Add a 2x2x2 block to the main tensors.
        u_offset: (i_start, j_start) in the 2x4 A matrix? No, we need to map indices.
        The A matrix is 2x4. Indices (i, j) with i in 0..1, j in 0..3.
        The B matrix is 4x6. Indices (j, k) with j in 0..3, k in 0..5.
        The C matrix is 2x6. Indices (i, k) with i in 0..1, k in 0..5.
        
        For a 2x2 sub-block of A (rows i0..i0+1, cols j0..j0+1):
        Local indices (li, lj) map to global (i0+li, j0+lj).
        U index = (i0+li)*m + (j0+lj).
        
        For a 2x2 sub-block of B (rows j0..j0+1, cols k0..k0+1):
        Local indices (lj, lk) map to global (j0+lj, k0+lk).
        V index = (j0+lj)*p + (k0+lk).
        
        For a 2x2 sub-block of C (rows i0..i0+1, cols k0..k0+1):
        Local indices (li, lk) map to global (i0+li, k0+lk).
        W index = (k0+lk)*n + (i0+li).
        """
        # Map local 2x2 indices to global indices
        # U_block is 4x7, indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        for li in range(2):
            for lj in range(2):
                local_idx = li * 2 + lj
                global_i = u_offset[0] + li
                global_j = u_offset[1] + lj
                u_global_idx = global_i * m + global_j
                U[u_global_idx, rank_offset:rank_offset+7] += U_block[local_idx, :]
        
        for lj in range(2):
            for lk in range(2):
                local_idx = lj * 2 + lk
                global_j = v_offset[0] + lj
                global_k = v_offset[1] + lk
                v_global_idx = global_j * p + global_k
                V[v_global_idx, rank_offset:rank_offset+7] += V_block[local_idx, :]
        
        for li in range(2):
            for lk in range(2):
                local_idx = li * 2 + lk
                global_i = w_offset[0] + li
                global_k = w_offset[1] + lk
                w_global_idx = global_k * n + global_i
                W[w_global_idx, rank_offset:rank_offset+7] += W_block[local_idx, :]

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # A sub: rows 0-1, cols 0-1. u_offset=(0,0)
    # B sub: rows 0-1, cols 0-1. v_offset=(0,0)
    # C sub: rows 0-1, cols 0-1. w_offset=(0,0)
    add_block(U_strassen, V_strassen, W_strassen, (0,0), (0,0), (0,0), 0)
    rank_total += 7
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # A sub: rows 0-1, cols 2-3. u_offset=(0,2)
    # B sub: rows 2-3, cols 0-1. v_offset=(2,0)
    # C sub: rows 0-1, cols 0-1. w_offset=(0,0)
    add_block(U_strassen, V_strassen, W_strassen, (0,2), (2,0), (0,0), 7)
    rank_total += 7
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # A sub: rows 0-1, cols 0-1. u_offset=(0,0)
    # B sub: rows 0-1, cols 2-3. v_offset=(0,2)
    # C sub: rows 0-1, cols 2-3. w_offset=(0,2)
    add_block(U_strassen, V_strassen, W_strassen, (0,0), (0,2), (0,2), 14)
    rank_total += 7
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A sub: rows 0-1, cols 2-3. u_offset=(0,2)
    # B sub: rows 2-3, cols 2-3. v_offset=(2,2)
    # C sub: rows 0-1, cols 2-3. w_offset=(0,2)
    add_block(U_strassen, V_strassen, W_strassen, (0,2), (2,2), (0,2), 21)
    rank_total += 7
    
    # Now we have computed C[:, 0:4] using B[:, 0:4].
    # We still need C[:, 4:6].
    # B[:, 4:6] is 4x2.
    # We can split B[:, 4:6] into top 2x2 and bottom 2x2.
    
    # Block 5: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
    # A sub: rows 0-1, cols 0-1. u_offset=(0,0)
    # B sub: rows 0-1, cols 4-5. v_offset=(0,4)
    # C sub: rows 0-1, cols 4-5. w_offset=(0,4)
    add_block(U_strassen, V_strassen, W_strassen, (0,0), (0,4), (0,4), 28)
    rank_total += 7
    
    # Block 6: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
    # A sub: rows 0-1, cols 2-3. u_offset=(0,2)
    # B sub: rows 2-3, cols 4-5. v_offset=(2,4)
    # C sub: rows 0-1, cols 4-5. w_offset=(0,4)
    add_block(U_strassen, V_strassen, W_strassen, (0,2), (2,4), (0,4), 35)
    rank_total += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
