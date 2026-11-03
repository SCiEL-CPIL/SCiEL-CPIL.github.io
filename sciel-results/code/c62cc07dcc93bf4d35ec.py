import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_SEED_START = time.time()


def _deadline_guard(seconds_remaining=115.0):
    """Return True if we should stop searching."""
    return (time.time() - _SEED_START) > seconds_remaining


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank 56 decomposition via 8 Strassen blocks of 2x2x2.
    # The tensor is decomposed into 8 independent 2x2x2 sub-tensors.
    # Each 2x2x2 tensor has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n*m, rank)) # 8x56
    V = np.zeros((m*p, rank)) # 32x56
    W = np.zeros((p*n, rank)) # 16x56
    
    # Strassen's algorithm for 2x2 multiplication patterns.
    # Indices in local 2x2 blocks:
    # U (A part): row i, col j
    # V (B part): row j, col k
    # W (C part): row i, col k (but flattened as k*n+i in global W)
    
    # Patterns for U, V, W for the 7 Strassen terms.
    # Term 1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    # Term 2: (a10+a11)b00 -> c10 += 1, c11 -= 1
    # Term 3: a00(b01-b11) -> c01 += 1, c11 += 1
    # Term 4: a11(b10-b00) -> c00 += 1, c10 += 1
    # Term 5: (a00+a01)b11 -> c00 -= 1, c01 += 1
    # Term 6: (a10-a00)(b00+b10) -> c11 += 1
    # Term 7: (a01-a11)(b01+b11) -> c00 += 1
    
    # Let's verify the W patterns carefully.
    # c00 = t1 + t4 - t5 + t7
    # c01 = t3 + t5
    # c10 = t2 + t4
    # c11 = t1 - t2 + t3 + t6
    
    U_patterns = np.array([
        [[1, 0], [0, 1]],  # t1: a00+a11
        [[0, 0], [1, 1]],  # t2: a10+a11
        [[1, 0], [0, 0]],  # t3: a00
        [[0, 0], [0, 1]],  # t4: a11
        [[1, 1], [0, 0]],  # t5: a00+a01
        [[-1, 0], [1, 0]], # t6: a10-a00
        [[0, 1], [0, -1]]  # t7: a01-a11
    ])
    
    V_patterns = np.array([
        [[1, 0], [0, 1]],  # t1: b00+b11
        [[1, 0], [0, 0]],  # t2: b00
        [[0, 1], [0, -1]], # t3: b01-b11
        [[-1, 0], [1, 0]], # t4: b10-b00
        [[0, 0], [0, 1]],  # t5: b11
        [[1, 0], [1, 0]],  # t6: b00+b10
        [[0, 1], [0, 1]]   # t7: b01+b11
    ])
    
    W_patterns = np.array([
        [[1, 0], [0, 1]],  # t1: c00+1, c11+1
        [[0, 0], [1, -1]], # t2: c10+1, c11-1
        [[0, 1], [0, 1]],  # t3: c01+1, c11+1
        [[1, 0], [1, 0]],  # t4: c00+1, c10+1
        [[-1, 1], [0, 0]], # t5: c00-1, c01+1
        [[0, 0], [0, 1]],  # t6: c11+1
        [[1, 0], [0, 0]]   # t7: c00+1
    ])

    # There are 8 blocks.
    # b = 0..3: A0 (cols 0,1) * B0 (rows 0,1) sub-blocks (cols 0..7 of B)
    # b = 4..7: A1 (cols 2,3) * B1 (rows 2,3) sub-blocks (cols 0..7 of B)
    
    for b in range(8):
        a_block = b // 4  # 0 or 1
        b_subblock = b % 4  # 0, 1, 2, 3
        
        # Offsets in the global matrices
        # A is 2x4. a_block=0 -> cols 0,1. a_block=1 -> cols 2,3.
        a_col_offset = a_block * 2
        
        # B is 4x8. 
        # If a_block=0, we use B rows 0,1 (B0).
        # If a_block=1, we use B rows 2,3 (B1).
        b_row_offset = a_block * 2
        
        # b_subblock determines the column chunk of the output C and B.
        # b_subblock=0 -> cols 0,1
        # b_subblock=1 -> cols 2,3
        # b_subblock=2 -> cols 4,5
        # b_subblock=3 -> cols 6,7
        b_col_offset = b_subblock * 2
        
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            # Global index: i*m + j
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_col_offset + j), col] = U_patterns[r, i, j]
            
            # Fill V
            # Global index: j*p + k
            for j in range(2):
                for k in range(2):
                    V[(b_row_offset + j) * p + (b_col_offset + k), col] = V_patterns[r, j, k]
            
            # Fill W
            # Global index: k*n + i
            for i in range(2):
                for k in range(2):
                    W[(b_col_offset + k) * n + i, col] = W_patterns[r, i, k]

    # Validation using vectorized operations for speed and safety
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T to be all ones.
    
    # Create index arrays for validation
    # i in [0,2), j in [0,4), k in [0,8)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Gather rows
    U_gathered = U[u_idx, :] # shape (n*m*p, rank)
    V_gathered = V[v_idx, :] # shape (n*m*p, rank)
    W_gathered = W[w_idx, :] # shape (n*m*p, rank)
    
    # Compute sum of products
    # T_flat = sum_r U_gathered[:, r] * V_gathered[:, r] * W_gathered[:, r]
    T_flat = np.sum(U_gathered * V_gathered * W_gathered, axis=1)
    
    if not np.allclose(T_flat, 1.0, atol=1e-9):
        # Fallback to schoolbook if validation fails (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Convert to lists with proper integer types for clean output
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]

    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
