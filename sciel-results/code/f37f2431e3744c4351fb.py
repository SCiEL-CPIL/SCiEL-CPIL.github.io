"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Correct Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Rows of U correspond to A00, A01, A10, A11
    # Rows of V correspond to B00, B01, B10, B11
    # Rows of W correspond to C00, C01, C10, C11
    
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 0, 0, 1, 0,  0,-1],   # A11
    ], dtype=float)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1,  1, 0, 1, 0],   # B01
        [0, 0, 0,  1, 0, 0, 1],   # B10
        [1, 0,-1,  0, 1, 0, 1],   # B11
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, -1, 0, 1, 1],   # C00
        [0, 1, 1,  1, 1, 0, 1],   # C01
        [0, 1, 0,  0, 0, 0, 0],   # C10
        [1, 1,-1,  0,-1, 0, 0],   # C11
    ], dtype=float)
    
    # Verify Strassen block locally
    # T_local = sum_r U[:,r] V[:,r] W[:,r]
    # Check if T_local[i,j,k,i',j',k'] is 1 iff i==i', j==j', k==k'
    # Actually, for 2x2x2, the tensor is delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    U_loc = S_U
    V_loc = S_V
    W_loc = S_W
    
    valid_local = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ip in range(2):
                    for jp in range(2):
                        for kp in range(2):
                            lhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            rhs = 0.0
                            for r in range(7):
                                rhs += U_loc[i*2+j, r] * V_loc[jp*2+kp, r] * W_loc[kp*2+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                valid_local = False
                                break
                        if not valid_local: break
                    if not valid_local: break
                if not valid_local: break
            if not valid_local: break
        if not valid_local: break
    if not valid_local:
        # Fallback to schoolbook if local verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct rank-25 decomposition
    num_strassen_blocks = p // 2  # 3
    remaining_cols = p % 2        # 1
    
    rank = num_strassen_blocks * 7 + remaining_cols * n * m  # 3*7 + 1*4 = 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Process Strassen blocks
    for block in range(num_strassen_blocks):
        k_base = block * 2
        for r in range(7):
            # U part: same for all A entries, independent of k
            U[:, col] = S_U[:, r]
            
            # V part: B entries for this block
            # Local index 0 -> j=0, k=0 -> global j*p + k_base
            # Local index 1 -> j=0, k=1 -> global j*p + k_base + 1
            # Local index 2 -> j=1, k=0 -> global j*p + k_base
            # Local index 3 -> j=1, k=1 -> global j*p + k_base + 1
            
            for local_v_idx in range(4):
                j = local_v_idx // 2
                kk = local_v_idx % 2
                global_v_idx = j * p + (k_base + kk)
                V[global_v_idx, col] = S_V[local_v_idx, r]
                
            # W part: C entries for this block
            # Local index 0 -> k=0, i=0 -> global (k_base+0)*n + 0
            # Local index 1 -> k=1, i=0 -> global (k_base+1)*n + 0
            # Local index 2 -> k=0, i=1 -> global (k_base+0)*n + 1
            # Local index 3 -> k=1, i=1 -> global (k_base+1)*n + 1
            
            for local_w_idx in range(4):
                kk = local_w_idx // 2
                i = local_w_idx % 2
                global_w_idx = (k_base + kk) * n + i
                W[global_w_idx, col] = S_W[local_w_idx, r]
                
            col += 1
            
    # Process remaining column (schoolbook)
    for k in range(num_strassen_blocks * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify the full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = np.dot(U[i * m + j], V[jp * p + kp] * W[kp * n + ip])
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 28
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
