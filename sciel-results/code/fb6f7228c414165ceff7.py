"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1,  0, 0, 1, 0],   # B01
        [0, 0, 0,  1, 0, 0, 1],   # B10
        [1, 0,-1,  0, 1, 0, 1],   # B11
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0,  1, 0, 0],   # C01
        [0, 1, 0, 1,  0, 0, 0],   # C10
        [1,-1, 1, 0,  0, 1, 0],   # C11
    ]
    
    # Construct rank-25 decomposition
    # 3 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using Schoolbook (rank 4)
    
    num_strassen_blocks = p // 2  # 3
    remaining_cols = p % 2        # 1
    
    rank = num_strassen_blocks * 7 + remaining_cols * n * m  # 3*7 + 1*4 = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process Strassen blocks
    for block in range(num_strassen_blocks):
        k_base = block * 2
        for r in range(7):
            # U part: same for all A entries, independent of k
            for idx_u in range(n * m):
                U[idx_u][col] = S_U[idx_u][r]
                
            # V part: B entries for this block
            # B indices in this block: j*p + (k_base + kk) for j in [0,1], kk in [0,1]
            # S_V rows correspond to B00, B01, B10, B11 (local indices 0,1,2,3)
            # Local index 0 -> j=0, k=0 -> global j*p + k_base
            # Local index 1 -> j=0, k=1 -> global j*p + k_base + 1
            # Local index 2 -> j=1, k=0 -> global j*p + k_base
            # Local index 3 -> j=1, k=1 -> global j*p + k_base + 1
            
            for local_v_idx in range(4):
                j = local_v_idx // 2
                kk = local_v_idx % 2
                global_v_idx = j * p + (k_base + kk)
                V[global_v_idx][col] = S_V[local_v_idx][r]
                
            # W part: C entries for this block
            # C indices in this block: (k_base + kk)*n + i for kk in [0,1], i in [0,1]
            # S_W rows correspond to C00, C01, C10, C11 (local indices 0,1,2,3)
            # Local index 0 -> k=0, i=0 -> global (k_base+0)*n + 0
            # Local index 1 -> k=1, i=0 -> global (k_base+1)*n + 0
            # Local index 2 -> k=0, i=1 -> global (k_base+0)*n + 1
            # Local index 3 -> k=1, i=1 -> global (k_base+1)*n + 1
            
            for local_w_idx in range(4):
                kk = local_w_idx // 2
                i = local_w_idx % 2
                global_w_idx = (k_base + kk) * n + i
                W[global_w_idx][col] = S_W[local_w_idx][r]
                
            col += 1
            
    # Process remaining column (schoolbook)
    for k in range(num_strassen_blocks * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify the decomposition
    # T[i*m+j, jp*p+kp, kp*n+ip] = 1 iff i==ip and j==jp and k==kp
    # Sum_r U[i*m+j, r] V[jp*p+kp, r] W[kp*n+ip, r] should be 1 if i==ip, j==jp, k==kp else 0
    # Note: The tensor is T[a, b, c] where a=i*m+j, b=j'*p+k', c=k'*n+i'
    # Non-zero only if i==i', j==j', k==k'.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i * m + j][r] * V[jp * p + kp][r] * W[kp * n + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook rank 28
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
