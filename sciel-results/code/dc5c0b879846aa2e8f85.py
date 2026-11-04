import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook decomposition generator
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Search for a decomposition of M_{n,m,p} with target rank R
    # Returns (U, V, W, rank) or None
    def search_decomposition(n, m, p, target_rank, time_limit=10.0):
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        # Target tensor entries
        # T[(i,j), (j,k), (k,i)] = 1
        # We store target values for non-zero entries
        # Index mapping:
        # u = i*m + j
        # v = j*p + k
        # w = k*n + i
        
        target_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_entries.append((u_idx, v_idx, w_idx, 1.0))
        
        def compute_error(U, V, W):
            err = 0.0
            for u_idx, v_idx, w_idx, val in target_entries:
                s = 0.0
                for r in range(target_rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                err += (s - val) ** 2
            # Check zero entries? 
            # For matrix multiplication tensor, if we only enforce the 1s, 
            # the 0s are automatically satisfied if the rank is low enough? 
            # No, we must ensure the 0s are 0.
            # However, the number of 0s is large.
            # Total entries in U*V*W space is (n*m)*(m*p)*(p*n).
            # For 3,3,2: 9*6*6 = 324.
            # Non-zeros: 18.
            # Zeros: 306.
            # Checking all is fast.
            
            # Iterate over all u, v, w
            for u in range(n * m):
                i_u = u // m
                j_u = u % m
                for v in range(m * p):
                    j_v = v // p
                    k_v = v % p
                    if j_u != j_v:
                        continue # Must be 0
                    for w in range(p * n):
                        k_w = w // n
                        i_w = w % n
                        if i_u != i_w or k_v != k_w:
                            target = 0.0
                        else:
                            target = 1.0
                            
                        s = 0.0
                        for r in range(target_rank):
                            s += U[u][r] * V[v][r] * W[w][r]
                        err += (s - target) ** 2
            return err

        # Random initialization with half-integers
        # Use small range to help convergence
        def init_random():
            U = np.random.randint(-2, 3, U_shape) * 0.5
            V = np.random.randint(-2, 3, V_shape) * 0.5
            W = np.random.randint(-2, 3, W_shape) * 0.5
            return U, V, W

        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try a few random restarts
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > start_time + time_limit:
                break
                
            U, V, W = init_random()
            current_error = compute_error(U, V, W)
            
            if current_error < 1e-9:
                return [[float(x) for x in row] for row in U], [[float(x) for x in row] for row in V], [[float(x) for x in row] for row in W], target_rank
            
            # Local search
            for step in range(1000):
                if time.time() > start_time + time_limit:
                    break
                    
                # Pick a random matrix and entry to perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                elif mat_idx == 1:
                    mat = V
                else:
                    mat = W
                    
                r_idx = random.randint(0, mat.shape[0] - 1)
                c_idx = random.randint(0, mat.shape[1] - 1)
                
                old_val = mat[r_idx][c_idx]
                # Perturb by +/- 0.5 or +/- 1
                delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                new_val = old_val + delta
                
                # Update
                mat[r_idx][c_idx] = new_val
                
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-9:
                        return [[float(x) for x in row] for row in U], [[float(x) for x in row] for row in V], [[float(x) for x in row] for row in W], target_rank
                else:
                    # Revert
                    mat[r_idx][c_idx] = old_val
                    
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        if best_error < 1e-9:
            return [[float(x) for x in row] for row in best_U], [[float(x) for x in row] for row in best_V], [[float(x) for x in row] for row in best_W], target_rank
            
        return None

    # Combine decompositions
    def combine_decomps(decomps):
        # decomps is a list of (U, V, W, rank)
        if not decomps:
            return None
            
        total_rank = sum(d[3] for d in decomps)
        
        # Determine shapes
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # We need to know n, m, p for the specific block
        # This function is generic, but U, V, W shapes must match.
        # For M_{3,3,3}, U is 9xR, V is 9xR, W is 9xR.
        # For M_{3,3,2}, U is 9xR, V is 6xR, W is 6xR.
        # They are not compatible for direct concatenation if dimensions differ!
        
        # Wait, the blocks are:
        # 1. M_{3,3,3}: U (9x23), V (9x23), W (9x23)
        # 2. M_{3,3,3}: U (9x23), V (9x23), W (9x23)
        # 3. M_{3,3,2}: U (9x17), V (6x17), W (6x17)
        
        # We cannot simply concatenate V and W because their row counts differ (9 vs 6).
        # The V matrix for the whole problem is 3*8 = 24 rows.
        # The W matrix for the whole problem is 8*3 = 24 rows.
        # The U matrix for the whole problem is 3*3 = 9 rows.
        
        # So we need to embed the block decompositions into the full space.
        
        # Let's define the full U, V, W as zeros.
        # Full U: 9 x Total_Rank
        # Full V: 24 x Total_Rank
        # Full W: 24 x Total_Rank
        
        # We need to map the block indices to the full indices.
        
        # This combination logic is complex to do generically in a helper.
        # I will hardcode the combination for the specific split: 3, 3, 2.
        
        return None

    # Hardcoded combination for 3, 3, 8 split into 3, 3, 2
    
    # 1. Get M_{3,3,3} decomposition
    # Try to find rank 23, else 24, else 25, else 27 (schoolbook)
    U_333_1, V_333_1, W_333_1, r_333_1 = None, None, None, 27
    for target_r in [23, 24, 25]:
        res = search_decomposition(3, 3, 3, target_r, time_limit=20.0)
        if res:
            U_333_1, V_333_1, W_333_1, r_333_1 = res
            break
    if not U_333_1:
        U_333_1, V_333_1, W_333_1, r_333_1 = schoolbook(3, 3, 3)
        
    # 2. Get second M_{3,3,3} decomposition (can reuse same if valid, but they are independent blocks)
    # Actually, since the k indices are disjoint, we can reuse the same U, V, W structure 
    # but shifted in the V and W indices.
    # However, the V indices for the first block are k=0,1,2.
    # The V indices for the second block are k=3,4,5.
    # The W indices for the first block are k=0,1,2.
    # The W indices for the second block are k=3,4,5.
    # The U indices are the same (i,j).
    
    # So we can use the same U, V, W matrices from the first decomposition,
    # but we need to place them in the correct columns of the full U, V, W.
    
    # 3. Get M_{3,3,2} decomposition
    U_332, V_332, W_332, r_332 = None, None, None, 18
    for target_r in [17, 18]:
        res = search_decomposition(3, 3, 2, target_r, time_limit=20.0)
        if res:
            U_332, V_332, W_332, r_332 = res
            break
    if not U_332:
        U_332, V_332, W_332, r_332 = schoolbook(3, 3, 2)

    # Combine
    total_rank = r_333_1 + r_333_1 + r_332
    
    U_full = [[0] * total_rank for _ in range(9)]
    V_full = [[0] * total_rank for _ in range(24)]
    W_full = [[0] * total_rank for _ in range(24)]
    
    col_offset = 0
    
    # Block 1: M_{3,3,3} (k=0,1,2)
    for i in range(3):
        for j in range(3):
            u_idx = i * 3 + j
            for r in range(r_333_1):
                U_full[u_idx][col_offset + r] = U_333_1[u_idx][r]
                
                # V index: j * 8 + k. Here k=0,1,2.
                for k in range(3):
                    v_idx = j * 8 + k
                    V_full[v_idx][col_offset + r] = V_333_1[j * 3 + k][r]
                    
                    # W index: k * 3 + i. Here k=0,1,2.
                    w_idx = k * 3 + i
                    W_full[w_idx][col_offset + r] = W_333_1[k * 3 + i][r]
    
    col_offset += r_333_1
    
    # Block 2: M_{3,3,3} (k=3,4,5)
    for i in range(3):
        for j in range(3):
            u_idx = i * 3 + j
            for r in range(r_333_1):
                U_full[u_idx][col_offset + r] = U_333_1[u_idx][r]
                
                # V index: j * 8 + k. Here k=3,4,5.
                for k in range(3, 6):
                    v_idx = j * 8 + k
                    # Map local k (0,1,2) to global k (3,4,5)
                    local_k = k - 3
                    V_full[v_idx][col_offset + r] = V_333_1[j * 3 + local_k][r]
                    
                    # W index: k * 3 + i. Here k=3,4,5.
                    w_idx = k * 3 + i
                    W_full[w_idx][col_offset + r] = W_333_1[local_k * 3 + i][r]
    
    col_offset += r_333_1
    
    # Block 3: M_{3,3,2} (k=6,7)
    for i in range(3):
        for j in range(3):
            u_idx = i * 3 + j
            for r in range(r_332):
                U_full[u_idx][col_offset + r] = U_332[u_idx][r]
                
                # V index: j * 8 + k. Here k=6,7.
                for k in range(6, 8):
                    v_idx = j * 8 + k
                    local_k = k - 6
                    V_full[v_idx][col_offset + r] = V_332[j * 2 + local_k][r]
                    
                    # W index: k * 3 + i. Here k=6,7.
                    w_idx = k * 3 + i
                    W_full[w_idx][col_offset + r] = W_332[local_k * 3 + i][r]
    
    col_offset += r_332
    
    # Verify
    error = 0
    for u in range(9):
        i_u = u // 3
        j_u = u % 3
        for v in range(24):
            j_v = v // 8
            k_v = v % 8
            if j_u != j_v:
                continue
            for w in range(24):
                k_w = w // 3
                i_w = w % 3
                if i_u == i_w and k_v == k_w:
                    target = 1.0
                else:
                    target = 0.0
                s_val = sum(U_full[u][r] * V_full[v][r] * W_full[w][r] for r in range(total_rank))
                error += (s_val - target) ** 2
                
    if error < 1e-9:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook if combination failed or search failed
    U, V, W, _ = schoolbook(3, 3, 8)
    return {"U": U, "V": V, "W": W}
