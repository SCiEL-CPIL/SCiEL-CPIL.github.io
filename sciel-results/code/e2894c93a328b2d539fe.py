import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify_decomposition(U, V, W):
        """Verify tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1"""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using numpy"""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based fallback)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    def try_rank_structured(r, max_trials=100):
        """Try to find a rank-r decomposition using structured search."""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Strategy: Use structured patterns for V and W
            # V has 27 rows (3x9), W has 18 rows (9x2)
            # Try different sparsity patterns
            
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Pattern 1: Each column of V has non-zeros in specific row groups
            # Row groups for V: rows 0-8 (j=0), 9-17 (j=1), 18-26 (j=2)
            # Row groups for W: rows 0-1 (k=0), 2-3 (k=1), ..., 16-17 (k=8)
            
            for c in range(r):
                # Choose which row groups to activate
                # For V: pick 1-3 rows
                v_num_nonzero = np.random.choice([2, 3, 3, 4])
                v_indices = np.random.choice(m_rows, size=v_num_nonzero, replace=False)
                for idx in v_indices:
                    V[idx, c] = np.random.choice([-1, 1, -1, 1, 2, -2])
                
                # For W: pick 1-2 rows
                w_num_nonzero = np.random.choice([1, 2, 2, 3])
                w_indices = np.random.choice(p_rows, size=w_num_nonzero, replace=False)
                for idx in w_indices:
                    W[idx, c] = np.random.choice([-1, 1, -1, 1, 2, -2])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    # Solve M @ u = 1
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break
                    
                    u_int = np.round(u).astype(np.int64)
                    
                    # Check exactness
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        valid = False
                        break
                        
                    U[i*m+j, :] = u_int
                    
            if not valid:
                continue
            
            # Full verification
            if verify_fast(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        return None
    
    def try_rank_block_based(r, max_trials=50):
        """Try block-based structured search."""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Use block structure: group columns of B
            # Try to exploit symmetry
            
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Assign columns to blocks
            # Each block handles a subset of k values
            block_size = max(1, r // 9)  # roughly one block per k
            
            for block in range(9):
                start_c = block * block_size
                end_c = min((block + 1) * block_size, r)
                num_in_block = end_c - start_c
                
                if num_in_block <= 0:
                    continue
                    
                k = block
                
                # For this block, V should have non-zeros in rows related to k
                # V[j, k] for j=0,1,2 -> rows k, p+k, 2p+k
                v_rows = [k, p + k, 2*p + k]
                
                # W should have non-zeros in rows for output i=0,1
                w_rows = [k*n, k*n + 1]
                
                for c in range(start_c, end_c):
                    # V: pick 1-3 of the v_rows
                    num_v = np.random.choice([1, 2, 2, 3])
                    for _ in range(num_v):
                        row = v_rows[np.random.randint(0, 3)]
                        V[row, c] = np.random.choice([-1, 1, -1, 1])
                    
                    # W: pick 1-2 of the w_rows
                    num_w = np.random.choice([1, 2, 2])
                    for _ in range(num_w):
                        row = w_rows[np.random.randint(0, 2)]
                        W[row, c] = np.random.choice([-1, 1, -1, 1])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break
                    
                    u_int = np.round(u).astype(np.int64)
                    
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        valid = False
                        break
                        
                    U[i*m+j, :] = u_int
                    
            if not valid:
                continue
            
            if verify_fast(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        return None
    
    # Try to find lower rank solutions
    best_result = None
    
    # First try structured search from rank 49 down
    for r in range(49, 42, -1):
        if time.time() > deadline - 25:
            break
        
        # Try structured search
        res = try_rank_structured(r, max_trials=30)
        if res:
            best_result = res
            break
        
        # Try block-based search
        res = try_rank_block_based(r, max_trials=20)
        if res:
            best_result = res
            break
    
    if best_result:
        return best_result
    
    # Fallback to rank 50
    return current_solution()
