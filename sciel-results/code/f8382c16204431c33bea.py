import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
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

    def try_structured_rank(r, max_trials=50):
        """Try to find rank-r solution using structured initialization"""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Strategy: Use the rank-50 solution as a guide for sparsity patterns
            # but try to compress by sharing terms more aggressively
            
            # Initialize with sparse patterns inspired by the known solution
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # For each column, assign sparse non-zero patterns
            for c in range(r):
                # V column: 3-5 non-zeros
                nz_v = random.randint(3, 5)
                v_indices = np.random.choice(m_rows, size=nz_v, replace=False)
                for idx in v_indices:
                    V[idx, c] = random.choice([-1, 1])
                
                # W column: 2-3 non-zeros
                nz_w = random.randint(2, 3)
                w_indices = np.random.choice(p_rows, size=nz_w, replace=False)
                for idx in w_indices:
                    W[idx, c] = random.choice([-1, 1])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    # Solve M @ u = 1 for integer u
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break
                    
                    u_int = np.round(u).astype(np.int64)
                    
                    # Check exactness with small tolerance
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        # Try rounding with different offsets
                        found = False
                        for delta in [-1, 0, 1]:
                            u_try = np.round(u + delta * 0.3).astype(np.int64)
                            if np.allclose(M @ u_try, np.ones(p), atol=1e-10):
                                u_int = u_try
                                found = True
                                break
                        if not found:
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

    def try_local_search_from_known():
        """Start from rank-50 and try to reduce by merging/eliminating terms"""
        if time.time() > deadline - 20:
            return None
        
        sol = current_solution()
        U = np.array(sol["U"], dtype=np.int64)
        V = np.array(sol["V"], dtype=np.int64)
        W = np.array(sol["W"], dtype=np.int64)
        rank = len(U[0])
        
        # Try to find pairs of columns that can be merged
        # Two columns c1, c2 can potentially be merged if their contributions
        # can be represented by a single column with adjusted values
        
        best_rank = rank
        best_sol = (U.copy(), V.copy(), W.copy())
        
        # Try removing each column and seeing if the system can still be satisfied
        for remove_col in range(rank):
            if time.time() > deadline - 15:
                break
            
            U_mod = np.delete(U, remove_col, axis=1)
            V_mod = np.delete(V, remove_col, axis=1)
            W_mod = np.delete(W, remove_col, axis=1)
            
            # Check if the remaining columns can be adjusted (small integer changes)
            # to still satisfy the equations
            # This is complex; skip for now as a heuristic
            
            pass
        
        return None

    # Try structured search for lower ranks with more trials
    best_result = None
    for r in range(49, 46, -1):
        if time.time() > deadline - 25:
            break
        res = try_structured_rank(r, max_trials=40)
        if res:
            best_result = res
            break
    
    if best_result:
        return best_result
    
    # Fallback to rank 50
    return current_solution()
