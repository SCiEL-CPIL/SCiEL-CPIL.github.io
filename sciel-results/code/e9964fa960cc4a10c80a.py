"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        """Fast vectorized verification."""
        # Precompute indices
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # We can iterate over i, j, k and check the sum
        # To speed up, we can vectorize the inner product over r
        # But looping 96 times is fine.
        
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    # Element-wise product and sum
                    # prod = u_row * v_row * w_row
                    # if abs(np.sum(prod) - 1.0) > tol: return False
                    # Optimization: np.dot(u_row * v_row, w_row)
                    if abs(np.dot(u_row * v_row, w_row) - 1.0) > tol:
                        return False
        return True

    def reduce_rank(U, V, W, n, m, p, time_budget):
        """Greedy rank reduction using fast verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if time.time() > time_budget:
            return U_arr, V_arr, W_arr
        
        # Greedy removal
        # We try to remove columns one by one.
        # To make it efficient, we check all columns in a pass.
        # If a column can be removed, we remove it and restart the pass?
        # Or continue?
        # Restarting is safer to ensure we don't miss interactions.
        
        while time.time() < time_budget:
            removed_any = False
            current_rank = U_arr.shape[1]
            for col in range(current_rank):
                if time.time() > time_budget:
                    break
                
                # Mask out column col
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                
                U_test = U_arr[:, mask]
                V_test = V_arr[:, mask]
                W_test = W_arr[:, mask]
                
                if verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    removed_any = True
                    break # Restart the loop with new rank
            
            if not removed_any:
                break
        
        return U_arr, V_arr, W_arr

    # Build the initial rank-77 decomposition
    U, V, W = build_4x4x6()
    
    # Verify the initial decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if not verify_decomposition_fast(U_arr, V_arr, W_arr, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce the rank
    time_budget = deadline - 10
    U_reduced, V_reduced, W_reduced = reduce_rank(U_arr, V_arr, W_arr, n, m, p, time_budget)
    
    # Final verification
    if verify_decomposition_fast(U_reduced, V_reduced, W_reduced, n, m, p):
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
