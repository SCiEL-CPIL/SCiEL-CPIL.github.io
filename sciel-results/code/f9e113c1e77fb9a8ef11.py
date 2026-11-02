import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Build the structured Strassen/Schoolbook decomposition (rank 39)
    terms = []
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def add_term(u_dict, v_dict, w_dict):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_list[idx] = val
        for idx, val in v_dict.items():
            v_list[idx] = val
        for idx, val in w_dict.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def strassen_block(col_offset):
        """7-term Strassen decomposition for 2x2x2 block at columns col_offset, col_offset+1"""
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1},
            {w_idx(col_offset,1): 1, w_idx(col_offset+1,1): -1}
        )
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,col_offset+1): 1, v_idx(1,col_offset+1): -1},
            {w_idx(col_offset+1,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,col_offset): 1, v_idx(0,col_offset): -1},
            {w_idx(col_offset,0): 1, w_idx(col_offset,1): 1}
        )
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): -1, w_idx(col_offset+1,0): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,col_offset): 1, v_idx(0,col_offset+1): 1},
            {w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1}
        )
    
    def schoolbook_2x1(col_k):
        """4-term schoolbook for 2x2x1 at column col_k"""
        add_term({u_idx(0,0): 1}, {v_idx(0,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(0,1): 1}, {v_idx(1,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(1,0): 1}, {v_idx(0,col_k): 1}, {w_idx(col_k,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,col_k): 1}, {w_idx(col_k,1): 1})
    
    # Decompose 2x2x7 as: Strassen(0,1) + Strassen(2,3) + Strassen(4,5) + Schoolbook(6)
    # That's 7 + 7 + 7 + 4 = 25 terms for the 2x2x7 part
    strassen_block(0)
    strassen_block(2)
    strassen_block(4)
    schoolbook_2x1(6)
    
    # 2x1x7 part: 14 terms
    for col_k in range(7):
        add_term({u_idx(0,2): 1}, {v_idx(2,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,col_k): 1}, {w_idx(col_k,1): 1})
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_list, v_list, w_list) in enumerate(terms):
        for r in range(n * m):
            U[r][t] = u_list[r]
        for r in range(m * p):
            V[r][t] = v_list[r]
        for r in range(p * n):
            W[r][t] = w_list[r]
    
    def check_decomposition(U_r, V_r, W_r, rank):
        U_arr = np.array(U_r, dtype=np.float64)
        V_arr = np.array(V_r, dtype=np.float64)
        W_arr = np.array(W_r, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    prod = U_arr[u_idx_val] * V_arr[v_idx_val] * W_arr[w_idx_val]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not check_decomposition(U, V, W, rank):
        # Fallback to schoolbook
        trivial_rank = n * m * p
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        rank = trivial_rank
    
    best_rank = rank
    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    
    # Try to improve using ALS on the full problem
    if time.time() < deadline - 20:
        n_sub, m_sub, p_sub = 2, 2, 7
        U_size = n_sub * m_sub  # 4
        V_size = m_sub * p_sub  # 14
        W_size = p_sub * n_sub  # 14
        
        def solve_subproblem(target_rank, max_iters=30):
            """Solve the 2x2x7 subproblem using ALS"""
            U_sub = np.random.randn(U_size, target_rank) * 0.5
            V_sub = np.random.randn(V_size, target_rank) * 0.5
            W_sub = np.random.randn(W_size, target_rank) * 0.5
            
            T_flat = np.ones(28)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U: fix V, W
                A_U = np.zeros((28, U_size * target_rank))
                b_U = np.ones(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = V_sub[v_idx_val, r] * W_sub[w_idx_val, r]
                                A_U[idx, u_idx_val * target_rank + r] = coeff
                            idx += 1
                U_sub, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_sub = U_sub.reshape(U_size, target_rank)
                
                # Update V: fix U, W
                A_V = np.zeros((28, V_size * target_rank))
                b_V = np.ones(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = U_sub[u_idx_val, r] * W_sub[w_idx_val, r]
                                A_V[idx, v_idx_val * target_rank + r] = coeff
                            idx += 1
                V_sub, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V_sub = V_sub.reshape(V_size, target_rank)
                
                # Update W: fix U, V
                A_W = np.zeros((28, W_size * target_rank))
                b_W = np.ones(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = U_sub[u_idx_val, r] * V_sub[v_idx_val, r]
                                A_W[idx, w_idx_val * target_rank + r] = coeff
                            idx += 1
                W_sub, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_sub = W_sub.reshape(W_size, target_rank)
                
                # Check error
                T_rec = np.zeros(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            u_idx_val = i * m_sub + j
                            v_idx_val = j * p_sub + k
                            w_idx_val = k * n_sub + i
                            T_rec[idx] = np.sum(U_sub[u_idx_val] * V_sub[v_idx_val] * W_sub[w_idx_val])
                            idx += 1
                
                error = np.linalg.norm(T_rec - T_flat)
                if error < 1e-9:
                    return U_sub, V_sub, W_sub
            
            return None
        
        # Try to find a better decomposition for the 2x2x7 part
        # Start from the structured solution and try to reduce rank
        for target_rank in range(24, 13, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            for _ in range(3):
                res = solve_subproblem(target_rank, max_iters=25)
                if res is not None:
                    U_sub, V_sub, W_sub = res
                    
                    new_rank = target_rank + 14
                    U_new = [[0] * new_rank for _ in range(n * m)]
                    V_new = [[0] * new_rank for _ in range(m * p)]
                    W_new = [[0] * new_rank for _ in range(p * n)]
                    
                    for r in range(target_rank):
                        for i in range(n_sub):
                            for j in range(m_sub):
                                U_new[i * m + j][r] = U_sub[i * m_sub + j, r]
                        for j in range(m_sub):
                            for k in range(p_sub):
                                V_new[j * p + k][r] = V_sub[j * p_sub + k, r]
                        for k in range(p_sub):
                            for i in range(n_sub):
                                W_new[k * n + i][r] = W_sub[k * n_sub + i, r]
                    
                    offset = target_rank
                    for col_k in range(7):
                        U_new[0 * m + 2][offset] = 1
                        V_new[2 * p + col_k][offset] = 1
                        W_new[col_k * n + 0][offset] = 1
                        offset += 1
                        U_new[1 * m + 2][offset] = 1
                        V_new[2 * p + col_k][offset] = 1
                        W_new[col_k * n + 1][offset] = 1
                        offset += 1
                    
                    if check_decomposition(U_new, V_new, W_new, new_rank):
                        best_rank = new_rank
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        found = True
                        break
            if found:
                break
                
    return best_result
