import numpy as np
import scipy.optimize as so
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # Helper to build target tensor
    def build_tensor(n, m, p):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    T[a, b, w] = 1.0
        indices = np.argwhere(T == 1)
        return indices

    # Helper to verify solution
    def verify_solution(U, V, W, indices, atol=1e-9):
        idx_a = indices[:, 0]
        idx_b = indices[:, 1]
        idx_w = indices[:, 2]
        vals = np.einsum('ar,br,wr->ab', U[idx_a], V[idx_b], W[idx_w])
        return np.allclose(vals, 1.0, atol=atol)

    # Helper to try rank r
    def try_rank(n, m, p, r, time_budget, init_U=None, init_V=None, init_W=None):
        dim_A = n * m
        dim_B = m * p
        dim_C = p * n
        indices = build_tensor(n, m, p)
        idx_a = indices[:, 0]
        idx_b = indices[:, 1]
        idx_w = indices[:, 2]

        def res_func(t):
            U = t[:dim_A * r].reshape(dim_A, r)
            V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
            W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
            vals = np.einsum('ar,br,wr->ab', U[idx_a], V[idx_b], W[idx_w])
            return vals - 1.0

        start_time_local = time.time()
        deadline_local = start_time_local + time_budget
        
        # Initializations
        inits = []
        if init_U is not None:
            inits.append((init_U, init_V, init_W))
            
        # Random initializations
        for i in range(3):
            if time.time() > deadline_local:
                break
            np.random.seed(42 + i)
            U0 = np.random.randn(dim_A, r) * 0.5
            V0 = np.random.randn(dim_B, r) * 0.5
            W0 = np.random.randn(dim_C, r) * 0.5
            inits.append((U0, V0, W0))

        best_sol = None
        best_cost = float('inf')

        for U0, V0, W0 in inits:
            if time.time() > deadline_local:
                break
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                res = so.least_squares(res_func, theta0, method='trf', bounds=(-10, 10), 
                                       ftol=1e-12, xtol=1e-12, gtot=1e-12, max_nfev=2000)
                if res.cost < best_cost:
                    best_cost = res.cost
                    if res.cost < 1e-10:
                        best_sol = res.x
                        break
            except Exception:
                pass

        if best_sol is None:
            return None

        U_sol = best_sol[:dim_A * r].reshape(dim_A, r)
        V_sol = best_sol[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W_sol = best_sol[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        # Round to half-integers
        U_int = np.round(U_sol * 2) / 2
        V_int = np.round(V_sol * 2) / 2
        W_int = np.round(W_sol * 2) / 2
        
        if verify_solution(U_int, V_int, W_int, indices):
            return U_int, V_int, W_int
            
        return None

    # Fallback: rank-42 Strassen-based solution
    def get_fallback_solution():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W

    # Try to find a better solution than rank 42
    time_remaining = deadline - time.time()
    
    if time_remaining > 40:
        # Try rank 27 with structured initialization
        # Use the Strassen 2x2 factors repeated for each block
        U_init = np.zeros((8, 27))
        V_init = np.zeros((24, 27))
        W_init = np.zeros((12, 27))
        
        # Strassen 2x2 factors (7 multiplications)
        S_U = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, -1],
            [1, 1, 0, 1, 0, 0, 0],
        ])
        S_V = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, -1, 1, 0, 1],
            [0, 0, -1, 1, 0, 1, 1],
            [1, -1, 0, 0, 1, 0, 0],
        ])
        S_W = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
        ])
        
        for block in range(3):
            co = block * 9
            # Fill 7 Strassen columns + 2 extra zero columns
            U_init[:, co:co+7] = S_U
            v_maps = [[0,1,6,7], [2,3,8,9], [4,5,10,11]]
            w_maps = [[0,2,1,3], [4,6,5,7], [8,10,9,11]]
            for i in range(4):
                V_init[v_maps[block][i], co:co+7] = S_V[i]
                W_init[w_maps[block][i], co:co+7] = S_W[i]
        
        if time.time() < deadline - 25:
            sol = try_rank(2, 4, 6, 27, min(30, deadline - time.time()), U_init, V_init, W_init)
            if sol is not None:
                U, V, W = sol
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 42
    U, V, W = get_fallback_solution()
    return {"U": U, "V": V, "W": W}
