import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build target tensor T
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                T[a, b, w] = 1.0

    # Fallback Rank 42 (Strassen)
    def get_fallback():
        U = np.zeros((dim_A, 42), dtype=np.float64)
        V = np.zeros((dim_B, 42), dtype=np.float64)
        W = np.zeros((dim_C, 42), dtype=np.float64)
        
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

    U_fb, V_fb, W_fb = get_fallback()
    
    start_time = time.time()
    deadline = 110.0
    
    def objective(params, r):
        if time.time() > deadline:
            return 1e10
        U = params[:r*dim_A].reshape(dim_A, r)
        V = params[r*dim_A:r*dim_A + r*dim_B].reshape(dim_B, r)
        W = params[r*dim_A + r*dim_B:].reshape(dim_C, r)
        
        # T_hat = einsum
        T_hat = np.einsum('ak,bk,wk->abw', U, V, W)
        res = T - T_hat
        return np.sum(res**2)

    def try_rank(r, time_limit):
        end_time = time.time() + time_limit
        np.random.seed(42)
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > end_time:
                break
            params0 = np.random.randn(r * (dim_A + dim_B + dim_C))
            res = minimize(objective, params0, args=(r,), method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < 1e-6:
                U = res.x[:r*dim_A].reshape(dim_A, r)
                V = res.x[r*dim_A:r*dim_A + r*dim_B].reshape(dim_B, r)
                W = res.x[r*dim_A + r*dim_B:].reshape(dim_C, r)
                return U, V, W
        return None

    # Try rank 40
    if time.time() < deadline - 30:
        res_40 = try_rank(40, 25)
        if res_40:
            U, V, W = res_40
            return {"U": U, "V": V, "W": W}

    # Try rank 38
    if time.time() < deadline - 30:
        res_38 = try_rank(38, 25)
        if res_38:
            U, V, W = res_38
            return {"U": U, "V": V, "W": W}

    # Fallback to 42
    return {"U": U_fb, "V": V_fb, "W": W_fb}
