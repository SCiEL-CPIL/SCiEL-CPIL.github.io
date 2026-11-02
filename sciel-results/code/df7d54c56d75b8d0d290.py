import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145  # Safety margin (evaluator kills at 150s)

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # The known rank-11 decomposition
    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        # Term 1: (A00 + A11) * (B00 + B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # Term 2: (A10 + A11) * B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # Term 3: A00 * (B01 - B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # Term 4: A11 * (B10 - B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # Term 5: (A00 + A01) * B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # Term 6: (A10 - A00) * (B00 + B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # Term 7: (A01 - A11) * (B10 + B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # Term 8: A00 * B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # Term 9: A01 * B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # Term 10: A10 * B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # Term 11: A11 * B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol, n_trials=100):
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            M = np.einsum('ijr,ij->r', U3, A)
            N = np.einsum('jkr,jk->r', V3, B)
            MN = M * N
            C_computed = np.einsum('kir,r->ik', W3, MN)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    sol11 = make_rank11()
    if not verify_solution(sol11, n_trials=10):
        return sol11

    def try_rank_r(rank, max_restarts=200, max_iter=200):
        if time_remaining() < 20:
            return None
            
        best_error = float('inf')
        best_params = None
        
        n_vars = n*m*rank + m*p*rank + p*n*rank
        
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            T_target = np.ones((n, m, p))
            
            error = np.sum((T_computed - T_target)**2)
            return error

        def params_from_sol(sol, target_rank):
            U = np.array(sol["U"], dtype=float).reshape(n, m, -1)
            V = np.array(sol["V"], dtype=float).reshape(m, p, -1)
            W = np.array(sol["W"], dtype=float).reshape(p, n, -1)
            r = U.shape[2]
            if r >= target_rank:
                U_t = U[:, :, :target_rank]
                V_t = V[:, :, :target_rank]
                W_t = W[:, :, :target_rank]
            else:
                # Pad with zeros
                U_t = np.zeros((n, m, target_rank))
                V_t = np.zeros((m, p, target_rank))
                W_t = np.zeros((p, n, target_rank))
                U_t[:, :, :r] = U
                V_t[:, :, :r] = V
                W_t[:, :, :r] = W
            return np.concatenate([U_t.ravel(), V_t.ravel(), W_t.ravel()])

        # Strategy 1: Start from rank-11 truncated
        params0 = params_from_sol(sol11, rank)
        res = minimize(objective, params0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x.copy()
        
        if best_error < 1e-8:
            return best_params, best_error

        # Strategy 2: Many random restarts with varied scales
        n_restarts = max_restarts
        scales = [0.1, 0.2, 0.5, 1.0, 2.0]
        for i in range(n_restarts):
            if time_remaining() < 5:
                break
                
            scale = scales[i % len(scales)]
            params0 = np.random.randn(n_vars) * scale
            
            res = minimize(objective, params0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-8:
                break
        
        return best_params, best_error

    def finalize_solution(best_params, rank):
        U_flat = best_params[:n*m*rank]
        V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = best_params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Build solution
        U_list = [[0]*rank for _ in range(n*m)]
        V_list = [[0]*rank for _ in range(m*p)]
        W_list = [[0]*rank for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    U_list[i*m+j][r] = U[i,j,r]
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    V_list[j*p+k][r] = V[j,k,r]
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_list[k*n+i][r] = W[k,i,r]
                    
        sol = {"U": U_list, "V": V_list, "W": W_list}
        
        if verify_solution(sol, n_trials=50):
            return sol
        return None

    # Try rank 10
    sol10 = None
    if time_remaining() > 30:
        result10 = try_rank_r(10, max_restarts=150, max_iter=150)
        if result10 is not None:
            best_params, best_error = result10
            if best_error < 1e-6:
                sol10 = finalize_solution(best_params, 10)
    
    if sol10 is not None:
        return sol10
    
    # Try rank 9
    sol9 = None
    if time_remaining() > 30:
        result9 = try_rank_r(9, max_restarts=100, max_iter=100)
        if result9 is not None:
            best_params, best_error = result9
            if best_error < 1e-6:
                sol9 = finalize_solution(best_params, 9)
    
    if sol9 is not None:
        return sol9
    
    return sol11
