import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def get_seed_solution():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank overflow")
            for idx, coeff in zip(u_indices, u_coeffs):
                U[idx][col] += coeff
            for idx, coeff in zip(v_indices, v_coeffs):
                V[idx][col] += coeff
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx][col] += coeff
            col += 1

        def add_strassen_block(k_start):
            a00, a01, a10, a11 = 0, 1, 3, 4
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [1], [b2k], [1], [c0k], [1])
            add_term([5], [1], [b2k], [1], [c1k], [1])
            
        return {"U": U, "V": V, "W": W}

    def build_tensor():
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    def verify_solution(U, V, W, T, tol=1e-9):
        R = len(U[0])
        # Check integer/half-integer
        for r in range(R):
            for i in range(n*m):
                val = U[i][r]
                if val != 0 and abs(val * 2 - round(val * 2)) > tol:
                    return False
            for j in range(m*p):
                val = V[j][r]
                if val != 0 and abs(val * 2 - round(val * 2)) > tol:
                    return False
            for k in range(p*n):
                val = W[k][r]
                if val != 0 and abs(val * 2 - round(val * 2)) > tol:
                    return False
        
        # Check residual
        residual = T.copy()
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
        if np.linalg.norm(residual) > tol:
            return False
        return True

    def try_rank(R, T, max_iters=50, time_limit=10):
        start_time = time.time()
        n_params = (n*m + m*p + p*n) * R
        x0 = np.random.randn(n_params) * 0.5
        
        def objective(x):
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Compute tensor
            T_calc = np.zeros((n, m, p))
            for r in range(R):
                # u_vec is (n, m), v_vec is (m, p), w_vec is (p, n)
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # Outer product structure: T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
                # This is not a simple outer product of vectors in standard sense, 
                # but we can compute it.
                for i in range(n):
                    for j in range(m):
                        uj = u[i, j]
                        if uj == 0: continue
                        for k in range(p):
                            T_calc[i, j, k] += uj * v[j, k] * w[k, i]
                            
            residual = T - T_calc
            return np.sum(residual**2)

        # We use a simple gradient descent or L-BFGS-B
        # L-BFGS-B might be too slow for many iterations if not careful.
        # Let's try a few random restarts with L-BFGS-B.
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            x0 = np.random.randn(n_params) * 0.5
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < 1e-6:
                # Found a low residual solution, try to snap to half-integers
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                # Convert to list of lists for verify_solution
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                if verify_solution(U_list, V_list, W_list, T):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
        return None

    sol = get_seed_solution()
    T = build_tensor()
    
    # Verification step
    if not verify_solution(sol["U"], sol["V"], sol["W"], T):
        raise ValueError("Seed solution failed verification")
    
    best_sol = sol
    best_rank = len(sol["U"][0])
    
    # Try to find lower rank
    # Start from 38 down to 35
    for R in range(best_rank - 1, 34, -1):
        if time.time() > 100: # Safety margin
            break
        candidate = try_rank(R, T, max_iters=20, time_limit=5)
        if candidate:
            best_sol = candidate
            best_rank = R
            # If we found a lower rank, we can stop or continue.
            # Continuing might find even lower, but diminishing returns.
            # Let's try to find the absolute minimum in the range.
            pass

    return best_sol
# EVOLVE-BLOCK-END
