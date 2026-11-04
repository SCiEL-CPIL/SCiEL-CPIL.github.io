import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check if entries are close to half-integers
        # This is a soft check, the problem requires exact half-integers
        # But for the search, we can allow floats and snap at the end
        
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_calc, Target, atol=tol):
            return False
        return True

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    def solve_for_W(U, V):
        """
        Given U and V, solve for W using least squares for each gamma.
        Returns W and the residual.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
        # We need to solve for W[gamma, r]
        # Equation: sum_r W[gamma, r] * M[alpha, beta, r] = Target[alpha, beta, gamma]
        
        W = np.zeros((dim_W, U_arr.shape[1]))
        residual = 0.0
        
        R = U_arr.shape[1]
        
        # Precompute M for all alpha, beta
        # M is (dim_U, dim_V, R)
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        for gamma in range(dim_W):
            # System: M[:, :, gamma]? No.
            # The coefficient for W[gamma, r] is M[alpha, beta, r].
            # We have dim_U * dim_V equations for R unknowns (W[gamma, 0..R-1]).
            
            # Matrix A is (dim_U * dim_V) x R
            A = M[:, :, :].reshape(dim_U * dim_V, R)
            # RHS b is (dim_U * dim_V)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            
            # Solve A x = b
            # Using lstsq
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            
            # Calculate residual for this gamma
            # r = b - A @ x
            r = b - A @ x
            residual += np.sum(r**2)
            
        return W, residual

    def objective(params):
        """
        Objective function for optimization.
        params: flattened U and V.
        """
        U_size = dim_U * R
        V_size = dim_V * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:]
        
        U = U_flat.reshape(dim_U, R)
        V = V_flat.reshape(dim_V, R)
        
        _, res = solve_for_W(U, V)
        return res

    def grad_objective(params):
        """
        Gradient of the objective function.
        """
        U_size = dim_U * R
        V_size = dim_V * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:]
        
        U = U_flat.reshape(dim_U, R)
        V = V_flat.reshape(dim_V, R)
        
        # Numerical gradient or analytical?
        # Analytical is complex. Numerical is slow.
        # Let's use a finite difference gradient with a small step.
        # Or better, use the chain rule.
        
        # Actually, let's just use scipy's minimize with 'BFGS' or 'L-BFGS-B' which computes gradients numerically or uses approximations.
        # But we can provide a gradient function to speed it up.
        
        # Let's stick to numerical gradient for simplicity, but it might be slow.
        # Alternative: Use 'Nelder-Mead' which doesn't need gradients? Slow.
        # Let's try 'L-BFGS-B' with numerical gradient first.
        
        pass # We'll rely on scipy's numerical gradient if we don't implement one.

    start_time = time.time()
    time_limit = 110.0 # Leave margin

    # Try ranks from a reasonable lower bound up to 75
    # Lower bound is 25. Let's start from 40 to save time, or 30.
    # We know 75 works.
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks
    # We will try to optimize U, V to find a W that is close to half-integers.
    
    for R in range(40, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Multiple restarts
        found = False
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V randomly
            # Use small integers
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            
            # Combine into params
            params_init = np.concatenate([U_init.flatten(), V_init.flatten()])
            
            # Optimize
            # Use L-BFGS-B
            # We need to minimize the residual.
            
            # To make it faster, we can check the residual at the initial point.
            W_check, res_check = solve_for_W(U_init, V_init)
            if res_check < 1e-9:
                if is_half_integer(W_check):
                    W_snap = snap_to_half_int(W_check)
                    if verify(U_init, V_init, W_snap):
                        best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        found = True
                        break
            
            # If not found, run optimization
            # Limit iterations to keep time low
            try:
                res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-9:
                params_opt = res.x
                U_opt = params_opt[:dim_U*R].reshape(dim_U, R)
                V_opt = params_opt[dim_U*R:].reshape(dim_V, R)
                
                W_opt, res_final = solve_for_W(U_opt, V_opt)
                
                if res_final < 1e-9:
                    # Check if W is close to half-integers
                    if is_half_integer(W_opt):
                        W_snap = snap_to_half_int(W_opt)
                        if verify(U_opt, V_opt, W_snap):
                            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_snap.tolist()}
                            best_rank = R
                            found = True
                            break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
