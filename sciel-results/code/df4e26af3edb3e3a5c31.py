"""
Improved tensor rank minimization for 3x3 * 3x7 matrix multiplication.
Uses a structured approach with better initialization, multiple optimization strategies,
and a final polishing pass to snap to half-integers.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    hard_limit = 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63
    
    def time_left():
        return hard_limit - (time.time() - start_time)
    
    def compute_tensor(U, V, W, r):
        """Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]"""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return T
    
    def compute_tensor_and_grad(params, r):
        """
        Computes the tensor T and the gradient of the squared error ||T - 1||^2.
        params: flattened array [U_flat, V_flat, W_flat]
        """
        U = params[0:r*n*m].reshape(n * m, r)
        V = params[r*n*m:2*r*n*m].reshape(m * p, r)
        W = params[2*r*n*m:3*r*n*m].reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = T - 1.0
        obj = np.sum(err * err)
        
        # Gradients
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        grad_W_3d = np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
        
        grad = 2.0 * np.concatenate([
            grad_U_3d.flatten(), 
            grad_V_3d.flatten(), 
            grad_W_3d.flatten()
        ])
        
        return obj, grad
    
    def objective_only(params, r):
        obj, _ = compute_tensor_and_grad(params, r)
        return obj
    
    def verify_solution(U, V, W, r):
        """Check if the solution is exact (within tolerance)"""
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def snap_to_half_integers(U, V, W, r):
        """Try to snap to half-integer values and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True
        
        # Try snapping to integers
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        
        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True
        
        return U, V, W, False
    
    # Schoolbook Solution
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1
    
    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)
    best_obj = float('inf')
    
    # Known lower bound for 3x3 * 3x7 is around 35-40 based on literature
    # Let's try to find solutions below schoolbook
    min_rank_to_try = 30
    max_rank_to_try = rank_schoolbook - 1
    
    # Optimization strategies
    def optimize_with_lbfgs(U0, V0, W0, r, maxiter=1000):
        """Optimize using L-BFGS-B"""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(
                lambda x: objective_only(x, r),
                params0,
                jac=lambda x: compute_tensor_and_grad(x, r)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            params_opt = res.x
            U_opt = params_opt[0:r*n*m].reshape(n*m, r)
            V_opt = params_opt[r*n*m:2*r*n*m].reshape(m*p, r)
            W_opt = params_opt[2*r*n*m:3*r*n*m].reshape(p*n, r)
            
            obj_val = objective_only(params_opt, r)
            return U_opt, V_opt, W_opt, obj_val
            
        except Exception:
            return U0, V0, W0, float('inf')
    
    def optimize_with_nelder_mead(U0, V0, W0, r, maxiter=500):
        """Optimize using Nelder-Mead (no gradient needed, but slower)"""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(
                lambda x: objective_only(x, r),
                params0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            params_opt = res.x
            U_opt = params_opt[0:r*n*m].reshape(n*m, r)
            V_opt = params_opt[r*n*m:2*r*n*m].reshape(m*p, r)
            W_opt = params_opt[2*r*n*m:3*r*n*m].reshape(p*n, r)
            
            obj_val = objective_only(params_opt, r)
            return U_opt, V_opt, W_opt, obj_val
            
        except Exception:
            return U0, V0, W0, float('inf')
    
    # Try different ranks from high to low
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time_left() < 5:
            break
            
        found_valid = False
        num_restarts = 3 if r > 45 else 5
        
        for trial in range(num_restarts):
            if time_left() < 5:
                break
            
            random.seed(trial * 1000 + r * 7)
            np.random.seed(trial * 1000 + r * 7)
            
            # Different initialization strategies
            if trial == 0:
                # Random small values + bias
                U0 = np.random.randn(n*m, r) * 0.1 + 0.5
                V0 = np.random.randn(m*p, r) * 0.1 + 0.5
                W0 = np.random.randn(p*n, r) * 0.1 + 0.5
            elif trial == 1:
                # Sparse initialization
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                for i in range(n*m):
                    U0[i, random.randint(0, r-1)] = 1.0
                for j in range(m*p):
                    V0[j, random.randint(0, r-1)] = 1.0
                for k in range(p*n):
                    W0[k, random.randint(0, r-1)] = 1.0
            elif trial == 2:
                # Structured initialization based on schoolbook pattern
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = (i * m * p + j * p + k) % r
                            U0[i*m+j, c] += 1.0
                            V0[j*p+k, c] += 1.0
                            W0[k*n+i, c] += 1.0
            else:
                # Very small random values
                U0 = np.random.randn(n*m, r) * 0.01
                V0 = np.random.randn(m*p, r) * 0.01
                W0 = np.random.randn(p*n, r) * 0.01
            
            # Try L-BFGS first
            U_opt, V_opt, W_opt, obj_val = optimize_with_lbfgs(U0, V0, W0, r, maxiter=800)
            
            # Check if we found a very good solution
            if obj_val < 1e-6:
                U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(U_opt, V_opt, W_opt, r)
                
                if is_valid:
                    best_rank = r
                    best_solution = (U_snap, V_snap, W_snap)
                    found_valid = True
                    break
            elif obj_val < 1e-3:
                # If we're close but not quite there, try Nelder-Mead for polishing
                if time_left() > 10:
                    U_opt2, V_opt2, W_opt2, obj_val2 = optimize_with_nelder_mead(
                        U_opt, V_opt, W_opt, r, maxiter=300
                    )
                    if obj_val2 < 1e-6:
                        U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(
                            U_opt2, V_opt2, W_opt2, r
                        )
                        if is_valid:
                            best_rank = r
                            best_solution = (U_snap, V_snap, W_snap)
                            found_valid = True
                            break
            
            # If we haven't found a valid solution, update best if this is better
            if obj_val < best_obj:
                best_obj = obj_val
                # Don't update best_solution unless we have a valid one
        
        if found_valid:
            continue  # Try next lower rank
    
    U_res, V_res, W_res = best_solution
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
