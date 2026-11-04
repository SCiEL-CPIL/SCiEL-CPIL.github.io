import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Build target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = Fraction(1)
    
    start_time = time.time()
    deadline = 140.0
    
    # Try rank 17 first
    for target_rank in [17, 16, 15]:
        if time.time() > deadline:
            break
        
        # Strategy: start from rank 18 schoolbook, try to merge two terms
        # into one by finding a linear combination that cancels one degree of freedom
        
        # Schoolbook terms indexed by (i,j,k)
        # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        
        # To go from rank 18 to rank 17, we need to find coefficients such that
        # sum of 17 rank-1 tensors = T
        
        # Approach: parameterize 17 terms with free variables, solve polynomial system
        # Use Newton's method with rational arithmetic
        
        # Better approach: use the structure. Try specific patterns.
        
        # Let me try: take all 18 terms, but express one as a linear combination
        # of the others using the fact that the tensor has symmetries.
        
        # Actually, let me try a direct approach with numpy optimization
        # but with very careful initialization and verification.
        
        np.random.seed(42)
        
        best_err = float('inf')
        best_solution = None
        
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook 18 terms, drop one, add perturbation
            U_init = np.zeros((dim_u, target_rank))
            V_init = np.zeros((dim_v, target_rank))
            W_init = np.zeros((dim_w, target_rank))
            
            drop_col = restart % 18
            for t in range(target_rank):
                src = t if t < drop_col else t + 1
                i, j, k = divmod(src, m * p)
                k2 = k % p
                j2 = k // p
                i2 = i
                U_init[i2 * m + j2, t] = 1
                V_init[j2 * p + k2, t] = 1
                W_init[k2 * n + i2, t] = 1
            
            # Add small perturbation
            scale = 0.1
            U_init += np.random.randn(dim_u, target_rank) * scale
            V_init += np.random.randn(dim_v, target_rank) * scale
            W_init += np.random.randn(dim_w, target_rank) * scale
            
            # Use L-BFGS-B to minimize ||T - sum_t U[:,t] outer V[:,t] outer W[:,t]||^2
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(params):
                U = params[0:dim_u*target_rank].reshape(dim_u, target_rank)
                V = params[dim_u*target_rank:dim_u*target_rank+dim_v*target_rank].reshape(dim_v, target_rank)
                W = params[dim_u*target_rank+dim_v*target_rank:].reshape(dim_w, target_rank)
                return U, V, W
            
            def objective(params):
                U, V, W = unflatten_params(params)
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - T_rec
                return np.sum(E**2)
            
            def gradient(params):
                U, V, W = unflatten_params(params)
                T_rec = np.einsum('at,bt,ct->abc', U, V, W)
                E = T - T_rec
                grad_U = -2 * np.einsum('abc,bt,ct->at', E, V, W)
                grad_V = -2 * np.einsum('abc,at,ct->bt', E, U, W)
                grad_W = -2 * np.einsum('abc,at,bt->ct', E, U, V)
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            params_init = flatten_params(U_init, V_init, W_init)
            
            from scipy.optimize import minimize
            try:
                res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 100, 'gtol': 1e-14, 'ftol': 1e-20, 'disp': False})
                
                if res.fun < best_err:
                    best_err = res.fun
                    best_solution = res.x.copy()
                
                if best_err < 1e-12:
                    U_sol, V_sol, W_sol = unflatten_params(best_solution)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(2.0 * U_sol) / 2.0
                    V_rounded = np.round(2.0 * V_sol) / 2.0
                    W_rounded = np.round(2.0 * W_sol) / 2.0
                    
                    T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
                    # Try integer rounding
                    U_rounded = np.round(U_sol)
                    V_rounded = np.round(V_sol)
                    W_rounded = np.round(W_sol)
                    T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
            except Exception:
                pass
        
        # If optimization didn't find exact solution, try structured search
        # for this specific rank
        if target_rank == 17:
            # Try specific structured patterns for rank 17
            # Pattern: keep all terms except modify two to merge
            pass
    
    # Fall back to rank 18 schoolbook
    U_std = np.zeros((dim_u, 18))
    V_std = np.zeros((dim_v, 18))
    W_std = np.zeros((dim_w, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j, col] = 1
                V_std[j * p + k, col] = 1
                W_std[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_std.tolist(),
        "V": V_std.tolist(),
        "W": W_std.tolist()
    }
