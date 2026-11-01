import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[a, b, c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks from 15 to 17
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((dim_u, rank_fb))
    V_fb = np.zeros((dim_v, rank_fb))
    W_fb = np.zeros((dim_w, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_fb
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb

    # Helper to check exactness after rounding
    def check_exact(U, V, W):
        # U, V, W are integer/half-integer
        # Scale to integers
        U_int = np.round(U * 2).astype(int)
        V_int = np.round(V * 2).astype(int)
        W_int = np.round(W * 2).astype(int)
        
        # Reconstruct T
        T_recon = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        # The reconstruction is 4 * T_target if U,V,W are half-integers?
        # No, if U,V,W are half-integers, U*V*W is 1/8 * int.
        # We want sum U V W = T.
        # So 8 * sum U V W should be integer?
        # Let's just check if sum U V W == T exactly.
        # Using float for check might be risky, but we can use integer arithmetic on scaled values.
        # Let U = U_int / 2, etc.
        # Sum (U_int/2) (V_int/2) (W_int/2) = Sum U_int V_int W_int / 8
        # We need Sum U_int V_int W_int / 8 == T
        # So Sum U_int V_int W_int == 8 * T
        
        # T has 0s and 1s.
        # So we check if T_recon == 8 * T
        return np.allclose(T_recon, 8 * T, atol=0)

    def try_rank(r, max_iters=5000):
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        # Use small values to help convergence
        U = np.random.randn(dim_u, r) * 0.1
        V = np.random.randn(dim_v, r) * 0.1
        W = np.random.randn(dim_w, r) * 0.1
        
        # Flatten parameters
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_p = params[:dim_u*r].reshape(dim_u, r)
            V_p = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
            W_p = params[dim_u*r+dim_v*r:].reshape(dim_w, r)
            
            # Compute tensor product
            # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            
            # Error
            err = np.sum((T_calc - T)**2)
            return err
        
        # Optimize
        # Use L-BFGS-B or similar
        # We need to be careful with time
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Simple gradient descent or Nelder-Mead?
            # Nelder-Mead is robust but slow.
            # Let's use a few steps of gradient descent with random restarts.
            # Or just check if a known solution exists for r=15?
            
            # Actually, for r=15, the system is stiff.
            # Let's try to find a solution using integer linear programming? No, too complex.
            
            # Let's try a specific known decomposition if available.
            # For 2x3x3, is rank 15 known?
            # Yes, rank 15 is possible.
            # But we need the coefficients.
            
            # Let's try to solve the linear system for the entries of U,V,W given a fixed structure?
            # No, it's nonlinear.
            
            # Let's try a heuristic: 
            # Start with rank 18 and try to merge columns.
            # Or just return rank 18 if we can't find better quickly.
            
            # Given the time limit, finding a new rank 15 decomposition from scratch is risky.
            # However, we can try to verify if a random rank 15 solution can be found.
            
            # Let's try one optimization run with a tight budget.
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-8})
                if res.fun < 1e-8:
                    U_p = res.x[:dim_u*r].reshape(dim_u, r)
                    V_p = res.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                    W_p = res.x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                    
                    # Try to snap to half-integers
                    # Scale by 2
                    U_s = U_p * 2
                    V_s = V_p * 2
                    W_s = W_p * 2
                    
                    # Round
                    U_r = np.round(U_s)
                    V_r = np.round(V_s)
                    W_r = np.round(W_s)
                    
                    # Check exactness
                    if check_exact(U_r/2, V_r/2, W_r/2):
                        return U_r/2, V_r/2, W_r/2
            except Exception:
                pass
                
        return None

    # Try rank 15
    res = try_rank(15)
    if res:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try rank 16
    if time.time() < deadline - 10:
        res = try_rank(16)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 17
    if time.time() < deadline - 5:
        res = try_rank(17)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 18
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
