import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Helper to compute residual
    def compute_residual(U, V, W):
        res = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j_prime in range(m):
                        target = 1.0 if j == j_prime else 0.0
                        actual = 0.0
                        for c in range(U.shape[1]):
                            actual += U[i * m + j, c] * V[j_prime * p + k, c] * W[k * n + i, c]
                        res += (actual - target) ** 2
        return res

    # Vectorized residual for optimization
    def residual(params, r):
        if time.time() > deadline:
            return 1e10
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        # Vectorized computation
        # We need sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
        # This is a tensor contraction.
        # Let's create a 4D tensor T[i, j, j', k] = sum_c ...
        
        # Reshape U, V, W to broadcast?
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # We want T[i, j, j', k] = sum_c U_3d[i, j, c] * V_3d[j', k, c] * W_3d[k, i, c]
        # This is tricky to vectorize directly without huge intermediate arrays.
        # n=2, m=3, p=7, r~40.
        # U_3d: 2x3x40
        # V_3d: 3x7x40
        # W_3d: 7x2x40
        
        # Let's use einsum
        # T[i, j, j', k] = < U[i, j, :], V[j', k, :], W[k, i, :] >
        
        # This can be done as:
        # 1. Compute U^T * V? No.
        # 2. Let's just loop over c, it's fast enough for r=40, n*m*p=42.
        #    40 * 42 * 4 (from i,j,j',k loops) = 7000 ops. Very fast.
        
        T = np.zeros((n, m, m, p))
        for c in range(r):
            u_c = U_3d[:, :, c] # (n, m)
            v_c = V_3d[:, :, c] # (m, p)
            w_c = W_3d[:, :, c] # (p, n)
            
            # Term for this c:
            # T[i, j, j', k] += u_c[i, j] * v_c[j', k] * w_c[k, i]
            
            # Outer products
            # u_c[i, j] * v_c[j', k] -> (n, m, m, p)
            # This is (n, m) x (1, 1, m, p) -> broadcast?
            # u_c[:, :, None, None] * v_c[None, None, :, :] -> (n, m, m, p)
            term = u_c[:, :, None, None] * v_c[None, None, :, :] # (n, m, m, p)
            # Now multiply by w_c[k, i] which depends on k (axis 3) and i (axis 0)
            # w_c is (p, n). We need w_c[k, i] for each i, k.
            # So we need to multiply term[i, j, j', k] by w_c[k, i].
            # w_c can be reshaped to (n, 1, 1, p) ? No, w_c[k, i] means index k, i.
            # w_c.T is (n, p). So w_c.T[i, k].
            # We want to multiply term[i, j, j', k] by w_c.T[i, k].
            # w_c.T reshaped to (n, 1, 1, p).
            term *= w_c.T[:, None, None, :]
            
            T += term
        
        # Target tensor
        # T_target[i, j, j', k] = 1 if j == j' else 0. Independent of i, k?
        # Wait, the condition is j == j'.
        # So T_target is a 4D array where T_target[i, j, j', k] = 1 if j==j' else 0.
        
        # Construct target
        # j, j' range 0..m-1.
        # Create a matrix J (m, m) with 1 on diagonal, 0 elsewhere.
        J = np.eye(m)
        # T_target[i, j, j', k] = J[j, j']
        # So T_target = np.ones((n, 1, 1, p)) * J[None, :, :, None]
        # Let's verify dimensions:
        # J is (m, m).
        # J[None, :, :, None] is (1, m, m, 1).
        # np.ones((n, 1, 1, p)) * J[None, :, :, None] -> (n, m, m, p).
        # This matches T's shape.
        
        T_target = np.ones((n, 1, 1, p)) * J[None, :, :, None]
        
        res = np.sum((T - T_target) ** 2)
        return res

    # Try to find lower rank
    # Known lower bounds: max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # Upper bound: 42 (schoolbook).
    # Let's try ranks from 21 up to 42.
    # Optimization is expensive, so we try a few random restarts for each rank.
    
    # We prioritize finding a solution with integer/half-integer entries.
    # The optimization finds real solutions. We then try to round.
    
    # To speed up, we can use the fact that the problem is homogeneous?
    # If we scale U by s, V by 1/s, W by 1, the product is invariant.
    # We can fix a normalization, e.g., ||U||_F = 1.
    # But let's stick to the unconstrained optimization for now, it's simpler.
    
    # Let's try rank 31, 32, ... up to 42?
    # Or maybe just 41 and 42?
    # Actually, let's try a range.
    
    # To avoid getting stuck, we'll use a time budget.
    # We'll try ranks in increasing order.
    # If we find a solution, we stop.
    
    # Ranks to try
    ranks_to_try = list(range(21, 43))
    
    # Shuffle ranks? No, increasing is better to find minimal rank.
    # But maybe 42 is easier? No, schoolbook is 42.
    # Let's try 21, 22, ...
    
    # Optimization settings
    max_restarts_per_rank = 3
    max_iter = 100
    
    from scipy.optimize import minimize
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        found = False
        for restart in range(max_restarts_per_rank):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(restart * 1000 + r)
            U_init = np.random.randn(n*m, r)
            V_init = np.random.randn(m*p, r)
            W_init = np.random.randn(p*n, r)
            
            # Scale initialization to avoid large/small numbers
            # The residual is sum of squares of terms.
            # If U, V, W are ~N(0,1), terms are ~N(0, r).
            # Residual ~ r * 126.
            # Let's try to normalize.
            # Actually, BFGS should handle it.
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use L-BFGS-B for better performance?
                # Or BFGS.
                result = minimize(residual, params_init, args=(r,), method='BFGS', 
                                  options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-5})
                
                if result.fun < 1e-4: # Tolerance for "found"
                    U_sol = result.x[:n*m*r].reshape(n*m, r)
                    V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    
                    # Try to round to half-integers
                    # Scale factors?
                    # The solution is not unique due to scaling.
                    # We can try to find a scaling that makes entries close to half-integers.
                    
                    # Heuristic: 
                    # 1. Round to nearest 0.5
                    # 2. Check if it works.
                    # 3. If not, try to adjust scaling?
                    
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    res_round = compute_residual(U_round, V_round, W_round)
                    
                    if res_round < 1e-6:
                        best_rank = r
                        best_U = U_round.tolist()
                        best_V = V_round.tolist()
                        best_W = W_round.tolist()
                        found = True
                        break
                    
                    # If rounding failed, maybe the solution is valid but not half-integer?
                    # The task requires integer/half-integer entries.
                    # So we MUST find a half-integer solution.
                    
                    # What if we can scale the solution to make it half-integer?
                    # Let U' = a U, V' = b V, W' = c W.
                    # a*b*c = 1 (to keep product invariant? No, product is a*b*c * Product).
                    # We need a*b*c * Product(U,V,W) = Product(U,V,W).
                    # So a*b*c = 1.
                    # We want a*U, b*V, c*W to be close to half-integers.
                    
                    # This is a hard problem.
                    # Let's try a simple heuristic:
                    # Pick a such that max |a*U - round(2*a*U)/2| is minimized?
                    
                    # Or just try a few random scalings?
                    # Or try to make the largest entry an integer?
                    
                    # Let's try to find a scaling factor s such that s * U_sol is close to half-integers.
                    # We can try s = 1/val where val is a value from U_sol.
                    
                    # But this is getting complicated.
                    # Let's stick to the direct rounding for now.
                    # If it fails, we might need a better optimizer or different initialization.
                    
            except Exception:
                pass
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
