import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # First, try to find a decomposition for 3x3x2
    n, m, p_small = 3, 3, 2
    
    # Target tensor for 3x3x2
    # Indices: i in 0..2, j in 0..2, k in 0..1
    # U index: i*m + j
    # V index: j*p_small + k
    # W index: k*n + i
    
    target = np.zeros((n*m, m*p_small, p_small*n))
    for i in range(n):
        for j in range(m):
            for k in range(p_small):
                u_idx = i * m + j
                v_idx = j * p_small + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    # Function to check if a set of matrices U, V, W decomposes the target
    def check_decomp(U, V, W, target_tensor, tol=1e-9):
        # Reconstruct tensor
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Here i is (i,j), j is (j,k), k is (k,i)
        # We can use einsum
        # U shape (nm, R), V shape (mp, R), W shape (pn, R)
        # T shape (nm, mp, pn)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T, target_tensor, atol=tol)
        
    # Function to try to find a decomposition of rank R for 3x3x2
    def find_decomp_332(R, max_restarts=10):
        # We will use a simple grid search or random search for small R
        # Since we expect small integer/half-integer solutions, we can search a bounded space
        
        # For 3x3x2, the rank is known to be 11 (Landsberg's bound says rank >= 11, and 11 is achievable)
        # Actually, for 3x3x3 the rank is 23. For 3x3x2, it's likely lower.
        # Let's try to find a solution with entries in {-1, -0.5, 0, 0.5, 1} or similar.
        
        # Instead of general optimization, let's try a specific known construction or a smarter search.
        # But first, let's improve the random search to be faster and more targeted.
        
        # If R is small, we can try to solve the linear system for fixed U, V?
        # No, it's a tensor decomposition, so it's nonlinear.
        
        # Let's stick to the optimization but make it faster.
        # We can use the fact that the solution is sparse or low-precision.
        
        # A better approach: Use the "Coppersmith-Winograd" style or "Landsberg" style construction?
        # For 3x3x2, a rank 11 construction exists.
        
        # Let's try to find a rank 11 solution.
        # We can initialize with random values and use L-BFGS-B or Powell.
        # But Nelder-Mead is slow.
        
        # Let's try a different initialization strategy:
        # Initialize U, V, W with small random integers or half-integers.
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p_small, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p_small*n, R))
            
            # Flatten variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p_small*R], x[n*m*R+m*p_small*R:]
                U_curr = U_flat.reshape(n*m, R)
                V_curr = V_flat.reshape(m*p_small, R)
                W_curr = W_flat.reshape(p_small*n, R)
                
                T = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                error = np.sum((T - target)**2)
                return error
            
            # Use Powell method which doesn't require gradients and is robust
            # But it's also slow.
            # Let's try to use a local search around the random initialization.
            
            # Actually, for such small problems, we can try to solve it exactly if we guess the structure.
            # But let's try a faster optimizer first.
            
            # We can use scipy.optimize.minimize with method='Powell'
            # But we need to import it.
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            except:
                return None
            
            if res.fun < 1e-8:
                # Found a solution
                U_flat, V_flat, W_flat = res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p_small*R], res.x[n*m*R+m*p_small*R:]
                U_sol = U_flat.reshape(n*m, R)
                V_sol = V_flat.reshape(m*p_small, R)
                W_sol = W_flat.reshape(p_small*n, R)
                
                # Try to snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if check_decomp(U_snap, V_snap, W_snap, target):
                    return U_snap, V_snap, W_snap
                else:
                    # Maybe the solution is exact but not half-integer?
                    if check_decomp(U_sol, V_sol, W_sol, target):
                         pass
                         
        return None

    # Try ranks for 3x3x2
    # The minimal rank for 3x3x2 is 11.
    U_332 = V_332 = W_332 = None
    for R in range(11, 14):
        if time.time() > deadline:
            break
        res = find_decomp_332(R, max_restarts=20)
        if res:
            U_332, V_332, W_332 = res
            break

    if U_332 is not None:
        # Construct 3x3x8 decomposition
        n, m, p = DIMENSIONS
        R_total = 4 * len(U_332[0]) # 4 * R
        
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        
        for b in range(4):
            offset = b * len(U_332[0])
            
            U[:, offset:offset+R] = U_332
            
            for j in range(m):
                for l in range(2):
                    k = 2 * b + l
                    V[j * p + k, offset:offset+R] = V_332[j * 2 + l, :]
            
            for l in range(2):
                for i in range(n):
                    k = 2 * b + l
                    W[k * n + i, offset:offset+R] = W_332[l * n + i, :]
                    
        # Verify
        target_338 = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_338[u_idx, v_idx, w_idx] = 1.0
        
        if check_decomp(U, V, W, target_338):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
