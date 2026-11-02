import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Precompute indices for the tensor
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Number of equations: n * m * p
    
    # Generate schoolbook solution as fallback
    U_sch = [[0] * rank_school for _ in range(n * m)]
    V_sch = [[0] * rank_school for _ in range(m * p)]
    W_sch = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][col] = 1
                V_sch[j * p + k][col] = 1
                W_sch[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    def check_validity(U, V, W, r):
        # U, V, W are numpy arrays of shape (rows, r)
        # Convert to 3D tensors for einsum
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W index is k*n+i, so W_3d[k, i, r]
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check if all entries are 1
        if not np.allclose(S, 1.0, atol=1e-9):
            return False
        return True

    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9

    def check_half_integers(U, V, W):
        for val in U.flatten():
            if not is_half_integer(val):
                return False
        for val in V.flatten():
            if not is_half_integer(val):
                return False
        for val in W.flatten():
            if not is_half_integer(val):
                return False
        return True

    import random
    random.seed(42)
    np.random.seed(42)
    
    # Search for lower rank
    # We prioritize finding rank 83, then 82, etc.
    # We use more restarts for lower ranks (closer to schoolbook) as they are more likely to exist?
    # Or maybe rank 83 is the target.
    
    for r in range(rank_school - 1, 70, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of restarts: more for higher r (closer to 84)
        # Rank 83: 50 restarts
        # Rank 82: 20 restarts
        # Rank 81: 10 restarts
        # ...
        restarts = max(5, 50 - (rank_school - 1 - r) * 5)
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            U_init = np.zeros((n * m, r))
            V_init = np.zeros((m * p, r))
            W_init = np.zeros((p * n, r))
            
            # Random sparse initialization
            # Each column has a few non-zeros
            for c in range(r):
                # Pick random indices
                i_u = random.randint(0, n * m - 1)
                i_v = random.randint(0, m * p - 1)
                i_w = random.randint(0, p * n - 1)
                
                U_init[i_u, c] = random.choice([1, -1, 0.5, -0.5])
                V_init[i_v, c] = random.choice([1, -1, 0.5, -0.5])
                W_init[i_w, c] = random.choice([1, -1, 0.5, -0.5])
                
                # Add a few more non-zeros to avoid trivial zeros
                if random.random() > 0.5:
                    i_u2 = random.randint(0, n * m - 1)
                    U_init[i_u2, c] = random.choice([1, -1, 0.5, -0.5])
                if random.random() > 0.5:
                    i_v2 = random.randint(0, m * p - 1)
                    V_init[i_v2, c] = random.choice([1, -1, 0.5, -0.5])
                if random.random() > 0.5:
                    i_w2 = random.randint(0, p * n - 1)
                    W_init[i_w2, c] = random.choice([1, -1, 0.5, -0.5])
            
            # Define residual function using einsum
            def residual(params):
                size_u = n * m * r
                size_v = m * p * r
                size_w = p * n * r
                
                U_flat = params[:size_u]
                V_flat = params[size_u:size_u + size_v]
                W_flat = params[size_u + size_v:]
                
                U_mat = U_flat.reshape(n * m, r)
                V_mat = V_flat.reshape(m * p, r)
                W_mat = W_flat.reshape(p * n, r)
                
                U_3d = U_mat.reshape(n, m, r)
                V_3d = V_mat.reshape(m, p, r)
                W_3d = W_mat.reshape(p, n, r)
                
                S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                return (S - 1.0).flatten()
            
            p0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual, p0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if sol.success or sol.cost < 1e-10:
                    size_u = n * m * r
                    size_v = m * p * r
                    size_w = p * n * r
                    
                    U_sol = sol.x[:size_u].reshape(n * m, r)
                    V_sol = sol.x[size_u:size_u + size_v].reshape(m * p, r)
                    W_sol = sol.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Round to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Check if rounded solution is exact
                    if check_validity(U_round, V_round, W_round, r):
                        if check_half_integers(U_round, V_round, W_round):
                            return {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                            
            except Exception as e:
                pass
                
    return fallback
