import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    
    # Tensor T
    # Shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def try_find_decomposition(rank):
        # Variables: U (nm x r), V (mp x r), W (pn x r)
        # Total vars: r * (nm + mp + pn)
        num_vars = rank * (n*m + m*p + p*n)
        
        def unpack(params):
            U = params[:rank*n*m].reshape(rank, n*m).T
            V = params[rank*n*m : rank*(n*m+m*p)].reshape(rank, m*p).T
            W = params[rank*(n*m+m*p):].reshape(rank, p*n).T
            return U, V, W

        def error(params):
            U, V, W = unpack(params)
            # Compute decomposition
            # T_decomp = sum_t outer(U[:,t], outer(V[:,t], W[:,t]))
            # Efficiently:
            # For each t, add U[:,t] @ V[:,t].T @ W[:,t].T? No.
            # T_decomp[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # We can compute this using einsum or loops.
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
            diff = T_decomp - T
            return np.sum(diff**2)

        # Random restarts
        best_err = float('inf')
        best_params = None
        start_time = time.time()
        max_time = 100 # seconds
        
        for restart in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with random small integers or half-integers
            # Scale: values usually in {-1, 0, 1}
            init_scale = 1.0
            params = np.random.uniform(-init_scale, init_scale, num_vars)
            
            # Use scipy minimize
            try:
                from scipy.optimize import minimize
                res = minimize(error, params, method='L-BFGS-B', options={'maxiter': 200})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
                
            if best_err < 1e-6:
                # Snap to half-integers
                U, V, W = unpack(best_params)
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
                if np.allclose(T_decomp, T, atol=1e-9):
                    return U, V, W
        
        return None

    # Try ranks from 19 to 23
    for r in range(19, 24):
        res = try_find_decomposition(r)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }
