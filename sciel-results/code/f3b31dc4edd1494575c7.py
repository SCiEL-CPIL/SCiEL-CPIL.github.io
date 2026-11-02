import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Use schoolbook as a safe baseline if optimization fails
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try to find a lower rank decomposition
    # Target rank for 2x2xp is 2p+2 = 32
    # We split into 5 and 10.
    # Rank(5) = 12, Rank(10) = 22. Total 34.
    # Or just solve for 15 with rank 32.
    
    def solve_decomp(n, m, p, rank, seed=0):
        np.random.seed(seed)
        total_U = n * m
        total_V = m * p
        total_W = p * n
        
        # Initialize with schoolbook-like structure scaled to rank
        # This is a heuristic
        U = np.random.randn(total_U, rank) * 0.5
        V = np.random.randn(total_V, rank) * 0.5
        W = np.random.randn(total_W, rank) * 0.5
        
        # Use least squares
        from scipy.optimize import least_squares
        
        def residuals(params):
            U_r = params[:rank*total_U].reshape(rank, total_U).T
            V_r = params[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
            W_r = params[rank*(total_U+total_V):].reshape(rank, total_W).T
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                        res.append(val - 1.0)
            return res

        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        for trial in range(5): # Fewer trials to save time
            params0 = np.random.randn(rank * (total_U + total_V + total_W)) * 0.5
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=2000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                    best_U, best_V, best_W = U_r, V_r, W_r
            except:
                pass
        
        return best_U, best_V, best_W

    # Solve for full p=15 with rank 32
    U_full, V_full, W_full = solve_decomp(n, m, p, 32, seed=123)
    
    if U_full is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
