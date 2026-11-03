import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Check time
    if time.time() > deadline - 10:
        return get_schoolbook()

    # Try to find a lower rank decomposition
    # We try rank 20
    target_rank = 20
    
    def try_rank(rank, max_seeds=5):
        if time.time() > deadline - 5:
            return None
            
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(seed)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            def residual(params):
                U_flat = params[:n*m*rank]
                V_flat = params[n*m*rank : n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank :]
                
                U_r = U_flat.reshape(n*m, rank)
                V_r = V_flat.reshape(m*p, rank)
                W_r = W_flat.reshape(p*n, rank)
                
                T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                res = T_rec - target
                return res.flatten()
            
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                sol = scipy.optimize.least_squares(residual, initial_params, method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if sol.cost < 1e-10:
                    U_opt = sol.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = sol.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = sol.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify
                    T_check = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    if np.allclose(T_check, target, atol=1e-9):
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            except Exception:
                continue
                
        return None

    result = try_rank(20)
    if result:
        return result
        
    # If rank 20 failed, try 21?
    if time.time() < deadline - 10:
        result = try_rank(21)
        if result:
            return result

    return get_schoolbook()
