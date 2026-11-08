from scipy.optimize import minimize
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 135s to be safe under 150s limit

    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def compute_residual(U, V, W):
        rank = U.shape[1]
        resid = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    resid += (val - 1.0)**2
        return resid

    def rank54_decomp():
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                U[1*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                W[k1*n + 1, col] -= 1
                col += 1
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] -= 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] -= 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                U[0*m + j0, col] += 1
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] -= 1
                W[k1*n + 0, col] += 1
                col += 1
                U[1*m + j0, col] += 1
                U[0*m + j0, col] -= 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                U[0*m + j1, col] += 1
                U[1*m + j1, col] -= 1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = rank54_decomp()
    
    # Try to find lower rank solutions
    found_lower = False
    # Lower bound is 30. Try 30 to 53
    for target_rank in range(30, 54):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            try:
                rng = np.random.RandomState(42 + attempt * 7 + target_rank)
                U = rng.randn(n*m, target_rank)
                V = rng.randn(m*p, target_rank)
                W = rng.randn(p*n, target_rank)
                
                # Normalize to prevent overflow
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
                
                def objective(x):
                    U_flat, V_flat, W_flat = x[:n*m*target_rank], x[n*m*target_rank:n*m*target_rank + m*p*target_rank], x[n*m*target_rank + m*p*target_rank:]
                    U = U_flat.reshape(n*m, target_rank)
                    V = V_flat.reshape(m*p, target_rank)
                    W = W_flat.reshape(p*n, target_rank)
                    return compute_residual(U, V, W)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                # Use L-BFGS-B
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round):
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        found_lower = True
                        found = True
                        break
                    else:
                        # Try without rounding if verify fails? 
                        # The contract says "integer/half-integer entries".
                        # If rounding fails, the solution is invalid.
                        pass
            except Exception:
                continue
        if found_lower:
            break
            
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        best_result = rank54_decomp()
        
    return best_result
# EVOLVE-BLOCK-END
