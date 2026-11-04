"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def compute_residual(U, V, W):
        """Compute the residual tensor for all (i,j,k) triples."""
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
                    residual[i, j, k] = val
        return residual
    
    def try_rank(rank, max_iters=10000, seed=42):
        """Try to find a valid decomposition of given rank using alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        target = np.ones((n, m, p))
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Alternating least squares: fix two, solve for third
            # Solve for U: for each i,j, minimize sum_k (U[i*m+j] * (V[j*p+k] * W[k*n+i].sum(axis=0)) - target[i,j,k])^2
            # This is: U[i*m+j] should be proportional to sum_k target[i,j,k] * V[j*p+k] * W[k*n+i]
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # Let A_r = V[j*p+k, r] * W[k*n+i, r] summed over k with weight 1
                    # Actually, for fixed i,j: we need sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r] * delta) = 1
                    # But it's trilinear, so we fix V and W, and for each k we have a linear constraint on U[i*m+j]
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each k
                    # This is a system of p equations in rank unknowns for U[i*m+j]
                    # Let's set up: for k in range(p): dot(U[i*m+j], V[j*p+k] * W[k*n+i]) = 1
                    # This is an overdetermined system if rank > p, underdetermined if rank < p
                    # Use least squares: minimize sum_k (dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1)^2
                    # Gradient: sum_k V[j*p+k]*W[k*n+i] * (dot(U[i*m+j], V[j*p+k]*W[k*n+i]) - 1) = 0
                    # So: U[i*m+j] = (sum_k V[j*p+k]*W[k*n+i]) / (sum_k (V[j*p+k]*W[k*n+i])^2) approximately
                    # More precisely, solve the linear system
                    
                    # Build the linear system: M @ u = b where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k] = V[j*p+k] * W[k*n+i]
                    b = np.ones(p)
                    
                    # Solve least squares
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j] = u_sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i] = U[i*m+j] * W[k*n+i]
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j*p+k] = v_sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j] = U[i*m+j] * V[j*p+k]
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i] = w_sol
                    except:
                        pass
            
            # Compute error
            res = compute_residual(U, V, W)
            err = np.sum((res - target) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
        
        if best_err < 1e-8:
            # Try to round to integers or half-integers
            for denom in [1, 2, 3, 4]:
                U_round = np.round(best_U * denom) / denom
                V_round = np.round(best_V * denom) / denom
                W_round = np.round(best_W * denom) / denom
                
                res = compute_residual(U_round, V_round, W_round)
                if np.allclose(res, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None
    
    # Try ranks in increasing order with multiple seeds
    for rank in range(18, 50):
        if time.time() > deadline - 20:
            break
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 10:
                break
            result = try_rank(rank, max_iters=5000, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook if we couldn't find better
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

# EVOLVE-BLOCK-END
