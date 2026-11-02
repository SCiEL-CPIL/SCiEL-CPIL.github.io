"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 110  # 110s deadline to be safe within 120s

    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p  # 54
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": [[float(x) for x in row] for row in U_schoolbook],
        "V": [[float(x) for x in row] for row in V_schoolbook],
        "W": [[float(x) for x in row] for row in W_schoolbook]
    }
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # For 3x3x6, try ranks from 53 down to 30
    # Actually, let's try a few specific lower ranks
    
    def build_tensor(rank, U, V, W):
        """Reconstruct tensor from decomposition"""
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T

    def target_tensor():
        """Build the target matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T_target = target_tensor()

    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form an exact decomposition (within tolerance)"""
        T = build_tensor(len(U[0]), U, V, W)
        return np.max(np.abs(T - T_target)) < tol

    def round_to_half(X):
        """Round entries to nearest multiple of 0.5"""
        return np.round(2 * X) / 2.0

    def try_rank(rank, max_iters=2000, tol=1e-8):
        """Try to find a rank-r decomposition using gradient descent"""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random matrices
        np.random.seed(42 + rank)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        def objective(params):
            if len(params) != (n * m + m * p + p * n) * rank:
                return 1e10
            U_p = params[:n * m * rank].reshape(n * m, rank)
            V_p = params[n * m * rank:(n * m + m * p) * rank].reshape(m * p, rank)
            W_p = params[(n * m + m * p) * rank:].reshape(p * n, rank)
            T = build_tensor(rank, U_p, V_p, W_p)
            return np.sum((T - T_target) ** 2)
        
        # Use scipy.optimize.minimize with L-BFGS-B
        from scipy.optimize import minimize
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol})
        except Exception:
            return None
        
        if time.time() > deadline - 5:
            return None
        
        if res.fun < 1e-6:
            U_r = res.x[:n * m * rank].reshape(n * m, rank)
            V_r = res.x[n * m * rank:(n * m + m * p) * rank].reshape(m * p, rank)
            W_r = res.x[(n * m + m * p) * rank:].reshape(p * n, rank)
            
            # Try rounding to half-integers
            U_rounded = round_to_half(U_r)
            V_rounded = round_to_half(V_r)
            W_rounded = round_to_half(W_r)
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return U_rounded, V_rounded, W_rounded
            
            # Try without rounding if entries are already close to integers/half-integers
            if verify_decomposition(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
        
        return None

    # Try ranks from 53 down to 35
    for rank in range(rank_schoolbook - 1, 34, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
            best_rank = rank
            break  # Found a valid lower rank, stop here

    return best_result
# EVOLVE-BLOCK-END
