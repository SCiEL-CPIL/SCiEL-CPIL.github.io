"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 150s limit, use 145s as hard stop with 5s margin
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Target tensor T
    # T[u, v, w] = 1 if u%3 == v//3 and v%3 == w//3 and w%3 == u//3
    target_flat = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_flat[u, v, w] = 1.0
    target_flat_list = target_flat.flatten()
    
    # Precompute for fast tensor computation
    def compute_tensor(U, V, W):
        """Fast vectorized tensor computation: T = sum_r U_r * V_r * W_r"""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Use einsum: 'ir,jr,kr->ijk'
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def error_func(params, rank):
        # Check deadline
        if time.time() > deadline:
            return 1e10
            
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        # Compute T = sum_r U_r * V_r * W_r (vectorized)
        T_calc = compute_tensor(U, V, W)
            
        diff = T_calc.flatten() - target_flat_list
        return np.dot(diff, diff)
    
    def compute_gradient(params, rank):
        """Compute gradient for faster convergence"""
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        diff_3d = diff.reshape((n * m, m * p, p * n))
        
        # Gradient: dT/dr = 2 * sum over i,j,k of diff[i,j,k] * (d/dU[i,r]) T_calc[i,j,k]
        # dT_calc[i,j,k]/dU[i',r'] = delta(i,i') * V[j,r'] * W[k,r']
        # So grad_U[i',r'] = 2 * sum_{j,k} diff[i',j,k] * V[j,r'] * W[k,r']
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff_3d, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff_3d, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff_3d, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def verify_and_round(U, V, W):
        # Check error
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_flat_list
        err = np.dot(diff, diff)
        
        if err < 1e-8:
            # Round to half-integers
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            # Verify rounded
            T_rounded = compute_tensor(U_r, V_r, W_r)
            diff_r = T_rounded.flatten() - target_flat_list
            err_r = np.dot(diff_r, diff_r)
            
            if err_r < 1e-8:
                return U_r.tolist(), V_r.tolist(), W_r.tolist(), True
        return None, None, None, False
    
    def try_polish(U, V, W, rank, max_iter=100):
        """High-precision polish step using Newton-like iterations"""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            T_calc = compute_tensor(U, V, W)
            diff = T_calc.flatten() - target_flat_list
            err = np.dot(diff, diff)
            
            if err < 1e-12:
                break
                
            # Compute gradient
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            grad = compute_gradient(params, rank)
            
            # Simple gradient descent with adaptive step
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
                
            step_size = min(0.1, 1e-3 / grad_norm)
            params -= step_size * grad
            
            U = params[:rank * n * m].reshape((n * m, rank))
            V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
            W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        return U, V, W

    # Schoolbook solution (Rank 27)
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
                
    best_result = {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

    # Try to find lower ranks
    # Rank 23 is the optimal known.
    # We try 23, 24, 25, 26.
    
    # Time allocation: prioritize rank 23
    time_per_rank = {
        23: 50.0,  # Most important
        24: 25.0,
        25: 15.0,
        26: 10.0
    }
    
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
            
        rank_budget = time_per_rank.get(rank, 15.0)
        rank_start = time.time()
        
        # Multiple random starts with diverse initialization
        trial = 0
        while time.time() - rank_start < rank_budget:
            if time.time() > deadline:
                break
                
            trial += 1
            
            # Different initialization strategies
            strategy = trial % 4
            
            if strategy == 0:
                # Random normal
                np.random.seed(trial * 1000 + rank * 7)
                init_U = np.random.normal(0, 1, (n * m, rank))
                init_V = np.random.normal(0, 1, (m * p, rank))
                init_W = np.random.normal(0, 1, (p * n, rank))
            elif strategy == 1:
                # Sparse initialization - random 0s and 1s
                np.random.seed(trial * 2000 + rank * 13)
                init_U = np.random.choice([0, 1], (n * m, rank)).astype(float)
                init_V = np.random.choice([0, 1], (m * p, rank)).astype(float)
                init_W = np.random.choice([0, 1], (p * n, rank)).astype(float)
                # Add small noise to avoid exact zeros
                init_U += np.random.normal(0, 0.1, (n * m, rank))
                init_V += np.random.normal(0, 0.1, (m * p, rank))
                init_W += np.random.normal(0, 0.1, (p * n, rank))
            elif strategy == 2:
                # Structured: start from schoolbook and add random perturbations
                # Take first 'rank' columns from schoolbook
                U_base = np.zeros((n * m, rank))
                V_base = np.zeros((m * p, rank))
                W_base = np.zeros((p * n, rank))
                for c in range(min(rank, 27)):
                    i, j, k = c // 9, (c // 3) % 3, c % 3
                    U_base[i * m + j, c] = 1
                    V_base[j * p + k, c] = 1
                    W_base[k * n + i, c] = 1
                np.random.seed(trial * 3000 + rank * 17)
                U_perturb = np.random.normal(0, 0.5, (n * m, rank))
                V_perturb = np.random.normal(0, 0.5, (m * p, rank))
                W_perturb = np.random.normal(0, 0.5, (p * n, rank))
                init_U = U_base + U_perturb
                init_V = V_base + V_perturb
                init_W = W_base + W_perturb
            else:
                # Random with scaling
                np.random.seed(trial * 4000 + rank * 23)
                scale = 1.0 / np.sqrt(rank)
                init_U = np.random.normal(0, scale, (n * m, rank))
                init_V = np.random.normal(0, scale, (m * p, rank))
                init_W = np.random.normal(0, scale, (p * n, rank))
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                # Use gradient-based optimization
                res = minimize(error_func, params, args=(rank,), 
                             jac=lambda p, r: compute_gradient(p, r),
                             method='L-BFGS-B', 
                             options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    U_final = res.x[:rank * n * m].reshape((n * m, rank))
                    V_final = res.x[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                    W_final = res.x[rank * n * m + rank * m * p :].reshape((p * n, rank))
                    
                    # Polish step
                    U_polished, V_polished, W_polished = try_polish(U_final, V_final, W_final, rank)
                    
                    U_r, V_r, W_r, ok = verify_and_round(U_polished, V_polished, W_polished)
                    if ok:
                        best_result = {
                            "U": U_r,
                            "V": V_r,
                            "W": W_r
                        }
                        return best_result
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END
