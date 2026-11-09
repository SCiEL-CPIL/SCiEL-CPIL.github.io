"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    n, m, p = DIMENSIONS
    
    # Build target tensor T[u,v,w] where u=im+j, v=jp+k, w=kn+i
    target_flat = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_flat[u, v, w] = 1.0
    target_list = target_flat.flatten()
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def error_func(params, rank):
        if time.time() > deadline:
            return 1e10
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_list
        return np.dot(diff, diff)
    
    def compute_gradient(params, rank):
        U = params[:rank * n * m].reshape((n * m, rank))
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
        W = params[rank * n * m + rank * m * p :].reshape((p * n, rank))
        
        T_calc = compute_tensor(U, V, W)
        diff = (T_calc.flatten() - target_list).reshape((n * m, m * p, p * n))
        
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    def verify_solution(U, V, W, tol=1e-8):
        """Verify exact decomposition and round to rational values"""
        T_calc = compute_tensor(U, V, W)
        diff = T_calc.flatten() - target_list
        err = np.dot(diff, diff)
        
        if err > 1e-6:
            return None
        
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        T_r = compute_tensor(U_r, V_r, W_r)
        if np.dot((T_r.flatten() - target_list)**2) < 1e-8:
            return (U_r, V_r, W_r)
        
        # Try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        T_i = compute_tensor(U_i, V_i, W_i)
        if np.dot((T_i.flatten() - target_list)**2) < 1e-8:
            return (U_i, V_i, W_i)
        
        # If error is very small, return as-is (float representation)
        if err < 1e-10:
            return (U, V, W)
        
        return None
    
    # Schoolbook solution (Rank 27)
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
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Verify schoolbook
    T_s = compute_tensor(U_school, V_school, W_school)
    if np.dot((T_s.flatten() - target_list)**2) > 1e-10:
        return best_result
    
    # Strassen-inspired rank-23 construction
    # Based on the structure that 3x3 multiplication can be done with 23 multiplications
    # We'll try multiple structured initializations
    
    def strassen_init_23(seed_val):
        """Generate structured rank-23 initialization based on Strassen-like patterns"""
        np.random.seed(seed_val)
        rank = 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Pattern 1: Diagonal elements (3)
        for d in range(3):
            U[d * m + d, d] = 1
            V[d * p + d, d] = 1
            W[d * n + d, d] = 1
        
        # Pattern 2: Block Strassen-like for off-diagonal (18 patterns)
        # For each pair (i,j) with i != j, create patterns
        pair_idx = 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r = 3 + pair_idx
                pair_idx += 1
                
                # Standard Strassen-inspired: M1 = A(i,:)*B(:,j)
                # But we need to combine multiple such products
                U[i * m + j, r] = 1
                V[j * p + i, r] = 1
                W[i * n + j, r] = 1
                
                # Add companion patterns
                if pair_idx < 6:
                    k = (i + j) % 3
                    if k != i and k != j:
                        U[k * m + i, r] = 0.5
                        V[i * p + k, r] = 0.5
                        W[k * n + j, r] = 0.5
                        U[i * m + k, r] = 0.5
                        V[k * p + j, r] = 0.5
                        W[i * n + k, r] = 0.5
        
        # Pattern 3: Additional mixing patterns (2 more)
        for extra in range(2):
            r = 21 + extra
            # Mix across all indices
            for idx in range(9):
                ii, jj = idx // 3, idx % 3
                if (ii + jj + extra) % 3 == 0:
                    U[ii * m + jj, r] = 1.0 / 3.0
                    V[jj * p + ii, r] = 1.0 / 3.0
                    W[ii * n + jj, r] = 1.0 / 3.0
        
        # Add small random perturbation to break symmetry
        U += np.random.randn(*U.shape) * 0.01
        V += np.random.randn(*V.shape) * 0.01
        W += np.random.randn(*W.shape) * 0.01
        
        return U, V, W
    
    def try_optimization(U, V, W, rank, max_iter=500, ftol=1e-14, gtol=1e-12):
        """Run L-BFGS-B optimization and verify result"""
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = minimize(error_func, params, args=(rank,),
                         jac=lambda p, r: compute_gradient(p, r),
                         method='L-BFGS-B',
                         options={'maxiter': max_iter, 'maxfun': max_iter * 10,
                                  'ftol': ftol, 'gtol': gtol})
            if res.fun < 1e-8:
                U_f = res.x[:rank * n * m].reshape((n * m, rank))
                V_f = res.x[rank * n * m : rank * n * m + rank * m * p].reshape((m * p, rank))
                W_f = res.x[rank * n * m + rank * m * p :].reshape((p * n, rank))
                verified = verify_solution(U_f, V_f, W_f)
                if verified is not None:
                    return verified
        except Exception:
            pass
        return None
    
    # Try rank 23 with structured initializations
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
        
        rank_budget = 35.0 if rank == 23 else 20.0
        rank_start = time.time()
        
        # Multiple structured initializations
        for init_type in range(6):
            if time.time() - rank_start > rank_budget or time.time() > deadline - 5:
                break
            
            if rank == 23:
                U_init, V_init, W_init = strassen_init_23(init_type * 100 + 42)
            else:
                # For higher ranks, use perturbed schoolbook subset
                np.random.seed(init_type * 777 + rank)
                indices = np.random.choice(rank_school, rank, replace=False)
                U_init = U_school[:, indices] + np.random.randn(*U_school[:, indices].shape) * 0.05
                V_init = V_school[:, indices] + np.random.randn(*V_school[:, indices].shape) * 0.05
                W_init = W_school[:, indices] + np.random.randn(*W_school[:, indices].shape) * 0.05
            
            result = try_optimization(U_init, V_init, W_init, rank)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return best_result
        
        # If structured init didn't work, try a few random restarts
        remaining = min(rank_budget - (time.time() - rank_start), deadline - 5 - time.time())
        if remaining > 3:
            num_random = int(remaining / 3)
            for r_trial in range(max(1, num_random)):
                if time.time() > deadline - 3:
                    break
                np.random.seed(r_trial * 313 + rank * 999)
                U_rand = np.random.randn(n * m, rank) * 0.5
                V_rand = np.random.randn(m * p, rank) * 0.5
                W_rand = np.random.randn(p * n, rank) * 0.5
                result = try_optimization(U_rand, V_rand, W_rand, rank, max_iter=300)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
