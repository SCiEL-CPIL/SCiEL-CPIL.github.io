"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T is of size (n*m) x (m*p) x (p*n) = 6 x 9 x 6.
    We seek rank-r decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    Returns dict with U (6 x r), V (9 x r), W (6 x r) where r is the rank.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    # Build the full tensor T[i,j,k]
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            j_a2 = j // p
            k_b = j % p
            if j_a != j_a2:
                continue
            for k in range(w_dim):
                k_b2 = k // n
                i_c = k % n
                if i_a != i_c or k_b != k_b2:
                    continue
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    
    # Schoolbook decomposition as baseline (rank 18)
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    def check_exact(U_r, V_r, W_r, tol=1e-6):
        """Check if the decomposition is exact."""
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.max(np.abs(recon - T)) < tol
    
    def objective_and_grad(params):
        """Compute objective and analytical gradient for tensor decomposition."""
        U_r = params[:u_dim * rank].reshape(u_dim, rank)
        V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
        W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
        
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        diff = recon - T
        obj = np.sum(diff * diff)
        
        # Gradient: d/dU[i,c] = 2 * sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
        dU = 2.0 * np.einsum('ijk,jc,kc->ic', diff, V_r, W_r)
        dV = 2.0 * np.einsum('ijk,ic,kc->jc', diff, U_r, W_r)
        dW = 2.0 * np.einsum('ijk,ic,jc->kc', diff, U_r, V_r)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return obj, grad
    
    def try_rank(rank, num_trials=100):
        """Try to find a rank-r decomposition using multiple optimization strategies."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            strategy = trial % 8
            
            if strategy == 0:
                # Pure random
                np.random.seed(1000 + trial * 7 + rank)
                U_init = np.random.randn(u_dim, rank)
                V_init = np.random.randn(v_dim, rank)
                W_init = np.random.randn(w_dim, rank)
            elif strategy == 1:
                # Sparse random (many zeros)
                np.random.seed(2000 + trial * 11 + rank)
                U_init = np.random.randn(u_dim, rank) * (np.random.rand(u_dim, rank) < 0.3)
                V_init = np.random.randn(v_dim, rank) * (np.random.rand(v_dim, rank) < 0.3)
                W_init = np.random.randn(w_dim, rank) * (np.random.rand(w_dim, rank) < 0.3)
            elif strategy == 2:
                # Based on partial schoolbook structure
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                for c in range(min(rank, 18)):
                    i_a = c // (m * p)
                    j_a = (c // p) % m
                    k_b = c % p
                    if i_a < n and j_a < m and k_b < p:
                        U_init[i_a * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_a, c] = 1.0
                # Add random perturbation
                np.random.seed(3000 + trial * 13 + rank)
                U_init += np.random.randn(u_dim, rank) * 0.1
                V_init += np.random.randn(v_dim, rank) * 0.1
                W_init += np.random.randn(w_dim, rank) * 0.1
            elif strategy == 3:
                # Block-structured: group terms by output row (exploit n=2)
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                # Assign roughly half terms to row 0, half to row 1
                terms_per_row = max(1, rank // n)
                for i_c in range(n):
                    for t in range(terms_per_row):
                        c = i_c * terms_per_row + t
                        if c >= rank:
                            break
                        np.random.seed(4000 + i_c * 17 + t * 23 + rank)
                        # U: only rows for output row i_c
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                        # V: all columns possible
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                        # W: only columns for output row i_c
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
            elif strategy == 4:
                # Winograd-like: try to exploit the structure
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                for i_c in range(n):
                    for t in range(rank // n):
                        c = i_c * (rank // n) + t
                        if c >= rank:
                            break
                        np.random.seed(5000 + i_c * 31 + t * 41 + rank)
                        for j_a in range(m):
                            U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                        for j_a in range(m):
                            for k_b in range(p):
                                V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                        for k_b in range(p):
                            W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
            elif strategy == 5:
                # Winograd-inspired: structured initialization exploiting 2x3x3 symmetry
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                np.random.seed(6000 + trial * 19 + rank)
                # Try to create terms that combine multiple products
                for c in range(rank):
                    # Alternate between output rows
                    i_c = c % n
                    # Distribute input columns
                    j_a = (c // n) % m
                    k_b = (c // (n * m)) % p
                    # Core contribution
                    U_init[i_c * m + j_a, c] = 1.0
                    V_init[j_a * p + k_b, c] = 1.0
                    W_init[k_b * n + i_c, c] = 1.0
                    # Add structured cross-terms (key for rank reduction)
                    # Each term can contribute to multiple output entries
                    if c % 2 == 0:
                        # Even terms: add contribution to other row's j_a
                        other_i = 1 - i_c
                        U_init[other_i * m + j_a, c] = np.random.choice([-1, 0, 1])
                        W_init[k_b * n + other_i, c] = np.random.choice([-1, 0, 1])
                    else:
                        # Odd terms: add contribution to adjacent k_b
                        k_b2 = (k_b + 1) % p
                        V_init[j_a * p + k_b2, c] = np.random.choice([-1, 0, 1])
                        W_init[k_b2 * n + i_c, c] = np.random.choice([-1, 0, 1])
            elif strategy == 6:
                # Symmetric initialization
                np.random.seed(7000 + trial * 23 + rank)
                U_init = np.random.randn(u_dim, rank)
                V_init = np.random.randn(v_dim, rank)
                W_init = np.random.randn(w_dim, rank)
                # Scale to have similar magnitudes
                U_init /= np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10
                V_init /= np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10
                W_init /= np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10
            else:
                # Hybrid: merge schoolbook terms pairwise to reduce rank
                U_init = np.zeros((u_dim, rank))
                V_init = np.zeros((v_dim, rank))
                W_init = np.zeros((w_dim, rank))
                np.random.seed(8000 + trial * 29 + rank)
                # Start with schoolbook-like structure but merge terms
                schoolbook_terms = []
                for i_a in range(n):
                    for j_a in range(m):
                        for k_b in range(p):
                            schoolbook_terms.append((i_a, j_a, k_b))
                # Shuffle and assign to rank slots, merging when possible
                np.random.shuffle(schoolbook_terms)
                for c in range(rank):
                    if c < len(schoolbook_terms):
                        i_a, j_a, k_b = schoolbook_terms[c]
                        U_init[i_a * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_a, c] = 1.0
                    # Add small random components to enable optimization
                    U_init[:, c] += np.random.randn(u_dim) * 0.2
                    V_init[:, c] += np.random.randn(v_dim) * 0.2
                    W_init[:, c] += np.random.randn(w_dim) * 0.2
            
            # Flatten parameters
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon.flatten() - T_flat
                return np.dot(diff, diff)
            
            def objective_grad(params):
                """Return (objective, gradient) for L-BFGS-B with analytical gradient."""
                return objective_and_grad(params)
            
            # Use multiple optimization methods
            if trial % 4 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
                jac = objective_grad
            elif trial % 4 == 1:
                method = 'Nelder-Mead'
                options = {'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-14}
                jac = None
            elif trial % 4 == 2:
                method = 'Powell'
                options = {'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-12}
                jac = None
            else:
                method = 'L-BFGS-B'
                options = {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
                jac = objective_grad
            
            try:
                from scipy.optimize import minimize
                if jac is not None:
                    result = minimize(objective, params_init, method=method, jac=jac, options=options)
                else:
                    result = minimize(objective, params_init, method=method, options=options)
            except Exception:
                continue
            
            if result.fun < 1e-8:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                if check_exact(U_r, V_r, W_r, tol=1e-5):
                    return U_r, V_r, W_r
                
                # Try to polish and round
                if result.fun < 1e-4:
                    # Attempt rounding to half-integers
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    if check_exact(U_round, V_round, W_round, tol=1e-5):
                        return U_round, V_round, W_round
                    
                    # Try integer rounding
                    U_int = np.round(U_r)
                    V_int = np.round(V_r)
                    W_int = np.round(W_r)
                    if check_exact(U_int, V_int, W_int, tol=1e-5):
                        return U_int, V_int, W_int
                    
                    # Final high-precision polish: re-optimize from rounded solution
                    if result.fun < 1e-3:
                        params_polish = np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()])
                        try:
                            result2 = minimize(objective, params_polish, method='L-BFGS-B', jac=objective_grad,
                                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
                            if result2.fun < 1e-10:
                                U_r2 = params_polish[:u_dim * rank].reshape(u_dim, rank)
                                V_r2 = params_polish[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                                W_r2 = params_polish[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                                if check_exact(U_r2, V_r2, W_r2, tol=1e-6):
                                    return U_r2, V_r2, W_r2
                        except Exception:
                            pass
        
        return None
    
    # Try ranks from 13 up to 17 (schoolbook is 18)
    # Focus on 13-15 first as these are more likely achievable
    candidate_ranks = [13, 14, 15, 16, 17]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Use more trials for smaller ranks where structure matters more
        num_trials = 120 if rank <= 15 else 60
        result = try_rank(rank, num_trials=num_trials)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break  # Found the lowest rank, stop
    
    # Convert to lists with integer/half-integer entries if possible
    def to_half_integer(mat):
        """Round values to nearest half-integer and verify."""
        rounded = np.round(mat * 2) / 2
        return rounded
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify the rounded version still works
    recon = np.einsum('ic,jc,kc->ijk', U_out, V_out, W_out)
    if np.max(np.abs(recon - T)) > 1e-4:
        # Rounding broke it, use original
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to Python lists
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
