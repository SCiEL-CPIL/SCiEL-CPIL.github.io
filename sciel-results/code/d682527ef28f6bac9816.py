import time
import random
import numpy as np
from scipy.optimize import least_squares, lsq_linear

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _snap_to_half_integers(U, V, W):
    """Snap arrays to nearest half-integers and verify."""
    U_s = (2 * U).round() / 2
    V_s = (2 * V).round() / 2
    W_s = (2 * W).round() / 2
    return U_s, V_s, W_s

def _compute_residual_norm(U, V, W, n, m, p, target_flat):
    """Compute residual norm efficiently."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    return np.linalg.norm(T.flatten() - target_flat)

def _precompute_target(n, m, p):
    """Precompute the target tensor and its flat version."""
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    return target, target.flatten()

def _solve_rank_alternating(n, m, p, R, target, target_flat, time_limit=10.0, num_restarts=12):
    """
    Alternating least squares with multiple restarts and structured initialization.
    """
    start_time = time.time()
    
    def solve_U_given_VW(V, W):
        """Solve for U given V and W using block-diagonal structure."""
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        U_new = np.zeros((n * m, R))
        
        # Precompute the coefficient matrix structure
        # For each (i,j), we need: sum_r U[i*m+j, r] * V[jp,k,r] * W[kp,ip,r] = target[i,j,jp,k,kp,ip]
        # The coefficient for U[i*m+j, r] in equation (i,j,jp,k,kp,ip) is V[jp,k,r]*W[kp,ip,r]
        
        # Build A for each (i,j) block - it's the same structure but different RHS
        # A[idx, r] where idx = ((jp*p+k)*p+kp)*n+ip, A[idx,r] = V[jp,k,r]*W[kp,ip,r]
        
        # This A is the same for all (i,j)! Only b changes.
        # Let's precompute A once.
        
        num_eqs_sub = m * p * p * n
        A_sub = np.zeros((num_eqs_sub, R))
        
        idx = 0
        for jp in range(m):
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        for r in range(R):
                            A_sub[idx, r] = V[jp * p + k, r] * W[kp * n + ip, r]
                        idx += 1
        
        # Now solve for each (i,j)
        for i in range(n):
            for j in range(m):
                b_sub = np.zeros(num_eqs_sub)
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                if (j == jp and k == kp and i == ip):
                                    b_sub[idx] = 1.0
                                idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    pass
        
        return U_new
    
    def solve_V_given_UW(U, W):
        """Solve for V given U and W."""
        U3 = U.reshape(n, m, R)
        W3 = W.reshape(p, n, R)
        
        V_new = np.zeros((m * p, R))
        
        num_eqs_sub = n * m * p * n
        A_sub = np.zeros((num_eqs_sub, R))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for kp in range(p):
                    for ip in range(n):
                        for r in range(R):
                            A_sub[idx, r] = U[i * m + j, r] * W[kp * n + ip, r]
                        idx += 1
        
        for jp in range(m):
            for k in range(p):
                b_sub = np.zeros(num_eqs_sub)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                if (j == jp and k == kp and i == ip):
                                    b_sub[idx] = 1.0
                                idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    V_new[jp * p + k, :] = sol
                except:
                    pass
        
        return V_new
    
    def solve_W_given_UV(U, V):
        """Solve for W given U and V."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        
        W_new = np.zeros((p * n, R))
        
        num_eqs_sub = n * m * m * p
        A_sub = np.zeros((num_eqs_sub, R))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for r in range(R):
                            A_sub[idx, r] = U[i * m + j, r] * V[jp * p + k, r]
                        idx += 1
        
        for kp in range(p):
            for ip in range(n):
                b_sub = np.zeros(num_eqs_sub)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                if (i == ip and j == jp and k == kp):
                                    b_sub[idx] = 1.0
                                idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    W_new[kp * n + ip, :] = sol
                except:
                    pass
        
        return W_new
    
    best_result = None
    best_norm = np.inf
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Structured: start from a perturbed schoolbook-like pattern
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
        elif restart == 1:
            # Sparse initialization: only a few non-zero entries per row
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            for _ in range(R * 3):
                U[random.randint(0, n*m-1), random.randint(0, R-1)] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                V[random.randint(0, m*p-1), random.randint(0, R-1)] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                W[random.randint(0, p*n-1), random.randint(0, R-1)] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            # Ensure no all-zero columns
            for r in range(R):
                if np.allclose(U[:, r], 0):
                    U[random.randint(0, n*m-1), r] = random.uniform(0.5, 1.5)
                if np.allclose(V[:, r], 0):
                    V[random.randint(0, m*p-1), r] = random.uniform(0.5, 1.5)
                if np.allclose(W[:, r], 0):
                    W[random.randint(0, p*n-1), r] = random.uniform(0.5, 1.5)
        elif restart == 2:
            # Binary initialization
            U = np.random.randint(0, 2, (n * m, R)).astype(float) - 0.5
            V = np.random.randint(0, 2, (m * p, R)).astype(float) - 0.5
            W = np.random.randint(0, 2, (p * n, R)).astype(float) - 0.5
            U *= random.uniform(0.5, 1.0)
            V *= random.uniform(0.5, 1.0)
            W *= random.uniform(0.5, 1.0)
        elif restart == 3:
            # Half-integer initialization
            U = np.random.randint(0, 4, (n * m, R)).astype(float) / 2 - 0.75
            V = np.random.randint(0, 4, (m * p, R)).astype(float) / 2 - 0.75
            W = np.random.randint(0, 4, (p * n, R)).astype(float) / 2 - 0.75
        else:
            # Random with varying scale
            scale = random.uniform(0.2, 1.5)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
        
        # Run alternating minimization
        for iteration in range(80):
            if time.time() - start_time > time_limit:
                break
            
            U = solve_U_given_VW(V, W)
            V = solve_V_given_UW(U, W)
            W = solve_W_given_UV(U, V)
            
            # Check convergence
            res_norm = _compute_residual_norm(U, V, W, n, m, p, target_flat)
            
            if res_norm < best_norm:
                best_norm = res_norm
                best_result = (U.copy(), V.copy(), W.copy())
            
            if res_norm < 1e-8:
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if _verify_fast(U_s, V_s, W_s, n, m, p):
                    return U_s, V_s, W_s
            
            if res_norm < 1e-12:
                break
    
    # Check best result
    if best_result is not None and best_norm < 1e-6:
        U, V, W = best_result
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
        U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
        if _verify_fast(U_s, V_s, W_s, n, m, p):
            return U_s, V_s, W_s
    
    return None

def _solve_rank_greedy_peel(n, m, p, R, target, target_flat, time_limit=10.0):
    """
    Greedy rank-1 peeling: iteratively find the best rank-1 approximation
    and subtract it, R times.
    """
    start_time = time.time()
    
    # The tensor T[i,j,jp,k,kp,ip] = delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    # A rank-1 term is u[i,j] * v[jp,k] * w[kp,ip]
    # We want to find u,v,w that best approximate the remaining tensor.
    
    # For a rank-1 term, the contribution to T[i,j,jp,k,kp,ip] is u[i,j]*v[jp,k]*w[kp,ip]
    # This is a separable tensor. We can find the best rank-1 approximation
    # by unfolding and using SVD.
    
    # Unfold the tensor into a matrix:
    # T_mat[ (i,j,jp,k), (kp,ip) ] = T[i,j,jp,k,kp,ip]
    # or T_mat[ (i,j), (jp,k,kp,ip) ] etc.
    
    # Let's use the unfolding: M[(i,j), (jp,k,kp,ip)]
    # Shape: (n*m, m*p*p*n)
    
    # Actually, for rank-1: u[i,j] * v[jp,k] * w[kp,ip]
    # Unfold as: M[(i,j), (jp,k,kp,ip)] = u[i,j] * (v[jp,k] * w[kp,ip])
    # The second factor isn't separable in a simple way for SVD.
    
    # Alternative: use the fact that the tensor is the identity in a sense.
    # Let me just do a random search for the rank-1 term.
    
    remaining = target.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(R):
        if time.time() - start_time > time_limit:
            break
        
        # Find best rank-1 approximation to remaining tensor
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        # Try several random rank-1 candidates
        for trial in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Random u, v, w
            u = np.random.randn(n, m)
            v = np.random.randn(m, p)
            w = np.random.randn(p, n)
            
            # Compute the rank-1 tensor
            # T1[i,j,jp,k,kp,ip] = u[i,j]*v[jp,k]*w[kp,ip]
            T1 = np.einsum('ij,jk,pi->ijjkpi', u, v, w, optimize=True)
            
            # Score = <T1, remaining> / ||T1||^2 (for projection)
            inner = np.sum(T1 * remaining)
            norm_sq = np.sum(T1 * T1)
            
            if norm_sq < 1e-12:
                continue
            
            score = inner / norm_sq
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            break
        
        # Project: the optimal coefficient for this direction
        # We want to minimize ||remaining - alpha * T1||^2
        # alpha = <remaining, T1> / ||T1||^2
        T1 = np.einsum('ij,jk,pi->ijjkpi', best_u, best_v, best_w, optimize=True)
        inner = np.sum(T1 * remaining)
        norm_sq = np.sum(T1 * T1)
        
        if norm_sq < 1e-12:
            break
        
        alpha = inner / norm_sq
        
        # Update remaining
        remaining -= alpha * T1
        
        # Store the rank-1 term with the coefficient absorbed
        U_list.append((alpha * best_u).flatten())
        V_list.append(best_v.flatten())
        W_list.append(best_w.flatten())
        
        # Check if we've converged
        if np.linalg.norm(remaining) < 1e-8:
            break
    
    if len(U_list) < R:
        return None
    
    # Build the full matrices
    U = np.column_stack([u.flatten() for u in U_list])
    V = np.column_stack([v.flatten() for v in V_list])
    W = np.column_stack([w.flatten() for w in W_list])
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    
    U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
    if _verify_fast(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
    
    return None

def _solve_rank_with_polish(n, m, p, R, target, target_flat, time_limit=10.0):
    """Try to find a decomposition of rank R using least_squares."""
    start_time = time.time()
    num_vars = R * (n * m + m * p + p * n)
    
    def residual(params):
        U = params[:R * n * m].reshape(n, m, R)
        V = params[R * n * m : R * (n * m + m * p)].reshape(m, p, R)
        W = params[R * (n * m + m * p):].reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,pir->ijjkpi', U, V, W, optimize=True)
        
        return (T - target).flatten()
    
    best_result = None
    best_residual_norm = np.inf
    
    num_attempts = 6
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        scale = random.uniform(0.3, 1.0)
        params = np.random.randn(num_vars) * scale
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=3000, xtol=1e-12, ftol=1e-12)
            res_norm = np.linalg.norm(result.fun)
            
            if res_norm < best_residual_norm:
                best_residual_norm = res_norm
                best_result = result
            
            if res_norm < 1e-8:
                U = result.x[:R * n * m].reshape(n * m, R)
                V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
                W = result.x[R * (n * m + m * p):].reshape(p * n, R)
                
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                
                U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if _verify_fast(U_s, V_s, W_s, n, m, p):
                    return U_s, V_s, W_s
                    
        except Exception:
            pass
    
    if best_result is not None and best_residual_norm < 1e-4:
        U = best_result.x[:R * n * m].reshape(n * m, R)
        V = best_result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
        W = best_result.x[R * (n * m + m * p):].reshape(p * n, R)
        
        U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
        if _verify_fast(U_s, V_s, W_s, n, m, p):
            return U_s, V_s, W_s
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
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
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Precompute target
    target, target_flat = _precompute_target(n, m, p)
    
    # Lower bound is max(nm, mp, pn) = 24
    # Try ranks from 47 down to 24
    for target_rank in range(47, 23, -1):
        if time.time() > DEADLINE - 10:
            break
        
        # Try alternating least squares first (faster, more restarts)
        if time.time() < DEADLINE - 15:
            result = _solve_rank_alternating(n, m, p, target_rank, target, target_flat, 
                                            time_limit=12.0, num_restarts=15)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        # Try greedy peeling
        if time.time() < DEADLINE - 20:
            result = _solve_rank_greedy_peel(n, m, p, target_rank, target, target_flat, 
                                            time_limit=8.0)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        # Try least_squares with polish
        if time.time() < DEADLINE - 25:
            result = _solve_rank_with_polish(n, m, p, target_rank, target, target_flat, 
                                            time_limit=8.0)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
