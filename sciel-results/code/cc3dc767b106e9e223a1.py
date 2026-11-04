"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import least_squares
from scipy.linalg import solve

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x7 matrix.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    schoolbook_rank = n * m * p  # 42
    
    # Build the target tensor
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def verify_exact(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        result = np.einsum('rc,sc,tc->rst', U_arr, V_arr, W_arr)
        return np.max(np.abs(result - T)) < tol
    
    def als_decompose(R, U_init, V_init, W_init, max_iter=100, tol=1e-8):
        """
        Alternating Least Squares for tensor decomposition.
        U: (N_U, R), V: (N_V, R), W: (N_W, R)
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column r, solve for U[:, r]
            for r in range(R):
                # T_calc[:, :, r] = sum_c U[:, c] * V[:, c] * W[:, c]
                # We want to find U[:, r] such that U[:, r] * (V[:, r] * W[:, r]) approximates T
                # Actually, the tensor is T = sum_r U[:, r] outer V[:, r] outer W[:, r]
                # So for fixed r, we need U[:, r] such that U[:, r] * (V[:, r] * W[:, r]) helps match T
                # This is tricky. Let's use a different approach.
                
                # The contribution of term r to T is U[:, r] outer V[:, r] outer W[:, r]
                # We want to minimize ||T - sum_r U[:, r] outer V[:, r] outer W[:, r]||^2
                # Fixing V and W, update U.
                
                # For each (a, b, c) in T, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # So for fixed r, U[a,r] appears in T[a,b,c] multiplied by V[b,r]*W[c,r]
                # This is a linear system in U[:, r]
                
                # Let's build the system: for each a, U[a,r] = ?
                # T[a,b,c] = sum_r' U[a,r'] * V[b,r'] * W[c,r']
                # For fixed a, this is linear in U[a, :]
                
                # Actually, let's do it differently. Fix V and W, solve for U.
                # T = sum_r U[:, r] outer V[:, r] outer W[:, r]
                # Reshape: T_reshaped[a, (b,c), r] = U[a, r] * V[b, r] * W[c, r]
                # This is not straightforward.
                
                # Let's use the standard ALS approach:
                # For U update: fix V, W. For each a, we have:
                # T[a, :, :] = sum_r U[a, r] * (V[:, r] outer W[:, r])
                # So T[a, :, :] is a linear combination of V[:, r] outer W[:, r] with coefficients U[a, r]
                
                # Build matrix M_a where M_a[:, r] = (V[:, r] outer W[:, r]).flatten()
                # Then T[a, :, :].flatten() = M_a @ U[a, :]
                
                # But this is expensive. Let's use a simpler approach with random restarts.
                pass
            
            # This ALS is complex. Let's use a different strategy.
            break
        
        return U, V, W
    
    def try_rank_structured(R, time_budget, max_restarts=50):
        """
        Try to find a valid decomposition of rank R using structured search.
        """
        if R >= best_rank:
            return None
        
        end_time = time.time() + time_budget
        total_vars = (N_U + N_V + N_W) * R
        
        def compute_residual(U, V, W):
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc.flatten() - T_flat)
        
        best_cost = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Structured initialization: start from schoolbook and merge terms
            # Each term in schoolbook has U, V, W with a single 1
            # Try to create R terms by randomly grouping schoolbook terms
            
            # Create random grouping of 42 schoolbook terms into R groups
            terms = list(range(schoolbook_rank))
            random.shuffle(terms)
            
            U = np.zeros((N_U, R), dtype=np.float64)
            V = np.zeros((N_V, R), dtype=np.float64)
            W = np.zeros((N_W, R), dtype=np.float64)
            
            # Assign terms to R bins
            for r in range(R):
                # Get terms for this bin
                start = r * schoolbook_rank // R
                end = (r + 1) * schoolbook_rank // R
                if r == R - 1:
                    end = schoolbook_rank
                
                for t in terms[start:end]:
                    i = t // (m * p)
                    j = (t % (m * p)) // p
                    k = t % p
                    
                    U[i * m + j, r] += 1
                    V[j * p + k, r] += 1
                    W[k * n + i, r] += 1
            
            # Optimize with least squares
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def res(params):
                len_U = N_U * R
                len_V = N_V * R
                U_p = params[:len_U].reshape(N_U, R)
                V_p = params[len_U:len_U+len_V].reshape(N_V, R)
                W_p = params[len_U+len_V:].reshape(N_W, R)
                T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                return (T_calc.flatten() - T_flat)
            
            try:
                sol = least_squares(res, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_U = sol.x[:N_U*R].reshape(N_U, R).copy()
                    best_V = sol.x[N_U*R:N_U*R+N_V*R].reshape(N_V, R).copy()
                    best_W = sol.x[N_U*R+N_V*R:].reshape(N_W, R).copy()
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_U is None or best_cost > 1e-4:
            return None
        
        # Refine to half-integers
        U_r = np.round(best_U * 2) / 2.0
        V_r = np.round(best_V * 2) / 2.0
        W_r = np.round(best_W * 2) / 2.0
        
        cost_r = compute_residual(U_r, V_r, W_r)
        if cost_r < 1e-9:
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Local refinement
        if time.time() < end_time:
            vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            
            for _pass in range(30):
                if time.time() > end_time:
                    break
                
                improved = False
                
                # Try perturbing U
                for row in range(N_U):
                    for c in range(R):
                        old = U_r[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            U_r[row, c] = v
                            if compute_residual(U_r, V_r, W_r) < 1e-9:
                                U_list = U_r.tolist()
                                V_list = V_r.tolist()
                                W_list = W_r.tolist()
                                if verify_exact(U_list, V_list, W_list):
                                    return {"U": U_list, "V": V_list, "W": W_list}
                        U_r[row, c] = old
                
                # Try perturbing V
                for row in range(N_V):
                    for c in range(R):
                        old = V_r[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            V_r[row, c] = v
                            if compute_residual(U_r, V_r, W_r) < 1e-9:
                                U_list = U_r.tolist()
                                V_list = V_r.tolist()
                                W_list = W_r.tolist()
                                if verify_exact(U_list, V_list, W_list):
                                    return {"U": U_list, "V": V_list, "W": W_list}
                        V_r[row, c] = old
                
                # Try perturbing W
                for row in range(N_W):
                    for c in range(R):
                        old = W_r[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            W_r[row, c] = v
                            if compute_residual(U_r, V_r, W_r) < 1e-9:
                                U_list = U_r.tolist()
                                V_list = V_r.tolist()
                                W_list = W_r.tolist()
                                if verify_exact(U_list, V_list, W_list):
                                    return {"U": U_list, "V": V_list, "W": W_list}
                        W_r[row, c] = old
        
        return None
    
    def try_merge_schoolbook(R, time_budget):
        """
        Try to merge schoolbook terms to reduce rank.
        """
        if R >= best_rank:
            return None
        
        end_time = time.time() + time_budget
        
        # Start with schoolbook
        U = np.zeros((N_U, schoolbook_rank), dtype=np.float64)
        V = np.zeros((N_V, schoolbook_rank), dtype=np.float64)
        W = np.zeros((N_W, schoolbook_rank), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Try to merge terms by setting some to zero and adjusting others
        # This is a heuristic approach
        
        best_cost = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for _ in range(100):
            if time.time() > end_time:
                break
            
            # Randomly select R terms to keep, set others to zero
            keep = set(random.sample(range(schoolbook_rank), min(R, schoolbook_rank)))
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            for c in range(schoolbook_rank):
                if c not in keep:
                    U_try[:, c] = 0
                    V_try[:, c] = 0
                    W_try[:, c] = 0
            
            # Optimize the kept terms
            x0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
            
            def res(params):
                len_U = N_U * schoolbook_rank
                len_V = N_V * schoolbook_rank
                U_p = params[:len_U].reshape(N_U, schoolbook_rank)
                V_p = params[len_U:len_U+len_V].reshape(N_V, schoolbook_rank)
                W_p = params[len_U+len_V:].reshape(N_W, schoolbook_rank)
                T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                return (T_calc.flatten() - T_flat)
            
            try:
                sol = least_squares(res, x0, method='lm', max_nfev=300)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_U = sol.x[:N_U*schoolbook_rank].reshape(N_U, schoolbook_rank).copy()
                    best_V = sol.x[N_U*schoolbook_rank:N_U*schoolbook_rank+N_V*schoolbook_rank].reshape(N_V, schoolbook_rank).copy()
                    best_W = sol.x[N_U*schoolbook_rank+N_V*schoolbook_rank:].reshape(N_W, schoolbook_rank).copy()
            except Exception:
                continue
        
        if best_U is None or best_cost > 1e-4:
            return None
        
        # Check if we can reduce to R by zeroing out columns
        U_r = np.round(best_U * 2) / 2.0
        V_r = np.round(best_V * 2) / 2.0
        W_r = np.round(best_W * 2) / 2.0
        
        # Find non-zero columns
        col_norms = np.linalg.norm(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]).reshape(3, -1), axis=0)
        nonzero_cols = np.where(col_norms > 1e-9)[0]
        
        if len(nonzero_cols) <= R:
            # Extract only non-zero columns
            U_final = U_r[:, nonzero_cols]
            V_final = V_r[:, nonzero_cols]
            W_final = W_r[:, nonzero_cols]
            
            if compute_residual(U_final, V_final, W_final) < 1e-9:
                U_list = U_final.tolist()
                V_list = V_final.tolist()
                W_list = W_final.tolist()
                if verify_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from just below schoolbook down
    time_elapsed = time.time() - start_time
    remaining = deadline - time_elapsed
    
    ranks_to_try = list(range(schoolbook_rank - 1, max(20, schoolbook_rank - 20), -1))
    
    for idx, R in enumerate(ranks_to_try):
        if time.time() > deadline:
            break
        
        frac = 1.0 / (idx + 1)
        alloc_time = remaining * 0.7 * frac
        if alloc_time < 1.0:
            break
        
        # Try structured search first
        result = try_rank_structured(R, alloc_time * 0.6, max_restarts=30)
        if result is not None:
            best_result = result
            best_rank = R
            break
        
        # Try merge approach
        if time.time() < deadline - 5:
            result = try_merge_schoolbook(R, alloc_time * 0.4)
            if result is not None:
                best_result = result
                best_rank = R
                break
    
    return best_result
# EVOLVE-BLOCK-END
