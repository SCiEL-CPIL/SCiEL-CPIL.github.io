import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W


def _compute_residuals(U, V, W, n, m, p):
    """Compute residuals for all tensor entries."""
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Build list of tensor entries where T=1
    entries_a = []
    entries_b = []
    entries_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries_a.append(a)
                entries_b.append(b)
                entries_c.append(c)
    
    entries_a = np.array(entries_a)
    entries_b = np.array(entries_b)
    entries_c = np.array(entries_c)
    
    # Compute predictions: sum_r U[a,r] * V[b,r] * W[c,r]
    UA = U[entries_a, :]  # (N, rank)
    VB = V[entries_b, :]  # (N, rank)
    WC = W[entries_c, :]  # (N, rank)
    
    pred = (UA * VB * WC).sum(axis=1)
    residuals = pred - 1.0
    
    return residuals, entries_a, entries_b, entries_c


def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition by optimizing from rank-32 starting point."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    # Precompute tensor entries
    entries_a = []
    entries_b = []
    entries_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries_a.append(a)
                entries_b.append(b)
                entries_c.append(c)
    
    entries_a = np.array(entries_a)
    entries_b = np.array(entries_b)
    entries_c = np.array(entries_c)
    N_entries = len(entries_a)
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_max_res = float('inf')
    
    # Try each column as the one to eliminate
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.85:
            break
        
        # Initialize with rank-32 solution
        U_cur = U0.copy()
        V_cur = V0.copy()
        W_cur = W0.copy()
        
        # We'll optimize all parameters with a penalty on the target column
        # Objective: soft_max(|residuals|) + lambda * norm(target_col)^2
        
        def objective_and_grad(params):
            # Unpack parameters
            idx = 0
            U_flat = params[idx:idx + nm * rank]; idx += nm * rank
            V_flat = params[idx:idx + mp * rank]; idx += mp * rank
            W_flat = params[idx:idx + pn * rank]; idx += pn * rank
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            # Compute residuals
            UA = U[entries_a, :]
            VB = V[entries_b, :]
            WC = W[entries_c, :]
            pred = (UA * VB * WC).sum(axis=1)
            res = pred - 1.0
            
            # Soft-max objective (smooth approximation to max |residual|)
            # Use log-sum-exp for numerical stability
            abs_res = np.abs(res)
            max_abs = np.max(abs_res)
            if max_abs > 1e-10:
                # Soft max: log(sum(exp(beta * x))) / beta for large beta
                beta = 50.0
                shifted = beta * abs_res
                max_shifted = np.max(shifted)
                soft_max = (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
            else:
                soft_max = 0.0
            
            # Penalty on target column to encourage it to go to zero
            target_U = U[:, target_col]
            target_V = V[:, target_col]
            target_W = W[:, target_col]
            penalty = 0.01 * (np.sum(target_U**2) + np.sum(target_V**2) + np.sum(target_W**2))
            
            obj = soft_max + penalty
            
            # Compute gradient
            # d(soft_max)/d(res_e) = beta * exp(beta * |res_e|) / sum(exp(beta * |res|)) * sign(res_e)
            if max_abs > 1e-10:
                beta = 50.0
                shifted = beta * abs_res
                max_shifted = np.max(shifted)
                exp_shifted = np.exp(shifted - max_shifted)
                sum_exp = np.sum(exp_shifted)
                weights = beta * exp_shifted / sum_exp  # (N,)
                d_obj_d_res = weights * np.sign(res)  # (N,)
            else:
                d_obj_d_res = np.zeros(N_entries)
            
            # Gradient w.r.t. U[a, r] = sum_{e: a_e=a} d_obj_d_res[e] * V[b_e, r] * W[c_e, r]
            # We can compute this efficiently using the precomputed entries
            
            # For each entry e, the contribution to gradient of U[entries_a[e], r] is:
            # d_obj_d_res[e] * V[entries_b[e], r] * W[entries_c[e], r]
            
            # Let's compute grad_U, grad_V, grad_W
            # grad_U[a, r] = sum_{e: entries_a[e]==a} d_obj_d_res[e] * V[entries_b[e], r] * W[entries_c[e], r]
            
            # Efficient computation:
            # For each e, we add d_obj_d_res[e] * V[entries_b[e], :] * W[entries_c[e], :] to grad_U[entries_a[e], :]
            
            grad_U = np.zeros((nm, rank))
            grad_V = np.zeros((mp, rank))
            grad_W = np.zeros((pn, rank))
            
            # Vectorized accumulation
            # d_obj_d_res shape (N,), V[entries_b] shape (N, rank), W[entries_c] shape (N, rank)
            # For U: grad_U[entries_a[e], r] += d_obj_d_res[e] * V[entries_b[e], r] * W[entries_c[e], r]
            
            # Use np.add.at for accumulated updates
            dres_VW = d_obj_d_res[:, None] * VB * WC  # (N, rank)
            np.add.at(grad_U, entries_a, dres_VW)
            
            dres_UW = d_obj_d_res[:, None] * UA * WC  # (N, rank)
            np.add.at(grad_V, entries_b, dres_UW)
            
            dres_UV = d_obj_d_res[:, None] * UA * VB  # (N, rank)
            np.add.at(grad_W, entries_c, dres_UV)
            
            # Add penalty gradient
            penalty_lambda = 0.01
            grad_U[:, target_col] += 2 * penalty_lambda * U[:, target_col]
            grad_V[:, target_col] += 2 * penalty_lambda * V[:, target_col]
            grad_W[:, target_col] += 2 * penalty_lambda * W[:, target_col]
            
            # Flatten gradient
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad
        
        # Pack initial parameters
        params0 = np.concatenate([U_cur.flatten(), V_cur.flatten(), W_cur.flatten()])
        
        # Run optimization
        try:
            result = minimize(
                objective_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_max_res:
                # Unpack result
                idx = 0
                U_flat = result.x[idx:idx + nm * rank]; idx += nm * rank
                V_flat = result.x[idx:idx + mp * rank]; idx += mp * rank
                W_flat = result.x[idx:idx + pn * rank]; idx += pn * rank
                
                U_res = U_flat.reshape(nm, rank)
                V_res = V_flat.reshape(mp, rank)
                W_res = W_flat.reshape(pn, rank)
                
                # Check if the target column is close to zero
                target_norm = (np.sum(U_res[:, target_col]**2) + 
                             np.sum(V_res[:, target_col]**2) + 
                             np.sum(W_res[:, target_col]**2))
                
                if target_norm < 1e-6:
                    # Remove the target column
                    U_new = np.delete(U_res, target_col, axis=1)
                    V_new = np.delete(V_res, target_col, axis=1)
                    W_new = np.delete(W_res, target_col, axis=1)
                    
                    # Verify the rank-31 solution
                    UA = U_new[entries_a, :]
                    VB = V_new[entries_b, :]
                    WC = W_new[entries_c, :]
                    pred = (UA * VB * WC).sum(axis=1)
                    max_res = np.max(np.abs(pred - 1.0))
                    
                    if max_res < 1e-8:
                        return U_new, V_new, W_new
        except Exception:
            pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-32 solution
    U, V, W = _build_rank32()
    
    best_U, best_V, best_W = U, V, W
    best_rank = 32
    
    # Try to find a rank-31 decomposition
    if time.time() < deadline - 30:
        search_budget = min(60.0, deadline - time.time() - 5.0)
        result = _try_rank31_search(U, V, W, n, m, p, search_budget)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 31
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
