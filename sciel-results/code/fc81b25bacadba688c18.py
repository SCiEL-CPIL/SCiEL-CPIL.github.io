import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # 20s safety margin before 150s kill
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if (i % m == j // p) and (i // m == k // n) and (j % p == k % n):
                    T[i, j, k] = 1.0
    
    # Schoolbook rank 12
    U_school = np.zeros((4, 12))
    V_school = np.zeros((6, 12))
    W_school = np.zeros((6, 12))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T"""
        rank = U.shape[1]
        computed = np.zeros((4, 6, 6))
        for i in range(4):
            for j in range(6):
                for k in range(6):
                    computed[i, j, k] = np.dot(U[i, :], V[j, :] * W[k, :])
        return np.max(np.abs(computed - T)) < tol
    
    def try_round_and_check(U, V, W, multipliers=[1, 2, 3, 4, 6, 12]):
        """Try rounding U, V, W to rational numbers with various denominators"""
        for mult in multipliers:
            U_r = np.round(U * mult) / mult
            V_r = np.round(V * mult) / mult
            W_r = np.round(W * mult) / mult
            if check_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        # Try half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if check_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None, None, None
    
    # Strategy 1: Finite field rank reduction from schoolbook
    # Work over GF(13) to find rank-11 or lower
    def gf13_rank_reduction():
        """Try to reduce rank over GF(13) from schoolbook-12"""
        q = 13
        T_gf = np.zeros((4, 6, 6), dtype=int)
        for i in range(4):
            for j in range(6):
                for k in range(6):
                    T_gf[i, j, k] = int(T[i, j, k]) % q
        
        # Schoolbook over GF(13)
        U_s = np.zeros((4, 12), dtype=int)
        V_s = np.zeros((6, 12), dtype=int)
        W_s = np.zeros((6, 12), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        
        np.random.seed(42)
        for trial in range(200):
            if time.time() > deadline:
                break
            # Try to eliminate one random column by adding a random multiple
            # of another column's factors
            U_trial = U_s.copy()
            V_trial = V_s.copy()
            W_trial = W_s.copy()
            
            # Pick a random column to eliminate and a random column to add
            elim_col = np.random.randint(0, 12)
            add_col = np.random.randint(0, 12)
            if elim_col == add_col:
                continue
            
            # Choose random coefficient in GF(13)
            coeff = np.random.randint(1, q)
            
            # U_trial[:, elim_col] += coeff * U_trial[:, add_col]
            # V_trial[:, elim_col] += coeff * V_trial[:, add_col]  
            # W_trial[:, elim_col] += coeff * W_trial[:, add_col]
            # But this doesn't eliminate; we need a different approach.
            
            # Better: try random linear combinations to see if we can get
            # a column that becomes zero in the tensor product
            # Actually, let's try a different approach: random rank-11 init
            
            # Random rank-11 initialization over GF(13)
            U_r = np.random.randint(0, q, size=(4, 11), dtype=int)
            V_r = np.random.randint(0, q, size=(6, 11), dtype=int)
            W_r = np.random.randint(0, q, size=(6, 11), dtype=int)
            
            # Check if this decomposes T over GF(13)
            valid = True
            for i in range(4):
                for j in range(6):
                    for k in range(6):
                        val = 0
                        for r in range(11):
                            val = (val + U_r[i, r] * V_r[j, r] * W_r[k, r]) % q
                        if val != T_gf[i, j, k]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid rank-11 decomposition over GF(13)
                # Try to lift to integers
                U_int = U_r.astype(float)
                V_int = V_r.astype(float)
                W_int = W_r.astype(float)
                return U_int, V_int, W_int
        
        return None, None, None
    
    # Strategy 2: Optimization with better initialization
    def compute_tensor_fast(params):
        U = params[0:44].reshape(4, 11)
        V = params[44:110].reshape(6, 11)
        W = params[110:176].reshape(6, 11)
        # Use einsum for speed
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return result
    
    def objective(params):
        computed = compute_tensor_fast(params)
        return np.sum((computed - T)**2)
    
    def objective_grad(params):
        """Objective with gradient for L-BFGS"""
        U = params[0:44].reshape(4, 11)
        V = params[44:110].reshape(6, 11)
        W = params[110:176].reshape(6, 11)
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = computed - T
        obj = np.sum(diff**2)
        
        # Gradient computation
        # dU[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
        # dV[j,r] = 2 * sum_{i,k} diff[i,j,k] * U[i,r] * W[k,r]
        # dW[k,r] = 2 * sum_{i,j} diff[i,j,k] * U[i,r] * V[j,r]
        
        dU = 2.0 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        dV = 2.0 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        dW = 2.0 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return obj, grad
    
    best_result = None
    best_obj = float('inf')
    
    # Try finite field approach first (fast)
    if time.time() < deadline:
        U_ff, V_ff, W_ff = gf13_rank_reduction()
        if U_ff is not None:
            # Check if it works over reals
            if check_decomposition(U_ff, V_ff, W_ff):
                U_r, V_r, W_r = try_round_and_check(U_ff, V_ff, W_ff)
                if U_r is not None:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": U_ff.tolist(), "V": V_ff.tolist(), "W": W_ff.tolist()}
    
    # Optimization-based search
    np.random.seed(123)
    
    # Better initialization: use schoolbook structure with one column perturbed
    for trial in range(30):
        if time.time() > deadline - 10:  # Leave 10s for final checks
            break
        
        # Initialize near schoolbook: take 11 of 12 schoolbook columns
        # and add small random perturbation
        if trial == 0:
            # Start from schoolbook minus one column
            idx = np.random.permutation(12)[:11]
            params0 = np.zeros(176)
            for new_col, old_col in enumerate(idx):
                for i in range(4):
                    params0[i*11 + new_col] = U_school[i, old_col]
                for j in range(6):
                    params0[44 + j*11 + new_col] = V_school[j, old_col]
                for k in range(6):
                    params0[110 + k*11 + new_col] = W_school[k, old_col]
            # Add small noise
            params0 += np.random.randn(176) * 0.1
        else:
            params0 = np.random.randn(176) * 0.5
        
        method = 'L-BFGS-B' if trial < 15 else 'Nelder-Mead'
        if method == 'L-BFGS-B':
            result = minimize(objective_grad, params0, method=method, jac=True,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        else:
            result = minimize(objective, params0, method=method,
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
        
        if best_obj < 1e-10:
            break
    
    # Check best result
    if best_result is not None and best_obj < 1e-8:
        U = best_result[0:44].reshape(4, 11)
        V = best_result[44:110].reshape(6, 11)
        W = best_result[110:176].reshape(6, 11)
        
        U_r, V_r, W_r = try_round_and_check(U, V, W)
        if U_r is not None:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
