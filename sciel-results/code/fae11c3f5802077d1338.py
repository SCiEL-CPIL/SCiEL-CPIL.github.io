import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    
    # Precompute index mappings for objective function
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                indices.append((u, v, w))

    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(params, R):
        size_U = n * m * R
        size_V = m * p * R
        U = params[:size_U].reshape((n * m, R))
        V = params[size_U:size_U + size_V].reshape((m * p, R))
        W = params[size_U + size_V:].reshape((p * n, R))
        return U, V, W

    def objective(params, R):
        U, V, W = unflatten_params(params, R)
        err = 0.0
        for u, v, w in indices:
            s = np.dot(U[u, :], V[v, :] * W[w, :])
            err += (s - 1.0) ** 2
        # Regularization to prefer small integers/half-integers
        # L1 norm encourages sparsity, but we want half-integers.
        # Let's just use a small L2 penalty to keep values bounded.
        reg = 1e-5 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
        return err + reg

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        for u, v, w in indices:
            s = np.dot(U[u, :], V[v, :] * W[w, :])
            if not np.isclose(s, 1.0, atol=1e-9):
                return False
        return True

    def polish_solution(U, V, W, R):
        """
        Try to fix the solution by snapping to half-integers and then
        performing a local search on the snapped values.
        """
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Local search: try to improve the error by shifting entries
        # We define a cost function based on the current error
        def current_error(U, V, W):
            err = 0.0
            for u, v, w in indices:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                err += (s - 1.0) ** 2
            return err
        
        best_err = current_error(U_round, V_round, W_round)
        best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
        
        # Simple hill climbing on the discrete half-integer lattice
        # This is expensive, so we limit it
        max_steps = 200
        step_size = 0.5
        
        for _ in range(max_steps):
            improved = False
            # Try shifting random entries
            for _ in range(50):
                # Pick a random matrix and entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row, col = np.random.randint(n * m), np.random.randint(R)
                    orig_val = best_U[row, col]
                    for delta in [-step_size, step_size, -2*step_size, 2*step_size]:
                        new_val = orig_val + delta
                        # Ensure we stay in a reasonable range
                        if abs(new_val) > 10: continue
                        best_U[row, col] = new_val
                        err = current_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            best_U[row, col] = orig_val
                elif mat_idx == 1:
                    row, col = np.random.randint(m * p), np.random.randint(R)
                    orig_val = best_V[row, col]
                    for delta in [-step_size, step_size, -2*step_size, 2*step_size]:
                        new_val = orig_val + delta
                        if abs(new_val) > 10: continue
                        best_V[row, col] = new_val
                        err = current_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            best_V[row, col] = orig_val
                else:
                    row, col = np.random.randint(p * n), np.random.randint(R)
                    orig_val = best_W[row, col]
                    for delta in [-step_size, step_size, -2*step_size, 2*step_size]:
                        new_val = orig_val + delta
                        if abs(new_val) > 10: continue
                        best_W[row, col] = new_val
                        err = current_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            best_W[row, col] = orig_val
            if not improved:
                break
                
        # Final check
        if verify_solution(best_U, best_V, best_W):
            return best_U, best_V, best_W
            
        return None

    def find_decomposition(R, num_restarts=20, timeout=10):
        start_time = time.time()
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            # Use small random values to help convergence
            np.random.seed(restart * 123 + R)
            params = np.random.randn(total_size) * 0.5
            
            res = minimize(objective, params, args=(R,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                U, V, W = unflatten_params(res.x, R)
                # Polish
                result = polish_solution(U, V, W, R)
                if result is not None:
                    return result
                    
        return None

    start_time = time.time()
    deadline = 140  # Safety margin

    # Try rank 16 first
    for R in [16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Adjust restarts based on time remaining
        time_remaining = deadline - time.time()
        restarts = int(time_remaining / 5) # ~5s per restart
        restarts = max(10, min(50, restarts))
        
        res = find_decomposition(R, num_restarts=restarts, timeout=deadline - time.time())
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
