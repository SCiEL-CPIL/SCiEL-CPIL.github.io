import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                indices.append((u, v, w))

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((U_size, rank_school))
    V_school = np.zeros((V_size, rank_school))
    W_school = np.zeros((W_size, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify_solution(U, V, W, atol=1e-9):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=atol): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=atol): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=atol): return False
        
        # Check exact decomposition
        for u, v, w in indices:
            s = np.dot(U[u, :], V[v, :] * W[w, :])
            if not np.isclose(s, 1.0, atol=atol):
                return False
        return True

    def snap_and_verify(U, V, W, atol=1e-9):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_solution(U_snap, V_snap, W_snap, atol):
            return U_snap, V_snap, W_snap
        return None

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
        def current_error(U, V, W):
            err = 0.0
            for u, v, w in indices:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                err += (s - 1.0) ** 2
            return err
        
        best_err = current_error(U_round, V_round, W_round)
        best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
        
        # Simple hill climbing on the discrete half-integer lattice
        max_steps = 100
        step_size = 0.5
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)

        for _ in range(max_steps):
            improved = False
            for _ in range(80):
                mat_idx = rng.randint(3)
                if mat_idx == 0:
                    row, col = rng.randint(U_size), rng.randint(R)
                    orig_val = best_U[row, col]
                    for delta in [-step_size, step_size, -2*step_size, 2*step_size]:
                        new_val = orig_val + delta
                        if abs(new_val) > 10: continue
                        best_U[row, col] = new_val
                        err = current_error(best_U, best_V, best_W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            best_U[row, col] = orig_val
                elif mat_idx == 1:
                    row, col = rng.randint(V_size), rng.randint(R)
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
                    row, col = rng.randint(W_size), rng.randint(R)
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
                
        if verify_solution(best_U, best_V, best_W):
            return best_U, best_V, best_W
            
        return None

    def find_decomposition(R, num_restarts=20, timeout=10):
        start_time = time.time()
        size_U = U_size * R
        size_V = V_size * R
        size_W = W_size * R
        total_size = size_U + size_V + size_W
        
        def objective(params, R):
            U = params[:size_U].reshape((U_size, R))
            V = params[size_U:size_U + size_V].reshape((V_size, R))
            W = params[size_U + size_V:].reshape((W_size, R))
            err = 0.0
            for u, v, w in indices:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                err += (s - 1.0) ** 2
            reg = 1e-6 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return err + reg

        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(restart * 123 + R)
            params = np.random.randn(total_size) * 0.5
            
            try:
                res = minimize(objective, params, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                continue

            if res.fun < 1e-8:
                U, V, W = res.x[:size_U].reshape((U_size, R)), \
                         res.x[size_U:size_U + size_V].reshape((V_size, R)), \
                         res.x[size_U + size_V:].reshape((W_size, R))
                result = polish_solution(U, V, W, R)
                if result is not None:
                    return result
                    
        return None

    def try_structured_decomposition(R):
        """
        Try a structured decomposition that exploits the 2x3x3 problem structure.
        We'll use a pattern based on the algebraic relations.
        """
        if R < 14 or R > 18:
            return None
            
        # Try various structured initializations
        rng = np.random.RandomState(42)
        
        for trial in range(50):
            if time.time() > deadline - 5:
                break
                
            # Structured initialization with symmetries
            U = np.zeros((U_size, R))
            V = np.zeros((V_size, R))
            W = np.zeros((W_size, R))
            
            # Fill with structured patterns
            for r in range(R):
                # Use a mix of structured and random values
                if trial % 3 == 0:
                    # Pattern 1: based on row/column indices
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] = 0.5 * ((i + j + r) % 3 - 1)
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, r] = 0.5 * ((j + k + r) % 3 - 1)
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = 0.5 * ((k + i + r) % 3 - 1)
                elif trial % 3 == 1:
                    # Pattern 2: alternating
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] = 0.5 * ((i * j + r) % 2 - 0.5)
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, r] = 0.5 * ((j * k + r) % 2 - 0.5)
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = 0.5 * ((k * i + r) % 2 - 0.5)
                else:
                    # Pattern 3: random but sparse
                    for i in range(n):
                        for j in range(m):
                            if rng.random() < 0.7:
                                U[i*m+j, r] = rng.choice([-1, -0.5, 0.5, 1])
                    for j in range(m):
                        for k in range(p):
                            if rng.random() < 0.7:
                                V[j*p+k, r] = rng.choice([-1, -0.5, 0.5, 1])
                    for k in range(p):
                        for i in range(n):
                            if rng.random() < 0.7:
                                W[k*n+i, r] = rng.choice([-1, -0.5, 0.5, 1])
            
            # Quick check if this is already valid
            if verify_solution(U, V, W):
                return U, V, W
            
            # If not, try to polish
            result = polish_solution(U, V, W, R)
            if result is not None:
                return result
        
        return None

    start_time = time.time()
    deadline = 135  # Safety margin (150s limit, 15s buffer)

    # First try structured decompositions for ranks 15-17
    for R in [15, 16, 17]:
        if time.time() > deadline - 30:
            break
            
        result = try_structured_decomposition(R)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Then try search-based
        time_remaining = deadline - time.time() - 5
        if time_remaining < 5:
            continue
            
        restarts = max(3, min(20, int(time_remaining / 2.0)))
        res = find_decomposition(R, num_restarts=restarts, timeout=deadline - time.time() - 3)
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
