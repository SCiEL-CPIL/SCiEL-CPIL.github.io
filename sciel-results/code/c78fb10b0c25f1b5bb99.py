"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 140s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def verify_np(U, V, W, r):
        """Vectorized verification using numpy."""
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    total_error += (val - 1.0) ** 2
        return total_error < 1e-9, total_error
    
    def solve_for_factor(factor_name, U, V, W, r):
        """Solve for one factor given the other two using least squares."""
        if factor_name == 'U':
            A_mat = np.zeros((n * m * p, n * m * r))
            b_vec = np.ones(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (i * m + j) * r + c
                            A_mat[e, col] = V[j * p + k, c] * W[k * n + i, c]
            u_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return u_flat.reshape(n * m, r)
        elif factor_name == 'V':
            A_mat = np.zeros((n * m * p, m * p * r))
            b_vec = np.ones(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (j * p + k) * r + c
                            A_mat[e, col] = U[i * m + j, c] * W[k * n + i, c]
            v_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return v_flat.reshape(m * p, r)
        elif factor_name == 'W':
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = (i * m + j) * p + k
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A_mat[e, col] = U[i * m + j, c] * V[j * p + k, c]
            w_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return w_flat.reshape(p * n, r)
        return None
    
    def alternating_least_squares(r, U, V, W, max_iters=80, tol=1e-12):
        """Alternating least squares to find a rank-r decomposition."""
        for iteration in range(max_iters):
            if time.time() > DEADLINE:
                return None
            U_new = solve_for_factor('U', U, V, W, r)
            if U_new is None:
                return None
            V_new = solve_for_factor('V', U_new, V, W, r)
            if V_new is None:
                return None
            W_new = solve_for_factor('W', U_new, V_new, W, r)
            if W_new is None:
                return None
            valid, error = verify_np(U_new, V_new, W_new, r)
            if valid:
                return U_new, V_new, W_new, error
            if iteration > 0:
                prev_error = verify_np(U, V, W, r)[1]
                if abs(prev_error - error) < tol and error > 1e-8:
                    return None
            U, V, W = U_new, V_new, W_new
        return None
    
    def polish_with_least_squares(U, V, W, r):
        """Use scipy.optimize.least_squares to polish the solution."""
        if time.time() > DEADLINE - 10:
            return None
        
        # Flatten all variables
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_flat = x[:n*m*r].reshape(n*m, r)
            V_flat = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_flat = x[n*m*r+m*p*r:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j] * V_flat[j*p+k] * W_flat[k*n+i])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12)
            if result.cost < 1e-10:
                U_sol = result.x[:n*m*r].reshape(n*m, r)
                V_sol = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_sol = result.x[n*m*r+m*p*r:].reshape(p*n, r)
                valid, error = verify_np(U_sol, V_sol, W_sol, r)
                if valid:
                    return U_sol, V_sol, W_sol, error
        except Exception:
            pass
        return None
    
    def try_round_and_verify(U, V, W, r):
        """Try to round to half-integers and verify."""
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        valid, error = verify_np(U_rounded, V_rounded, W_rounded, r)
        if valid:
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        valid, error = verify_np(U_int, V_int, W_int, r)
        if valid:
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        return None
    
    def sparse_init_U(r, seed=42):
        """Create sparse structured initialization for U."""
        np.random.seed(seed)
        U = np.zeros((n * m, r))
        for row in range(n * m):
            nnz = min(3, r)
            cols = np.random.choice(r, size=nnz, replace=False)
            for c in cols:
                U[row, c] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        return U
    
    def try_rank(r):
        """Try to find a rank-r decomposition with multiple initializations."""
        if time.time() > DEADLINE - 5:
            return None
        
        # Try multiple different initialization strategies
        for strategy in range(8):
            if time.time() > DEADLINE - 5:
                break
            
            seed = 42 + strategy * 137
            
            if strategy == 0:
                # Sparse U init
                U = sparse_init_U(r, seed)
                np.random.seed(seed)
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            elif strategy == 1:
                # Dense random
                np.random.seed(seed)
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            elif strategy == 2:
                # Structured: U has identity-like structure
                U = np.zeros((n * m, r))
                for i in range(n):
                    for j in range(m):
                        col = (i * m + j) % r
                        U[i * m + j, col] = 1.0
                np.random.seed(seed)
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif strategy == 3:
                # V has identity-like structure
                np.random.seed(seed)
                U = np.random.randn(n * m, r) * 0.5
                V = np.zeros((m * p, r))
                for j in range(m):
                    for k in range(p):
                        col = (j * p + k) % r
                        V[j * p + k, col] = 1.0
                W = np.random.randn(p * n, r) * 0.5
            elif strategy == 4:
                # W has identity-like structure
                np.random.seed(seed)
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.zeros((p * n, r))
                for k in range(p):
                    for i in range(n):
                        col = (k * n + i) % r
                        W[k * n + i, col] = 1.0
            elif strategy == 5:
                # Block diagonal structure
                np.random.seed(seed)
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                block_size = max(1, r // 3)
                for b in range(min(3, r)):
                    start = b * block_size
                    end = min(start + block_size, r)
                    U[:, start:end] = np.random.randn(n * m, end-start) * 0.3
                    V[:, start:end] = np.random.randn(m * p, end-start) * 0.3
                    W[:, start:end] = np.random.randn(p * n, end-start) * 0.3
            elif strategy == 6:
                # Scale-invariant init
                np.random.seed(seed)
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
                for c in range(r):
                    norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                    if norm > 1e-10:
                        scale = (36.0 / norm) ** (1/3)
                        U[:, c] *= scale
                        V[:, c] *= scale
                        W[:, c] *= scale
            else:
                # Random with different scales
                np.random.seed(seed)
                U = np.random.randn(n * m, r) * 0.7
                V = np.random.randn(m * p, r) * 0.7
                W = np.random.randn(p * n, r) * 0.7
            
            result = alternating_least_squares(r, U, V, W, max_iters=60)
            if result is not None:
                U_sol, V_sol, W_sol, error = result
                rounded = try_round_and_verify(U_sol, V_sol, W_sol, r)
                if rounded is not None:
                    return rounded
                
                # Try polish
                polished = polish_with_least_squares(U_sol, V_sol, W_sol, r)
                if polished is not None:
                    U_p, V_p, W_p, err_p = polished
                    rounded = try_round_and_verify(U_p, V_p, W_p, r)
                    if rounded is not None:
                        return rounded
        
        return None
    
    # Try ranks from 18 upward (known lower bound is max(n*m, m*p, p*n) = 18)
    for r in range(18, 37):
        if time.time() > DEADLINE - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
