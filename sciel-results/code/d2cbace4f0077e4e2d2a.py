import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def als_decompose(target_rank, max_iters=300, tol=1e-10, seed=None):
        """Alternating Least Squares for tensor decomposition with structured init."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Structured initialization: use patterns from known matrix mult algorithms
        # For rank R, initialize with a mix of structured and random
        if target_rank >= 14:
            # Try structured initialization based on block patterns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # First fill with a structured pattern
            # Use the fact that we're multiplying (2,3) x (3,3)
            # Pattern: some entries will be 0, 1, -1, or combinations
            for r in range(target_rank):
                # Assign random but structured values
                U[:, r] = rng.choice([0, 1, -1, 0.5, -0.5], size=n*m)
                V[:, r] = rng.choice([0, 1, -1, 0.5, -0.5], size=m*p)
                W[:, r] = rng.choice([0, 1, -1, 0.5, -0.5], size=p*n)
            
            # Add small random perturbation to break symmetry
            U += rng.randn(n * m, target_rank) * 0.1
            V += rng.randn(m * p, target_rank) * 0.1
            W += rng.randn(p * n, target_rank) * 0.1
        else:
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
        
        prev_error = np.inf
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
                
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    A = np.zeros((target_rank, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    A = np.zeros((target_rank, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    A = np.zeros((target_rank, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
            
            error = compute_error(U, V, W, target_rank)
            
            if error < tol:
                break
            
            # Early stopping if error is not decreasing
            if error > prev_error * 1.01 and iter_ > 20:
                break
            prev_error = error
        
        return U, V, W, error
    
    # Try ranks from 14 upward
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        max_restarts = 40 if target_rank <= 15 else 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            U, V, W, error = als_decompose(target_rank, max_iters=200, tol=1e-10, seed=restart * 7919 + target_rank * 104729)
            
            if U is None:
                break
            
            if error < 1e-8:
                # Polish with high-precision refinement
                # Newton-like refinement: treat as nonlinear system
                for polish_iter in range(50):
                    if time.time() - start_time > deadline:
                        break
                    
                    # Compute residual
                    U_r = U.reshape(n, m, target_rank)
                    V_r = V.reshape(m, p, target_rank)
                    W_r = W.reshape(p, n, target_rank)
                    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    residual = T - result
                    
                    if np.linalg.norm(residual) < 1e-12:
                        break
                    
                    # Perturb slightly toward exact solution
                    # Use gradient-like update
                    step = 1e-6
                    for r in range(target_rank):
                        for i in range(n):
                            for j in range(m):
                                # d(U[i,j,r])/d(residual[i,j,k]) involves V[j,k,r]*W[k,i,r]
                                grad_U = 0
                                for k in range(p):
                                    grad_U += residual[i, j, k] * V[j * p + k, r] * W[k * n + i, r]
                                U[i * m + j, r] -= step * grad_U
                                
                        for j in range(m):
                            for k in range(p):
                                grad_V = 0
                                for i in range(n):
                                    grad_V += residual[i, j, k] * U[i * m + j, r] * W[k * n + i, r]
                                V[j * p + k, r] -= step * grad_V
                                
                        for k in range(p):
                            for i in range(n):
                                grad_W = 0
                                for j in range(m):
                                    grad_W += residual[i, j, k] * U[i * m + j, r] * V[j * p + k, r]
                                W[k * n + i, r] -= step * grad_W
                    
                    error = compute_error(U, V, W, target_rank)
                    if error < 1e-12:
                        break
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            max_err = max(max_err, abs(val - T[i, j, k]))
                
                if max_err < 1e-8:
                    # Round to nearest half-integer
                    def round_to_half(arr):
                        rounded = np.round(arr * 2) / 2
                        max_diff = np.max(np.abs(rounded - arr))
                        if max_diff < 1e-4:
                            # Verify rounded solution
                            U_rounded = rounded if arr is U else np.round(arr * 2) / 2
                            max_err_rounded = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = 0.0
                                        for r in range(target_rank):
                                            val += U_rounded[i * m + j, r] * np.round(V * 2) / 2[j * p + k, r] * np.round(W * 2) / 2[k * n + i, r]
                                        max_err_rounded = max(max_err_rounded, abs(val - T[i, j, k]))
                            if max_err_rounded < 1e-4:
                                return rounded
                        return arr
                    
                    best_U = round_to_half(U.copy())
                    best_V = round_to_half(V.copy())
                    best_W = round_to_half(W.copy())
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Final verification of best solution
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err
    
    # If we have a solution, verify it
    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            # Fallback to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
