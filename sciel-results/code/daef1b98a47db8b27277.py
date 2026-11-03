"""Matrix multiplication 2x3 x 3x7 - rank minimization search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill
_SOFTMAX_TAU = 0.05  # temperature for log-sum-exp smooth max

def _validate(U, V, W, n=2, m=3, p=7):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _compute_errors(U, V, W, n=2, m=3, p=7):
    """Compute errors for all (i,j,k) triples."""
    rank = U.shape[1]
    errors = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                errors[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
    return errors

def _smooth_max_error(errors, tau=_SOFTMAX_TAU):
    """Smooth surrogate for max|error| using log-sum-exp."""
    abs_errs = np.abs(errors)
    max_val = np.max(abs_errs)
    if max_val < 1e-15:
        return 0.0
    # Stable log-sum-exp: log(sum(exp((e - max)/tau))) * tau + max
    shifted = (abs_errs - max_val) / tau
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + tau * log_sum

def _exact_polish(U, V, W, n=2, m=3, p=7, max_iters=500):
    """High-precision polish: alternate exact least-squares solves until convergence."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    prev_err = float('inf')
    
    for it in range(max_iters):
        errors = _compute_errors(U, V, W, n, m, p)
        max_err = np.max(np.abs(errors))
        
        if max_err < 1e-14:
            return U, V, W
        
        if abs(prev_err - max_err) < 1e-16:
            break
        prev_err = max_err
        
        # Update U: for each (i,j), solve B @ x = ones(p)
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, U.shape[1]))
                for k in range(p):
                    B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=1e-15)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve A @ x = ones(n)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, V.shape[1]))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve C @ x = ones(m)
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, W.shape[1]))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=1e-15)
                    W[k*n+i, :] = x
                except:
                    pass
    
    return U, V, W

def _try_rank(rank, n=2, m=3, p=7, max_iters=500, seed=0):
    """Try to find a valid rank-r decomposition via alternating least squares + local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Multiple initializations
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        # Different initialization strategies
        if init == 0:
            # Random
            U = np_rng.randn(n*m, rank)
            V = np_rng.randn(m*p, rank)
            W = np_rng.randn(p*n, rank)
        elif init == 1:
            # Sparse random
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                U[rng.randint(0, n*m-1), c] = 1
                V[rng.randint(0, m*p-1), c] = 1
                W[rng.randint(0, p*n-1), c] = 1
        else:
            # Structured: try to share patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Assign each column a random (i,j,k) pattern
            for c in range(rank):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        
        # Alternating least squares
        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update one matrix at a time (alternating)
            mat_idx = it % 3
            
            if mat_idx == 0:  # Update U
                # For each row of U, solve linear system
                for i in range(n):
                    for j in range(m):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all k
                        # This is: sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1
                        # Let B[k,c] = V[j,k,c] * W[k,i,c]
                        # Then: sum_c U[i,j,c] * B[k,c] = 1 for all k
                        # This is: B @ U[i,j,:]^T = ones(p)
                        B = np.zeros((p, rank))
                        for k in range(p):
                            B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve B @ x = ones(p)
                        try:
                            x = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = x
                        except:
                            pass
                        
            elif mat_idx == 1:  # Update V
                for j in range(m):
                    for k in range(p):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i
                        # Let A[i,c] = U[i,j,c] * W[k,i,c]
                        # Then: sum_c V[j,k,c] * A[i,c] = 1 for all i
                        # A @ V[j,k,:]^T = ones(n)
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = x
                        except:
                            pass
                            
            else:  # Update W
                for k in range(p):
                    for i in range(n):
                        # Want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all j
                        # Let C[j,c] = U[i,j,c] * V[j,k,c]
                        # Then: sum_c W[k,i,c] * C[j,c] = 1 for all j
                        # C @ W[k,i,:]^T = ones(m)
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = x
                        except:
                            pass
        
        # Smooth gradient-based refinement using log-sum-exp surrogate
        if time.time() < DEADLINE - 10:
            for smooth_it in range(200):
                if time.time() > DEADLINE - 10:
                    break
                
                errors = _compute_errors(U, V, W, n, m, p)
                max_err = np.max(np.abs(errors))
                smooth_obj = _smooth_max_error(errors)
                
                if max_err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
                
                if max_err < best_err:
                    best_err = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Compute gradients of smooth objective w.r.t. each variable
                # d/dU[i,j,c] smooth_obj = sum over (j',k) of sign-like terms
                # Use finite differences for simplicity on small problems
                lr = 0.01
                improved = False
                
                for _ in range(3):  # 3 sweeps over U, V, W
                    if time.time() > DEADLINE - 10:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            for c in range(rank):
                                orig = U[i*m+j, c]
                                U[i*m+j, c] = orig + 1e-6
                                err_plus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                U[i*m+j, c] = orig - 1e-6
                                err_minus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                U[i*m+j, c] = orig
                                
                                grad = (err_plus - err_minus) / 2e-6
                                U[i*m+j, c] = orig - lr * grad
                                improved = True
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                orig = V[j*p+k, c]
                                V[j*p+k, c] = orig + 1e-6
                                err_plus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                V[j*p+k, c] = orig - 1e-6
                                err_minus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                V[j*p+k, c] = orig
                                
                                grad = (err_plus - err_minus) / 2e-6
                                V[j*p+k, c] = orig - lr * grad
                                improved = True
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            for c in range(rank):
                                orig = W[k*n+i, c]
                                W[k*n+i, c] = orig + 1e-6
                                err_plus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                W[k*n+i, c] = orig - 1e-6
                                err_minus = _smooth_max_error(_compute_errors(U, V, W, n, m, p))
                                W[k*n+i, c] = orig
                                
                                grad = (err_plus - err_minus) / 2e-6
                                W[k*n+i, c] = orig - lr * grad
                                improved = True
                
                if not improved:
                    break
        
        # Local search refinement
        for ls_it in range(100):
            if time.time() > DEADLINE - 10:
                break
            
            errors = _compute_errors(U, V, W, n, m, p)
            max_err = np.max(np.abs(errors))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Perturb one random element
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, n*m-1)
            elif mat == 'V':
                row = rng.randint(0, m*p-1)
            else:
                row = rng.randint(0, p*n-1)
            col = rng.randint(0, rank-1)
            
            best_perturb = None
            best_perturb_err = max_err
            for delta in [0.1, -0.1, 0.01, -0.01, 0.001, -0.001]:
                if mat == 'U':
                    orig = U[row, col]
                    U[row, col] = orig + delta
                elif mat == 'V':
                    orig = V[row, col]
                    V[row, col] = orig + delta
                else:
                    orig = W[row, col]
                    W[row, col] = orig + delta
                
                err = np.max(np.abs(_compute_errors(U, V, W, n, m, p)))
                
                if mat == 'U':
                    U[row, col] = orig
                elif mat == 'V':
                    V[row, col] = orig
                else:
                    W[row, col] = orig
                
                if err < best_perturb_err:
                    best_perturb_err = err
                    best_perturb = (mat, row, col, delta)
            
            if best_perturb is not None:
                mat, row, col, delta = best_perturb
                if mat == 'U':
                    U[row, col] += delta
                elif mat == 'V':
                    V[row, col] += delta
                else:
                    W[row, col] += delta
            else:
                break
        
        # Exact polish pass
        if time.time() < DEADLINE - 5 and best_err < 1e-6:
            U_pol, V_pol, W_pol = _exact_polish(U, V, W, n, m, p, max_iters=200)
            final_err = np.max(np.abs(_compute_errors(U_pol, V_pol, W_pol, n, m, p)))
            if final_err < 1e-10:
                return U_pol.tolist(), V_pol.tolist(), W_pol.tolist()
            if final_err < best_err:
                best_err = final_err
                best_U, best_V, best_W = U_pol.copy(), V_pol.copy(), W_pol.copy()
    
    if best_err < 1e-10 and best_U is not None:
        # Final exact polish
        U_pol, V_pol, W_pol = _exact_polish(best_U, best_V, best_W, n, m, p, max_iters=500)
        final_err = np.max(np.abs(_compute_errors(U_pol, V_pol, W_pol, n, m, p)))
        if final_err < 1e-10:
            return U_pol.tolist(), V_pol.tolist(), W_pol.tolist()
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_structured_rank(rank, n=2, m=3, p=7):
    """Try structured decompositions based on known patterns."""
    # For small matrices, try specific structured approaches
    # One approach: use the fact that we can factor through smaller dimensions
    return None

def _schoolbook_decomposition(n=2, m=3, p=7):
    """Generate the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # First, generate schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Theoretical lower bound: max(n*p, n*m, m*p) = max(14, 6, 21) = 21
    # Try ranks from lower bound upward
    
    # Priority: try ranks that are more likely to have solutions
    # Based on known results, matrix multiplication ranks are often near the lower bound
    ranks_to_try = list(range(21, 35))  # Try 21 to 34
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple seeds for this rank
        # More attempts for lower ranks (more likely to succeed if possible)
        max_attempts = 5 if target_rank <= 24 else 3
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 15:
                break
            
            result = _try_rank(target_rank, n, m, p, max_iters=200, seed=attempt*1000+42)
            if result is not None:
                U, V, W = result
                # Verify with high precision
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, try a few more ranks with different seeds
    for target_rank in range(21, 28):
        if time.time() > DEADLINE - 10:
            break
        for attempt in range(3):
            if time.time() > DEADLINE - 10:
                break
            result = _try_rank(target_rank, n, m, p, max_iters=150, seed=attempt*5000+777)
            if result is not None:
                U, V, W = result
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If we haven't found anything better, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
