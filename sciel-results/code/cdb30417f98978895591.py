import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    # lags 1 to N-1
    max_off_diag = np.max(np.abs(ac[1:N]))
    return max_off_diag / (s * s)

def _coordinate_descent(x, max_time=10.0):
    """
    Perform coordinate descent on the signs of x.
    Since x[i] is +/- 1, a move is a flip.
    We evaluate the change in objective for flipping each coordinate.
    To speed this up, we can use the fact that flipping x[k] from 1 to -1
    changes the autoconvolution C by:
    C_new = C_old - 2*x[k]*(x + shift(x, k) + shift(x, -k) - 2*x[k]*delta(lag, 0)?)
    
    Actually, let's just recompute for simplicity first, or use a fast approximation.
    Recomputing FFT for each of N coordinates is O(N^2 log N).
    N=400. 400 * 400 * 9 ~ 1.4M ops. 
    If we do this for 100 iterations, that's 140M ops. Might be too slow in Python if not optimized.
    
    Let's try a faster local search:
    1. Compute C.
    2. For each i, estimate delta C if we flip x[i].
       Flipping x[i] (from v to -v) changes C[l] by:
       -2*v * (x[i+l] + x[i-l] - 2*v*delta(l,0)) ... wait.
       C[l] = sum_j x[j] x[j+l]
       C_new[l] = sum_j x'_j x'_{j+l}
       x'_j = x_j except x'_i = -x_i.
       Terms involving x_i:
       1. j=i: x_i x_{i+l} -> -x_i x_{i+l}
       2. j+l=i => j=i-l: x_{i-l} x_i -> x_{i-l} (-x_i) = -x_{i-l} x_i
       3. If l=0, j=i is counted twice? No, sum is over j.
          C[0] = sum x_j^2. Flipping x_i doesn't change C[0].
          For l != 0:
          Change in C[l] = -2 * x_i * (x_{i+l} + x_{i-l}) (with boundary checks)
          
    So we can compute the new C in O(N) for each flip? No, we need to update the whole C array?
    No, we just need to know if max |C| decreases.
    
    Let's stick to a simpler approach first: Randomized local search with restarts, but smarter.
    Or just use the "Model-and-optimize" inspiration idea:
    Use scipy.optimize.minimize on a continuous relaxation, then round.
    
    Let's try scipy.optimize.minimize with L-BFGS-B on x in [-1, 1].
    Then round to signs.
    """
    from scipy.optimize import minimize
    
    # Start from random signs or current x
    x0 = x.copy()
    
    def obj_and_grad(x_cont):
        # Objective: max |C| / S^2
        # This is non-smooth. Use soft-max or just gradient of max?
        # Gradient of max |C_k| is sign(C_k) * grad C_k at the max index.
        # Let's use a simple gradient descent step or trust region.
        
        s = np.sum(x_cont)
        if abs(s) < 1e-6:
            return 1e10, np.zeros_like(x_cont)
        
        C = _autoconv_fft(x_cont)
        # We want to minimize max_{l in 1..N-1} |C_l| / s^2
        # Let f = max |C_l| / s^2
        # grad f = (grad max |C_l| * s^2 - max |C_l| * 2 s grad s) / s^4
        #         = (grad max |C_l| / s^2 - 2 max |C_l| grad s / s^3)
        
        # Find max |C_l|
        idx_max = np.argmax(np.abs(C[1:N])) + 1
        C_max = C[idx_max]
        
        # Gradient of C_l wrt x:
        # C_l = sum_j x_j x_{j+l}
        # d C_l / d x_k = x_{k+l} + x_{k-l} (with boundary 0)
        
        # Construct gradient vector for C_l
        grad_C_l = np.zeros(N)
        # x_{k+l} term: for k=0..N-1-l, grad_C_l[k] += x[k+l]
        # x_{k-l} term: for k=l..N-1, grad_C_l[k] += x[k-l]
        
        # Vectorized:
        # Term 1: shift x by -l
        # grad_C_l[0:N-l] += x[l:N]
        # Term 2: shift x by l
        # grad_C_l[l:N] += x[0:N-l]
        
        grad_C_l[:N-idx_max] += x_cont[idx_max:]
        grad_C_l[idx_max:] += x_cont[:N-idx_max]
        
        # Sign of C_max
        sign_C = np.sign(C_max)
        
        # grad of max |C_l| is sign_C * grad_C_l
        grad_num = sign_C * grad_C_l
        grad_s = np.ones(N)
        
        val = abs(C_max) / (s*s)
        
        grad_den = 2 * s * grad_s
        
        grad_total = (grad_num * (s*s) - abs(C_max) * grad_den) / (s*s)**2
        # Simplify:
        # grad_total = grad_num / s^2 - 2 * abs(C_max) * grad_s / s^3
        
        return val, grad_total

    # Use L-BFGS-B
    # Constraints: x_i in [-1, 1]
    bounds = [(-1, 1)] * N
    
    # Run minimization
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 50, 'maxfun': 1000})
        x_opt = res.x
    except Exception:
        x_opt = x0

    # Round to signs
    x_rounded = np.sign(x_opt)
    # Handle zeros (should not happen often)
    x_rounded[x_rounded == 0] = 1.0
    
    return x_rounded

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Initial random start
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _objective(best_x)
    
    # Try a few random restarts with optimization
    num_restarts = 10
    
    for i in range(num_restarts):
        if _check_deadline():
            break
        
        # Random initialization
        x_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Optimize
        x_opt = _coordinate_descent(x_init)
        
        # Evaluate
        score = _objective(x_opt)
        
        if score < best_score:
            best_score = score
            best_x = x_opt.copy()
            
        # Local search polish on best_x
        # Simple flip-based local search
        improved = True
        while improved and not _check_deadline():
            improved = False
            for idx in range(N):
                if _check_deadline():
                    break
                old_val = best_x[idx]
                best_x[idx] = -old_val
                new_score = _objective(best_x)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_x[idx] = old_val
                    # If no improvement, revert
                    # Note: this loop is O(N) evaluations.
                    # We can break early if we find an improvement?
                    # Standard local search: try all, take best? Or first improvement?
                    # First improvement is faster.
                    # But here we are checking if *this specific* flip improves.
                    # If it does, we accept and restart the loop?
                    # Or just continue?
                    # Let's restart the loop to be safe (standard hill climbing).
                    # But that's expensive.
                    # Let's just do first improvement and break inner loop.
                    break
            
    return best_x
