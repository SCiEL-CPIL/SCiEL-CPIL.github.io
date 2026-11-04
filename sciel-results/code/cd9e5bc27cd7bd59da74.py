"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
import scipy.fft as fft

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.fft as fft

def compute_objective(A):
    """
    Compute the value of the objective function:
    max |A * A| / (sum A)^2
    We assume A is normalized such that sum A = 1 for optimization,
    but the ratio is scale invariant.
    """
    if np.allclose(A, 0):
        return 1e9
    # Autoconvolution
    # Use FFT for efficiency
    # Length of convolution is 2*N - 1
    size = 2 * N
    A_fft = fft.fft(A, n=size)
    C_fft = A_fft * A_fft
    C = fft.ifft(C_fft).real[:2*N-1]
    
    max_abs_conv = np.max(np.abs(C))
    sum_A = np.sum(A)
    
    if abs(sum_A) < 1e-12:
        return 1e9
        
    return max_abs_conv / (sum_A * sum_A)

def normalize_A(A):
    s = np.sum(A)
    if abs(s) < 1e-12:
        return A
    return A / s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_A = np.ones(N)
    best_val = compute_objective(best_A)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline
    
    # Generate a few diverse initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Random signs
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], size=N))
        
    # 3. Random continuous
    for _ in range(5):
        candidates.append(np.random.uniform(-1, 1, size=N))
        
    # 4. Structured: Alternating signs (might cancel out)
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 5. Structured: Blocks of same sign
    for block_size in [2, 4, 8, 16]:
        signs = []
        for i in range(0, N, block_size):
            s = 1.0 if (i // block_size) % 2 == 0 else -1.0
            signs.extend([s] * min(block_size, N - i))
        candidates.append(np.array(signs))

    # Optimize each candidate
    for i, A0 in enumerate(candidates):
        if time_up():
            break
            
        A0 = A0.copy()
        # Normalize to sum 1 to help optimization landscape
        A0 = normalize_A(A0)
        
        # Define the function to minimize
        # We minimize max |C| subject to sum A = 1
        # Powell doesn't handle constraints directly well, so we use a penalty or projection.
        # Simple projection: after each step, normalize.
        # But Powell works on the raw vector.
        # Let's use a wrapper that normalizes.
        
        def obj_func(x):
            # Normalize x to sum 1
            s = np.sum(x)
            if abs(s) < 1e-12:
                return 1e9
            x_norm = x / s
            return compute_objective(x_norm)
        
        try:
            # Use Powell method
            # maxiter is a soft limit, we check time in the callback if possible, 
            # but Powell doesn't support callbacks easily in all versions.
            # We rely on the time check between candidates and a reasonable maxiter.
            # N=400, Powell might take a while. Let's limit iterations per candidate.
            res = opt.minimize(obj_func, A0, method='Powell', options={'maxiter': 50, 'maxfev': 2000})
            
            if time_up():
                break
                
            if res.success or res.fun < best_val:
                A_opt = normalize_A(res.x)
                val = compute_objective(A_opt)
                if val < best_val:
                    best_val = val
                    best_A = A_opt
        except Exception as e:
            pass
            
    # If we have time left, do a few more random restarts
    while not time_up():
        A0 = np.random.uniform(-1, 1, size=N)
        A0 = normalize_A(A0)
        
        def obj_func(x):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return 1e9
            x_norm = x / s
            return compute_objective(x_norm)
            
        try:
            # Smaller budget for random restarts
            res = opt.minimize(obj_func, A0, method='Powell', options={'maxiter': 20, 'maxfev': 500})
            
            if time_up():
                break
                
            if res.fun < best_val:
                A_opt = normalize_A(res.x)
                val = compute_objective(A_opt)
                if val < best_val:
                    best_val = val
                    best_A = A_opt
        except Exception:
            break

    return best_A
# EVOLVE-BLOCK-END
