import numpy as np
import time
from scipy.optimize import differential_evolution

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    # Bounds for differential evolution
    # We restrict the search to a reasonable range to help convergence
    # Since the score is scale-invariant (max_conv / sum^2), we can fix the scale
    # or let the optimizer find it. Fixing scale is often more efficient.
    # Let's try fixing the L1 norm or L2 norm?
    # The objective is homogeneous of degree 0.
    # However, DE works in a box.
    # Let's use bounds [0, 1] for all elements.
    # To prevent the trivial solution f=0, we can normalize inside the objective
    # or rely on the fact that max_conv/sum^2 is well-defined for f != 0.
    # If f is very small, numerical issues might arise, but DE usually explores
    # the bulk of the box.
    
    # Actually, a better approach for homogeneous objectives is to fix the norm.
    # But DE doesn't handle constraints easily.
    # Let's stick to box [0, 1] and normalize inside the objective if sum is too small?
    # No, the task says "minimize the normalized maximum".
    # score = max(f*f) / (sum(f))^2.
    # This is invariant to scaling f by c > 0.
    # So we can fix sum(f) = 1, for example.
    # But DE optimizes over a box.
    # If we fix sum(f)=1, we are on a simplex.
    # We can parameterize by f_i >= 0 and normalize at the end of the objective eval.
    # If we normalize, the landscape is flat along the scaling direction.
    # DE might wander along this flat direction, wasting evaluations.
    # Better: Fix the L2 norm to 1? Or just use [0,1] and accept the scaling invariance.
    # Let's try [0, 1] bounds.
    
    bounds = [(0.0, 1.0)] * n

    def objective(f_vec):
        # f_vec is a numpy array of shape (n,)
        # Ensure non-negative (bounds handle this, but just in case)
        f_vec = np.maximum(f_vec, 0.0)
        
        sum_f = np.sum(f_vec)
        if sum_f < 1e-12:
            return 1.0  # Penalize zero solution
            
        # Compute autoconvolution
        conv = _autoconv(f_vec)
        max_conv = np.max(conv)
        
        # Normalized score
        score = max_conv / (sum_f * sum_f)
        
        return score

    # Initial population size
    # DE population size is typically 5-15 times the dimension.
    # n=600 is large. popsize=5 -> 3000 individuals.
    # Each eval takes ~0.5ms (FFT of size 1200).
    # 3000 evals * 0.5ms = 1.5s per generation.
    # In 145s, we can do ~90 generations.
    # This is a reasonable number for global search.
    
    # Use a smaller popsize to allow more generations or faster per-gen?
    # popsize=5 is standard. Let's try popsize=4 or 5.
    # maxiter: We don't want to hit maxiter if time is left.
    # We'll use a callback to stop if time runs out.
    
    stop_flag = [False]
    
    def callback(xk, convergence):
        if time_remaining() < 10.0:
            stop_flag[0] = True
            return True
        return False

    # Strategy:
    # 1. Run DE with a moderate population and maxiter set high, relying on callback.
    # 2. To speed up, we can use a smaller population and more iterations?
    #    DE convergence depends on population diversity.
    #    Let's use popsize=5.
    
    # Initial seed
    rng = np.random.default_rng(42)
    
    # Run Differential Evolution
    # tol: 1e-6
    # maxiter: 1000 (will be stopped by callback)
    # polish: False (we don't need local polish, we can do it separately if time permits)
    # init: 'latinhypercube' for better spread
    
    try:
        res = differential_evolution(
            objective,
            bounds,
            seed=42,
            popsize=5,
            maxiter=1000,
            tol=1e-5,
            init='latinhypercube',
            polish=False,
            callback=callback,
            updating='immediate'
        )
        
        best_f = res.x
        best_score = res.fun
        
    except Exception as e:
        # Fallback to a simple random search if DE fails
        best_f = None
        best_score = float('inf')
        
        for _ in range(1000):
            if time_remaining() < 5.0:
                break
            f_rand = rng.random(n)
            s = _compute_score(f_rand)
            if s < best_score:
                best_score = s
                best_f = f_rand

    # If DE didn't find anything good, use a deterministic fallback
    if best_f is None or best_score > 1.0:
        # Use a smooth bump as fallback
        x = np.linspace(-0.25, 0.25, n, endpoint=False)
        best_f = np.exp(-x**2 / (2 * 0.08**2))
        best_score = _compute_score(best_f)

    # Optional: Local refinement if time permits
    if time_remaining() > 20.0:
        # Simple local search: coordinate descent with small steps
        # This might not be necessary if DE is good, but let's try a few iterations
        # of a simple gradient-free local search (Nelder-Mead is too slow for 600 dims)
        # Instead, we can try a few random perturbations around the best point
        
        current_f = best_f.copy()
        current_score = best_score
        
        for _ in range(50):
            if time_remaining() < 10.0:
                break
            
            # Random perturbation
            noise = rng.normal(0, 0.01, n)
            test_f = np.maximum(current_f + noise, 0.0)
            test_score = _compute_score(test_f)
            
            if test_score < current_score:
                current_f = test_f
                current_score = test_score
        
        best_f = current_f
        best_score = current_score

    # Final check
    best_f = np.maximum(best_f, 0.0)
    
    # Ensure the function is not all zeros
    if np.all(best_f == 0):
        best_f = np.ones(n)
        
    return list(best_f)
