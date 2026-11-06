import numpy as np
import time
import math

def evaluate_max_norm(coeffs, grid_size=2048, refine_top_k=10, refine_iters=50):
    """
    Evaluate the maximum modulus of P(z) on |z|=1.
    coeffs: list of 24 coefficients in {+1, -1}
    Returns the maximum |P(z)|.
    """
    # Convert to numpy array for speed
    c = np.array(coeffs, dtype=np.float64)
    
    # 1. Dense sampling on a grid
    # z = exp(i * theta)
    # theta from 0 to 2*pi
    thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    
    # Efficient evaluation: P(e^{i theta}) = sum c_k e^{i k theta}
    # We can use matrix multiplication or FFT-like structure, but for N=24 and M=2048,
    # direct vectorized dot product is fast enough.
    # k = 0..23
    k = np.arange(24)
    # exp(i * theta_j * k_l) for all j, l
    # This creates a (2048, 24) matrix.
    # E = exp(1j * np.outer(thetas, k))
    # vals = E @ c
    # This is O(N*M). 24 * 2048 = ~50k operations. Very fast.
    
    # To save memory and time, we can compute in chunks or just do it directly.
    # Let's use the direct method.
    E = np.exp(1j * np.outer(thetas, k))
    vals = E @ c
    abs_vals = np.abs(vals)
    
    # Find peaks
    # A simple way to find peaks: local maxima in abs_vals
    # We need to be careful with circular boundary.
    # Pad with first and last element to handle wrap-around
    padded = np.concatenate(([abs_vals[-1]], abs_vals, [abs_vals[0]]))
    
    # Find indices where padded[i] is a local max
    # padded[i] > padded[i-1] and padded[i] > padded[i+1]
    # Use vectorized comparison
    is_peak = (padded[1:-1] > padded[:-2]) & (padded[1:-1] > padded[2:])
    peak_indices = np.where(is_peak)[0]
    
    if len(peak_indices) == 0:
        # Fallback to global max
        max_idx = np.argmax(abs_vals)
        peak_indices = np.array([max_idx])
    else:
        # Sort peaks by value descending
        peak_values = abs_vals[peak_indices]
        sort_idx = np.argsort(peak_values)[::-1]
        peak_indices = peak_indices[sort_idx]
        
    # Refine top K peaks using golden section search
    max_val = 0.0
    for idx in peak_indices[:refine_top_k]:
        # Interval around the grid point
        # The grid spacing is 2*pi / grid_size
        delta = 2 * np.pi / grid_size
        # Search in [theta_idx - delta, theta_idx + delta]
        # Actually, the peak is between theta_idx-1 and theta_idx+1 roughly.
        # Let's use the interval between the midpoints of adjacent grid points.
        # theta_left = (idx - 0.5) * delta
        # theta_right = (idx + 0.5) * delta
        theta_left = (idx - 0.5) * delta
        theta_right = (idx + 0.5) * delta
        
        # Golden section search for maximum of |P(e^{i theta})|
        # Function to evaluate magnitude at a specific theta
        def mag_at_theta(theta):
            # Reuse k
            z = np.exp(1j * theta)
            # P(z) = sum c_k z^k
            # Use Horner's rule for speed? Or vectorized?
            # Vectorized is fine for single point.
            return np.abs(np.sum(c * (z ** k)))
            
        # Golden section parameters
        gr = (math.sqrt(5) - 1) / 2 # 0.618
        a = theta_left
        b = theta_right
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        f_c = mag_at_theta(c_pt)
        f_d = mag_at_theta(d_pt)
        
        for _ in range(refine_iters):
            if f_c < f_d:
                a = c_pt
                c_pt = d_pt
                f_c = f_d
                d_pt = a + gr * (b - a)
                f_d = mag_at_theta(d_pt)
            else:
                b = d_pt
                d_pt = c_pt
                f_d = f_c
                c_pt = b - gr * (b - a)
                f_c = mag_at_theta(c_pt)
                
        # The max is at the midpoint of the final interval or the better of c, d
        f_mid = mag_at_theta((a+b)/2)
        local_max = max(f_c, f_d, f_mid)
        if local_max > max_val:
            max_val = local_max
            
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Initial sequence: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:24]
    best_val = evaluate_max_norm(best_coeffs)
    
    # Also try a few random starts
    np.random.seed(42)
    for _ in range(5):
        rand_coeffs = np.random.choice([-1, 1], size=24).tolist()
        val = evaluate_max_norm(rand_coeffs)
        if val < best_val:
            best_val = val
            best_coeffs = rand_coeffs
            
    # Local Search with Simulated Annealing
    # Parameters
    T = 0.05 # Initial temperature
    T_min = 0.001
    alpha = 0.99 # Cooling rate
    # We need to be careful with the step size of T.
    # The objective is max_norm. Lower is better.
    
    current_coeffs = best_coeffs[:]
    current_val = best_val
    
    # Pre-compute k for reuse? 
    # evaluate_max_norm recreates it. 
    # For speed in local search, we might want a faster evaluator that skips the full grid 
    # if a quick check shows it's bad, or use a coarser grid first.
    
    # Let's optimize the evaluation function for the inner loop.
    # A full grid of 2048 + refinement is ~1ms? 
    # 115s / 1ms = 115,000 evaluations.
    # Each eval tries 24 flips. So ~4000 full iterations.
    # This seems reasonable.
    
    # To speed up, we can use a smaller grid for the "trial" evaluation 
    # and only do full refinement if the trial value is promising.
    
    def fast_eval(coeffs):
        # Use smaller grid, no refinement, just max on grid
        c = np.array(coeffs, dtype=np.float64)
        thetas = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        k = np.arange(24)
        E = np.exp(1j * np.outer(thetas, k))
        vals = E @ c
        return np.max(np.abs(vals))
        
    # We will use fast_eval for the SA loop, and precise_eval for the best candidates.
    # However, fast_eval might miss peaks. 
    # Let's stick to the precise eval but optimize it.
    # The bottleneck is `np.exp(1j * np.outer(thetas, k))`.
    # We can pre-compute the basis vectors? No, thetas are fixed.
    # We can pre-compute the matrix E once!
    
    # Pre-compute E for the grid used in evaluate_max_norm
    # But evaluate_max_norm uses 2048.
    # Let's create a dedicated evaluator that uses pre-computed E.
    
    grid_size = 1024 # Smaller for speed in loop? 
    # Let's use 1024 for the main loop and refine.
    # If we use 1024, the grid error is ~ 2*pi/1024 ~ 0.006.
    # Refinement fixes this.
    
    thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    k = np.arange(24)
    E_grid = np.exp(1j * np.outer(thetas, k))
    
    def eval_precise_fast(coeffs):
        c = np.array(coeffs, dtype=np.float64)
        vals = E_grid @ c
        abs_vals = np.abs(vals)
        
        # Find peaks
        padded = np.concatenate(([abs_vals[-1]], abs_vals, [abs_vals[0]]))
        is_peak = (padded[1:-1] > padded[:-2]) & (padded[1:-1] > padded[2:])
        peak_indices = np.where(is_peak)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(abs_vals)])
        else:
            peak_values = abs_vals[peak_indices]
            sort_idx = np.argsort(peak_values)[::-1]
            peak_indices = peak_indices[sort_idx[:5]] # Refine top 5
            
        max_val = 0.0
        delta = 2 * np.pi / grid_size
        
        for idx in peak_indices:
            theta_left = (idx - 0.5) * delta
            theta_right = (idx + 0.5) * delta
            
            def mag_at_theta(theta):
                z = np.exp(1j * theta)
                return np.abs(np.sum(c * (z ** k)))
                
            gr = (math.sqrt(5) - 1) / 2
            a = theta_left
            b = theta_right
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            f_c = mag_at_theta(c_pt)
            f_d = mag_at_theta(d_pt)
            
            for _ in range(40): # Fewer iterations for speed
                if f_c < f_d:
                    a = c_pt
                    c_pt = d_pt
                    f_c = f_d
                    d_pt = a + gr * (b - a)
                    f_d = mag_at_theta(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    f_d = f_c
                    c_pt = b - gr * (b - a)
                    f_c = mag_at_theta(c_pt)
                    
            local_max = max(f_c, f_d)
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    # Re-initialize best with the faster evaluator
    best_val = eval_precise_fast(best_coeffs)
    current_coeffs = best_coeffs[:]
    current_val = best_val
    
    # Simulated Annealing
    T = 0.02
    T_min = 0.0001
    alpha = 0.995
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(0, 24)
        
        # Flip
        trial_coeffs = current_coeffs[:]
        trial_coeffs[i] = -trial_coeffs[i]
        
        trial_val = eval_precise_fast(trial_coeffs)
        
        delta_val = trial_val - current_val
        
        if delta_val < 0:
            # Improvement
            current_coeffs = trial_coeffs
            current_val = trial_val
            if current_val < best_val:
                best_val = current_val
                best_coeffs = current_coeffs[:]
        else:
            # Worse, accept with probability exp(-delta/T)
            if T > T_min:
                prob = math.exp(-delta_val / T)
                if np.random.random() < prob:
                    current_coeffs = trial_coeffs
                    current_val = trial_val
                    
        T *= alpha
        if T < T_min:
            T = T_min # Or restart? Let's just stay at min.
            
    return best_coeffs

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Initial sequence: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:24]
    
    # Also try a few random starts
    np.random.seed(42)
    for _ in range(5):
        rand_coeffs = np.random.choice([-1, 1], size=24).tolist()
        # We need an evaluator. Let's define it inside or use a helper.
        # To avoid code duplication, we define the evaluator logic here.
        
    # Define pre-computed structures
    grid_size = 1024
    thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    k = np.arange(24)
    E_grid = np.exp(1j * np.outer(thetas, k))
    
    def eval_precise_fast(coeffs):
        c = np.array(coeffs, dtype=np.float64)
        vals = E_grid @ c
        abs_vals = np.abs(vals)
        
        # Find peaks
        padded = np.concatenate(([abs_vals[-1]], abs_vals, [abs_vals[0]]))
        is_peak = (padded[1:-1] > padded[:-2]) & (padded[1:-1] > padded[2:])
        peak_indices = np.where(is_peak)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(abs_vals)])
        else:
            peak_values = abs_vals[peak_indices]
            sort_idx = np.argsort(peak_values)[::-1]
            peak_indices = peak_indices[sort_idx[:5]] # Refine top 5
            
        max_val = 0.0
        delta = 2 * np.pi / grid_size
        
        for idx in peak_indices:
            theta_left = (idx - 0.5) * delta
            theta_right = (idx + 0.5) * delta
            
            def mag_at_theta(theta):
                z = np.exp(1j * theta)
                return np.abs(np.sum(c * (z ** k)))
                
            gr = (math.sqrt(5) - 1) / 2
            a = theta_left
            b = theta_right
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            f_c = mag_at_theta(c_pt)
            f_d = mag_at_theta(d_pt)
            
            for _ in range(40): # Fewer iterations for speed
                if f_c < f_d:
                    a = c_pt
                    c_pt = d_pt
                    f_c = f_d
                    d_pt = a + gr * (b - a)
                    f_d = mag_at_theta(d_pt)
                else:
                    b = d_pt
                    d_pt = c_pt
                    f_d = f_c
                    c_pt = b - gr * (b - a)
                    f_c = mag_at_theta(c_pt)
                    
            local_max = max(f_c, f_d)
            if local_max > max_val:
                max_val = local_max
                
        return max_val

    # Evaluate initial best
    best_val = eval_precise_fast(best_coeffs)
    
    # Try random starts
    for _ in range(10):
        rand_coeffs = np.random.choice([-1, 1], size=24).tolist()
        val = eval_precise_fast(rand_coeffs)
        if val < best_val:
            best_val = val
            best_coeffs = rand_coeffs
            
    # Local Search with Simulated Annealing
    current_coeffs = best_coeffs[:]
    current_val = best_val
    
    T = 0.02
    T_min = 0.0001
    alpha = 0.995
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(0, 24)
        
        # Flip
        trial_coeffs = current_coeffs[:]
        trial_coeffs[i] = -trial_coeffs[i]
        
        trial_val = eval_precise_fast(trial_coeffs)
        
        delta_val = trial_val - current_val
        
        if delta_val < 0:
            # Improvement
            current_coeffs = trial_coeffs
            current_val = trial_val
            if current_val < best_val:
                best_val = current_val
                best_coeffs = current_coeffs[:]
        else:
            # Worse, accept with probability exp(-delta/T)
            if T > T_min:
                prob = math.exp(-delta_val / T)
                if np.random.random() < prob:
                    current_coeffs = trial_coeffs
                    current_val = trial_val
                    
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_coeffs
# EVOLVE-BLOCK-END
