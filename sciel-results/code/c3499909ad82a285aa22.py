import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # We search in the space of non-negative vectors.
    # Since the ratio is scale-invariant, we can normalize ||h||_1 = 1 after each step.
    # We use a simple stochastic hill-climbing / simulated annealing approach.
    
    best_h = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Initialize with a random sparse-ish vector
    # Start with a small random support
    initial_support_size = 10
    init_h = np.zeros(N)
    init_indices = rng.choice(N, size=initial_support_size, replace=False)
    init_h[init_indices] = 1.0 / initial_support_size
    
    current_h = init_h.copy()
    current_score = compute_ratio(current_h)
    
    if current_score > best_score:
        best_score = current_score
        best_h = current_h.copy()
        
    # Temperature for simulated annealing
    # We want to accept worse moves early on to escape local minima
    T_start = 0.1
    T_end = 1e-6
    
    # Number of steps
    # We estimate how many steps we can do.
    # compute_ratio is O(N log N) due to FFT. N=50 is very small.
    # Let's try 50,000 steps.
    max_steps = 100000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Cooling schedule
        progress = step / max_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a perturbation
        # We perturb a few random indices
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        new_h = current_h.copy()
        
        # Add noise
        noise = rng.normal(0, 0.1, size=num_perturb)
        # Scale noise relative to current magnitude? 
        # Since we normalize, absolute scale matters less, but relative structure matters.
        # Let's add noise proportional to the mean weight of the support to avoid blowing up.
        # Actually, just adding small absolute noise works if we normalize afterwards.
        
        for i, idx in enumerate(indices):
            new_h[idx] += noise[i]
            
        # Enforce non-negativity
        new_h = np.maximum(new_h, 0.0)
        
        # Normalize to ||h||_1 = 1
        l1_new = np.sum(new_h)
        if l1_new < 1e-10:
            # If it became zero, restart this step
            continue
            
        new_h /= l1_new
        
        new_score = compute_ratio(new_h)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Always accept better or equal
            current_h = new_h
            current_score = new_score
        else:
            # Accept worse with probability exp((new - current) / T)
            # Note: we are maximizing, so delta = new - current is negative
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_h = new_h
                    current_score = new_score
                    
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # Occasionally, try a more drastic restart if stuck?
        # The SA cooling should handle exploration.
        
    # Final Polish: 
    # The SA might stop at a point where small perturbations don't help.
    # We can try to "prune" the support: 
    # If a weight is very small, setting it to 0 might increase the ratio 
    # (reduces L2^2 slightly, but might reduce L1*||hh||inf more if it contributed to peak).
    # Actually, removing a small weight reduces L1 and L2^2. 
    # Ratio R = L2^2 / (L1 * Peak).
    # If we remove h_i:
    # L1' = L1 - h_i
    # L2'^2 = L2^2 - h_i^2
    # Peak might decrease or stay same.
    # It's complex. Let's just do a local coordinate descent on the weights for the final vector.
    
    # Local Search on Weights
    # Fix the support of best_h
    support = np.where(best_h > 1e-8)[0]
    if len(support) == 0:
        return list(best_h)
        
    w = best_h[support].copy()
    current_w_score = best_score
    
    # Simple coordinate ascent on weights
    # We want to maximize R(w). 
    # Since R is scale invariant, we can fix sum(w)=1.
    # We update one weight at a time, adjusting another to keep sum=1? 
    # Or just update one and renormalize.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(w)):
            # Try to increase w_i
            # This requires decreasing others to keep sum=1.
            # Let's just do a small perturbation on the full vector restricted to support
            # and renormalize.
            
            # Actually, simpler: try moving mass from i to j
            # Or just random perturbation on the sub-vector w
            w_new = w.copy()
            idx = rng.integers(0, len(w))
            w_new[idx] += rng.normal(0, 0.05)
            w_new = np.maximum(w_new, 0)
            s = np.sum(w_new)
            if s < 1e-10:
                continue
            w_new /= s
            
            # Evaluate
            h_test = np.zeros(N)
            h_test[support] = w_new
            s_test = compute_ratio(h_test)
            
            if s_test > current_w_score:
                w = w_new
                current_w_score = s_test
                improved = True
        
        if not improved:
            # If no improvement in a full pass, break
            # Or continue with smaller steps? 
            # Let's break to save time
            pass # We can do more passes if time permits, but 1000 is plenty for 50D
        
    if current_w_score > best_score:
        best_h = np.zeros(N)
        best_h[support] = w
        best_score = current_w_score
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
