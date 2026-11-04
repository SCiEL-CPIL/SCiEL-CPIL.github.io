"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _blocks_to_signal(params):
    """
    Convert block parameters to a signal of length N.
    params: array of shape (K, 3) where columns are [start, width, amplitude].
    start: continuous index (can be fractional)
    width: continuous width (can be fractional)
    amplitude: real value
    """
    signal = np.zeros(N)
    # Vectorized accumulation is tricky for overlapping blocks with fractional edges.
    # Given small K (e.g., 5-10), a loop is fast enough.
    for k in range(len(params)):
        start, width, amp = params[k]
        if width <= 0:
            continue
        # Determine integer indices affected
        # Block covers [start, start + width]
        # Indices i such that i < start + width and i >= start
        # i_min = ceil(start), i_max = floor(start + width - epsilon)
        
        # More robust: accumulate contribution of each index
        # Since K is small, we can just check each index? No, N=400 is small, K is small.
        # But we want speed. 
        # Let's use the fact that blocks are wide.
        
        # Fractional edge 1: start
        # Fractional edge 2: end = start + width
        
        i_start = int(np.ceil(start))
        i_end = int(np.floor(start + width))
        
        # Indices fully inside: [i_start, i_end - 1] ? 
        # Let's be precise.
        # Contribution at index i is amp * length(intersection([i, i+1], [start, start+width]))
        
        # This is effectively a convolution of a delta comb with a rectangular pulse.
        # For each block, we add amp * (overlap).
        
        # Indices potentially affected: floor(start) to ceil(start+width)
        idx_min = max(0, int(np.floor(start)))
        idx_max = min(N - 1, int(np.ceil(start + width)))
        
        if idx_min > idx_max:
            continue
            
        indices = np.arange(idx_min, idx_max + 1)
        
        # Calculate overlap for each index
        # Interval for index i is [i, i+1]
        # Interval for block is [start, start+width]
        
        # left_bound = max(i, start)
        # right_bound = min(i+1, start+width)
        # overlap = max(0, right_bound - left_bound)
        
        left_bounds = np.maximum(indices.astype(float), start)
        right_bounds = np.minimum((indices + 1).astype(float), start + width)
        overlaps = np.maximum(0, right_bounds - left_bounds)
        
        signal[indices] += amp * overlaps
        
    return signal

def _evaluate_params(params):
    """Evaluate the metric for a set of block parameters."""
    x = _blocks_to_signal(params)
    return _metric(x)

def _differential_evolution_block(deadline, rng, initial_params, pop_size=50, max_iters=200):
    """
    Custom Differential Evolution for the block parameters.
    """
    K = len(initial_params)
    dim = 3 * K # start, width, amp for each block
    
    # Initialize population
    # We need to ensure validity: width > 0, start in [0, N], amp in [-2, 2]
    # We'll use a bounded DE.
    
    # Bounds for DE
    # start: [0, N]
    # width: [0.5, N/2] (min width 0.5 to avoid zero contribution, max N/2 to allow overlap)
    # amp: [-2.0, 2.0]
    
    bounds = []
    for _ in range(K):
        bounds.append((0.0, float(N)))      # start
        bounds.append((0.5, float(N)/2))    # width
        bounds.append((-2.0, 2.0))          # amp
    
    # Initialize population
    pop = np.zeros((pop_size, dim))
    for i in range(pop_size):
        for j in range(dim):
            lo, hi = bounds[j]
            pop[i, j] = rng.uniform(lo, hi)
    
    # Evaluate initial population
    pop_fitness = np.array([_evaluate_params(pop[i].reshape(K, 3)) for i in range(pop_size)])
    
    # Sort by fitness (lower is better)
    sort_idx = np.argsort(pop_fitness)
    pop = pop[sort_idx]
    pop_fitness = pop_fitness[sort_idx]
    
    best_params = pop[0].copy()
    best_fit = pop_fitness[0]
    
    # DE Parameters
    F = 0.8  # Scale factor
    CR = 0.9 # Crossover rate
    
    iters_done = 0
    while iters_done < max_iters:
        if time.time() > deadline:
            break
            
        # Generate mutant vector for each individual
        new_pop = pop.copy()
        new_fitness = pop_fitness.copy()
        
        for i in range(pop_size):
            # Select 3 distinct random indices different from i
            mask = np.ones(pop_size, dtype=bool)
            mask[i] = False
            random_idx = rng.choice(np.where(mask)[0], size=3, replace=False)
            a, b, c = random_idx
            
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Clip to bounds
            for j in range(dim):
                lo, hi = bounds[j]
                mutant[j] = np.clip(mutant[j], lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Evaluate trial
            fit_trial = _evaluate_params(trial.reshape(K, 3))
            
            if fit_trial < pop_fitness[i]:
                pop[i] = trial
                pop_fitness[i] = fit_trial
                
                if fit_trial < best_fit:
                    best_fit = fit_trial
                    best_params = trial.copy()
            
            iters_done += 1
            # Check time frequently in inner loop if pop_size is large
            if time.time() > deadline:
                break
        
    return best_params, best_fit

def _local_search_grid(x, deadline, rng, steps=20):
    """
    Refine the discrete signal by trying small shifts of contiguous blocks.
    This acts as a "coarse-to-fine" polish on the discrete result.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Identify "blocks" in the current signal? 
    # Or just try shifting random segments?
    # Let's try shifting random contiguous segments of length L by +/- 1.
    
    for _ in range(steps):
        if time.time() > deadline:
            break
            
        # Pick a random segment
        start = rng.integers(0, N - 10)
        length = rng.integers(5, 50)
        if start + length > N:
            continue
            
        # Try shifting right by 1
        x_test = best_x.copy()
        x_test[start:start+length] = 0
        # Shift values
        # This is complex if we just move values. 
        # The parametric approach is better for this. 
        # So this local search is a bit redundant if we trust the DE result.
        # Let's do a simple sign flip polish instead, it's safer.
        
    return best_x, best_val

def _construct():
    deadline = time.time() + 145.0
    
    rng = np.random.default_rng(42)
    
    # Start with a random baseline (binary)
    best_x = rng.integers(0, 2, size=N).astype(float) * 2 - 1
    best_val = _metric(best_x)
    
    # Strategy: Parametric Block Optimization
    # Try different numbers of blocks K
    # K=1 is trivial (rect pulse), K=2, K=3, K=5
    
    # We will run DE for each K, using the remaining time budget.
    # DE is stochastic, so we run it for a fixed number of iterations or time.
    
    # Let's allocate time to K=2, K=3, K=5.
    # K=5 is likely the sweet spot for N=400.
    
    # To save time, we'll run DE for K=5 first, then K=3, then K=2 if time permits.
    # Or run them in parallel? No, single thread.
    
    # Let's run DE for K=5 with a generous budget.
    K_list = [5, 3, 2]
    
    for K in K_list:
        if time.time() > deadline - 10:
            break
            
        # Initialize params for this K
        # Random init
        init_params = np.zeros((K, 3))
        for k in range(K):
            init_params[k, 0] = rng.uniform(0, N) # start
            init_params[k, 1] = rng.uniform(10, N/4) # width
            init_params[k, 2] = rng.uniform(-1, 1) # amp
            
        # Run DE
        # Time budget: 40s per K? 
        # We have ~140s total. 
        # Let's give 40s to K=5, 30s to K=3, 20s to K=2.
        
        if K == 5:
            time_budget = 40.0
        elif K == 3:
            time_budget = 30.0
        else:
            time_budget = 20.0
            
        de_deadline = time.time() + time_budget
        
        best_params, best_fit = _differential_evolution_block(
            de_deadline, rng, init_params, pop_size=30, max_iters=1000
        )
        
        # Convert best params to signal
        x_candidate = _blocks_to_signal(best_params)
        
        # Check if it's better
        if best_fit < best_val:
            best_val = best_fit
            best_x = x_candidate
            # Print for debugging? No, just track.
            
    # Final Polish: Sign Flip Local Search on the best discrete signal found
    # This can fix edge effects of the continuous block approximation.
    if time.time() < deadline - 5:
        # Simple sign flip search
        polish_deadline = deadline - 2.0
        no_improve = 0
        while time.time() < polish_deadline and no_improve < 200:
            idx = rng.integers(0, N)
            x_test = best_x.copy()
            x_test[idx] *= -1
            val_test = _metric(x_test)
            if val_test < best_val:
                best_val = val_test
                best_x = x_test
                no_improve = 0
            else:
                no_improve += 1

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
