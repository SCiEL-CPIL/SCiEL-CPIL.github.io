import numpy as np
import time
from scipy import signal

N = 400
TIME_LIMIT = 140  # seconds, with margin before 150s kill

def _autoconv(f):
    """Compute full autoconvolution."""
    return np.convolve(f, f, mode='full')

def _objective(f):
    """Compute max|autoconv| / (sum(f))^2. Return inf if sum is 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(f)
    return np.max(np.abs(c)) / (s * s)

def _objective_quick(f):
    """Quick objective for search - use FFT for speed."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    # FFT-based convolution
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, size=size)
    C = np.fft.ifft(F * F).real[:2*n-1]
    return np.max(np.abs(C)) / (s * s)

def _discrete_local_search(f, deadline, max_iters=5000):
    """Local search over sign flips for binary-valued sequences."""
    f = f.copy().astype(float)
    current_obj = _objective_quick(f)
    
    # Precompute contributions for fast evaluation
    n = len(f)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try flipping each element, keep best improvement
        best_gain = 0
        best_idx = -1
        
        # Sample a subset of indices for speed
        if n > 100 and it % 10 == 0:
            # Random subset
            indices = np.random.choice(n, size=min(n, 200), replace=False)
        else:
            indices = np.arange(n)
        
        for idx in indices:
            f[idx] = -f[idx]
            obj = _objective_quick(f)
            f[idx] = -f[idx]  # restore
            
            gain = current_obj - obj
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx >= 0:
            f[best_idx] = -f[best_idx]
            current_obj -= best_gain
        else:
            # No improvement found, break
            break
    
    return f, current_obj

def _real_local_search(f, deadline, max_iters=10000, step_sizes=None):
    """Local search over real-valued adjustments."""
    f = f.copy().astype(float)
    n = len(f)
    current_obj = _objective(f)
    
    if step_sizes is None:
        step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    
    # Try adjusting individual elements
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_change = None  # (idx, new_val)
        
        # Sample indices
        if n > 50:
            indices = np.random.choice(n, size=min(n, 100), replace=False)
        else:
            indices = np.arange(n)
        
        for idx in indices:
            for step in step_sizes:
                for sign in [1, -1]:
                    new_val = f[idx] + sign * step
                    f[idx] = new_val
                    obj = _objective(f)
                    f[idx] = f[idx]  # restore (already set)
                    
                    gain = current_obj - obj
                    if gain > best_gain:
                        best_gain = gain
                        best_change = (idx, new_val)
        
        if best_change is not None:
            idx, new_val = best_change
            f[idx] = new_val
            current_obj -= best_gain
        else:
            # Reduce step sizes
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 1e-6:
                break
    
    return f, current_obj

def _binary_to_real(f_bin, deadline):
    """Convert binary pattern to real-valued and optimize."""
    f = f_bin.astype(float)
    f, obj = _real_local_search(f, deadline)
    return f, obj

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Seed: all ones
    f_seed = np.ones(N)
    obj_seed = _objective(f_seed)
    best_f = f_seed.copy()
    best_obj = obj_seed
    
    # Phase 1: Try various binary initializations and discrete local search
    np.random.seed(12345)
    
    # Candidate binary patterns
    binary_candidates = []
    
    # All ones (seed)
    binary_candidates.append(np.ones(N))
    
    # Alternating
    binary_candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block patterns
    for block_size in [2, 4, 8, 16, 32, 64, 128]:
        pattern = np.ones(N)
        for start in range(0, N, block_size * 2):
            end = min(start + block_size, N)
            pattern[start:end] = -1
        binary_candidates.append(pattern.copy())
    
    # Random patterns
    for _ in range(5):
        binary_candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Phase 1: Discrete local search on binary patterns
    for f_bin in binary_candidates:
        if time.time() > deadline - 30:  # Reserve 30s for real search
            break
        
        # Quick check: if sum is 0, skip
        s = np.sum(f_bin)
        if abs(s) < 1e-10:
            continue
        
        f_opt, obj = _discrete_local_search(f_bin, deadline - 30)
        
        if obj < best_obj:
            best_obj = obj
            best_f = f_opt.copy()
    
    # Phase 2: Real-valued local search from best binary solution
    if time.time() < deadline - 10:
        f_real, obj_real = _real_local_search(best_f, deadline - 5)
        if obj_real < best_obj:
            best_obj = obj_real
            best_f = f_real
    
    # Phase 3: Try real-valued search from seed
    if time.time() < deadline - 5:
        f_real2, obj_real2 = _real_local_search(np.ones(N), deadline - 2)
        if obj_real2 < best_obj:
            best_obj = obj_real2
            best_f = f_real2
    
    # Final check
    if time.time() < deadline:
        final_obj = _objective(best_f)
        if final_obj < best_obj:
            best_obj = final_obj
    
    return best_f

# EVOLVE-BLOCK-END
