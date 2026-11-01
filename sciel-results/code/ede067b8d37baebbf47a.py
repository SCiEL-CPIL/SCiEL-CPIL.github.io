import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 115.0  # Safety margin before 150s limit

def _compute_c3(h):
    """
    Compute the C3 upper bound: max_k |conv(h,h)[k]| / (sum_i |h_i|)^2
    Uses FFT for efficient autoconvolution.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-12:
        return float('inf')
    
    return np.max(np.abs(ac)) / (sum_abs_h ** 2)

def _objective(x, beta=50.0):
    """
    Smooth surrogate for C3 upper bound using tanh mapping and soft-max.
    x is unconstrained; h = tanh(x) maps to (-1,1).
    """
    h = np.tanh(x)
    
    # Autoconvolution via FFT
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    # Soft-max of |ac| with temperature 1/beta
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        soft_max = max_val
    else:
        shifted = beta * (abs_ac - max_val)
        log_sum = np.log(np.sum(np.exp(shifted))) / beta
        soft_max = max_val + log_sum
    
    # Smooth sum of |h_i|
    eps = 1e-12
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    return soft_max / (sum_abs_h ** 2)

def _gradient(x, beta=50.0):
    """
    Numerical gradient of the objective using central differences.
    For N=400, this is 800 FFT evaluations - too slow.
    Use a cheaper approach: gradient via finite difference on a subset.
    """
    # Actually, let's use a simpler approach: just use numerical gradient
    # but with a step size that balances accuracy and speed
    # For the continuous relaxation phase, we'll use a moderate number of iterations
    
    eps = 1e-5
    grad = np.zeros(N)
    f0 = _objective(x, beta)
    
    # Use forward differences to save half the evaluations
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective(x_plus, beta)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def _local_search_discrete(h, max_iterations=50000, time_limit=10.0):
    """
    Fast discrete local search with simulated annealing.
    Uses incremental autocorrelation update for speed.
    """
    start_time = time.time()
    
    # Precompute initial autocorrelation
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)
    current_val = _compute_c3(h)
    
    best_h = h.copy()
    best_val = current_val
    
    T0 = 0.05
    cooling_rate = 0.9999
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Flip a random bit
        i = np.random.randint(N)
        old_sign = h[i]
        h[i] = -old_sign
        
        # Update autocorrelation incrementally
        # ac_new[k] = ac_old[k] - 2*old_sign*(h[(k-i) mod N] + h[(i-k) mod N]) ... 
        # Actually for linear convolution, flipping h[i] changes:
        # conv[k] changes by -2*old_sign*(h[k-i] + h[i-k]) for valid indices
        # This is complex with boundaries. Let's just recompute for simplicity
        # but limit iterations.
        
        new_val = _compute_c3(h)
        delta = new_val - current_val
        
        T = T0 * (cooling_rate ** it)
        
        if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            h[i] = old_sign
    
    return best_h, best_val

def _continuous_relaxation(h_init, max_iter=200, time_limit=15.0):
    """
    Continuous relaxation using L-BFGS-B with numerical gradients.
    Maps from unconstrained space to [-1,1] via tanh.
    """
    start_time = time.time()
    
    # Initialize x such that tanh(x) ≈ h_init
    # tanh(x) = h => x = arctanh(h), but clip to avoid infinities
    h_clipped = np.clip(h_init, -0.99, 0.99)
    x0 = np.arctanh(h_clipped)
    
    def obj_and_grad(x):
        f = _objective(x)
        g = _gradient(x)
        return f, g
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6,
                                'maxls': 20})
    except Exception:
        # Fallback: just use the initial
        return h_init, _compute_c3(h_init)
    
    h_opt = np.tanh(res.x)
    # Round to binary for discrete evaluation
    h_bin = np.sign(h_opt)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        h_bin[zero_mask] = 1.0
    
    val_bin = _compute_c3(h_bin)
    
    # Also evaluate the continuous solution
    val_cont = _compute_c3(h_opt)
    
    # Return the better of binary and continuous
    if val_bin <= val_cont:
        return h_bin, val_bin
    else:
        return h_opt, val_cont

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # All ones (seed)
    initializations.append(np.ones(N))
    
    # Alternating
    initializations.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Random binary
    for _ in range(8):
        initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Block patterns with various block sizes
    for block_size in [4, 8, 16, 20, 25, 40, 50]:
        n_blocks = N // block_size
        signs = np.random.choice([-1.0, 1.0], size=n_blocks)
        pattern = np.repeat(signs, block_size)[:N]
        initializations.append(pattern)
    
    # Sine-based patterns
    for freq in [1, 2, 3, 5, 8, 10]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        initializations.append(np.sign(np.sin(x)))
    
    # Phase 2: Continuous relaxation from multiple starts
    num_relaxations = min(10, len(initializations))
    
    for i in range(num_relaxations):
        if time.time() >= deadline - 25.0:
            break
        
        h_init = initializations[i].copy()
        remaining = deadline - time.time() - 20.0
        if remaining < 3.0:
            break
        
        h_rel, val_rel = _continuous_relaxation(h_init, max_iter=150, time_limit=min(12.0, remaining))
        
        if val_rel < best_val:
            best_val = val_rel
            best_h = h_rel.copy()
    
    # Phase 3: Discrete refinement of the best continuous solution
    if best_h is not None and time.time() < deadline - 15.0:
        # Ensure we're working with a binary solution for discrete search
        h_bin = np.sign(best_h)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = 1.0
        
        val_bin = _compute_c3(h_bin)
        if val_bin < best_val:
            best_val = val_bin
            best_h = h_bin.copy()
        
        # Discrete local search
        remaining = deadline - time.time() - 5.0
        if remaining > 3.0:
            h_ds, val_ds = _local_search_discrete(best_h.copy(), 
                                                   max_iterations=100000, 
                                                   time_limit=min(10.0, remaining))
            if val_ds < best_val:
                best_val = val_ds
                best_h = h_ds.copy()
    
    # Phase 4: Try a few more discrete local searches from different starts
    if time.time() < deadline - 5.0:
        # Use the best found and a couple of random starts
        search_starts = [best_h.copy() if best_h is not None else np.ones(N)]
        search_starts.append(np.random.choice([-1.0, 1.0], size=N))
        
        for h_start in search_starts:
            if time.time() >= deadline - 3.0:
                break
            
            remaining = deadline - time.time() - 2.0
            if remaining < 2.0:
                break
            
            h_ls, val_ls = _local_search_discrete(h_start.copy(),
                                                   max_iterations=50000,
                                                   time_limit=min(5.0, remaining))
            if val_ls < best_val:
                best_val = val_ls
                best_h = h_ls.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
