"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv

def _soft_max_obj_and_grad(x, beta=100.0):
    """
    Compute soft-max objective and gradient.
    Objective: 2*N*softmax(conv) / (sum(h))^2
    where softmax(conv) = logsumexp(beta*conv)/beta
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv_fft(h)
    m = np.max(conv)
    
    # Compute logsumexp with numerical stability
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum_exp = m + np.log(np.sum(exp_shifted)) / beta
    
    # Value: 2*N*log_sum_exp / s^2
    val = 2.0 * N * log_sum_exp / (s * s)
    
    # Gradient computation
    # d(log_sum_exp)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # d(conv[k])/d(h[j]) = 2*h[k-j] (with h[i]=0 outside [0, N-1])
    # grad_logsumexp[j] = 2 * sum_{k=0}^{2N-2} weights[k] * h[k-j]
    # Note: h[k-j] is non-zero only if 0 <= k-j < N => j <= k < j+N
    
    # We want c[j] = sum_{k} weights[k] * h[k-j] for j=0..N-1
    # This is the cross-correlation of weights and h, but shifted.
    # Let's use the sliding window trick for efficiency and correctness.
    
    # weights has length 2N-1.
    # For a given j, we need weights[j : j+N] dot h[0 : N].
    
    # Using stride tricks to create a view of weights with shape (N, N)
    # where row j is weights[j : j+N]
    from numpy.lib.stride_tricks import as_strided
    
    # Ensure weights is contiguous for stride tricks to work safely
    weights = np.ascontiguousarray(weights)
    
    # Create the sliding window view
    # Shape: (N, N)
    # Strides: (itemsize, itemsize)
    w_sliding = as_strided(weights, shape=(N, N), strides=(weights.strides[0], weights.strides[0]))
    
    # grad_logsumexp[j] = 2 * (w_sliding[j] @ h)
    grad_logsumexp = 2.0 * (w_sliding @ h)
    
    # Now, d(val)/d(h[j]) = 2*N/s^2 * grad_logsumexp[j] + 2*N*log_sum_exp * d(1/s^2)/d(h[j])
    # d(1/s^2)/d(h[j]) = -2/s^3
    # grad_val = 2*N/s^2 * grad_logsumexp[j] - 4*N*log_sum_exp/s^3
    # grad_val = (2*N/s^2) * (grad_logsumexp[j] - 2*log_sum_exp/s)
    
    grad_val = (2.0 * N / (s * s)) * (grad_logsumexp - 2.0 * log_sum_exp / s)
    
    return val, grad_val

def _objective_soft_max(x, beta=100.0):
    """Objective function for scipy minimize."""
    val, _ = _soft_max_obj_and_grad(x, beta)
    return val

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations focusing on low-DC / zero-sum components
    initializations = []
    
    # 1. Random normal with mean shifted to near zero
    for _ in range(10):
        h0 = rng.standard_normal(N)
        h0 -= np.mean(h0) # Enforce zero mean
        # Add a small DC component to avoid sum=0 exactly initially, 
        # but the optimizer will scale it. Actually, we want to explore the 
        # regime where sum(h) is small relative to peak conv?
        # The objective is scale invariant: f(ah) = f(h).
        # So we can fix sum(h) = 1 or similar during optimization?
        # No, the gradient handles the scaling.
        # Let's just use zero-mean starts.
        if abs(np.sum(h0)) < 1e-10:
             h0[0] += 1e-6
        initializations.append(h0)
        
    # 2. Comb patterns (alternating signs) - naturally low DC
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.05 * rng.standard_normal(N)
        initializations.append(h0)
        
    # 3. Random Walk with Drift (as per directive)
    # h[i] = sum_{j=0}^i (noise[j] - drift)
    # This creates a signal that wanders but doesn't explode.
    for _ in range(10):
        noise = rng.standard_normal(N)
        drift = 0.1
        steps = noise - drift
        h0 = np.cumsum(steps)
        # Normalize to have reasonable magnitude
        h0 = h0 / np.std(h0) if np.std(h0) > 1e-10 else h0
        initializations.append(h0)
        
    # 4. Sine waves (low frequency)
    for freq in [1, 2, 3, 5]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)

    # Optimization Strategy: Simulated Annealing on Beta (Temperature)
    # We start with low beta (smooth landscape) and increase beta (sharper landscape)
    # At each step, we run a few iterations of L-BFGS-B.
    # We accept worse solutions with probability exp(-delta/T) to escape local minima.
    
    # Temperature schedule for the annealing process (not the beta, but the acceptance temp)
    # And we also ramp up beta to sharpen the max.
    
    # Let's define a schedule of (beta, anneal_temp)
    # beta: sharpness of soft-max
    # anneal_temp: temperature for Metropolis acceptance
    
    schedule = []
    # Phase 1: Broad exploration
    for beta in [10.0, 20.0, 40.0]:
        for temp in [1.0, 0.5, 0.1]:
            schedule.append((beta, temp))
            
    # Phase 2: Focused refinement
    for beta in [100.0, 200.0, 500.0]:
        for temp in [0.05, 0.01, 0.001]:
            schedule.append((beta, temp))

    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        # Ensure sum is not zero for the first evaluation
        if abs(np.sum(h_curr)) < 1e-10:
            h_curr[0] += 1.0
            
        val_curr = _objective_hard(h_curr)
        
        for beta, temp in schedule:
            if time.time() > deadline - 10.0:
                break
            
            # Run L-BFGS-B for a few iterations
            try:
                res = minimize(_objective_soft_max, h_curr, 
                              jac=lambda x: _soft_max_obj_and_grad(x, beta)[1],
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 30})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            # Metropolis Acceptance
            if val_new < val_curr:
                h_curr = h_new
                val_curr = val_new
            else:
                delta = val_new - val_curr
                if delta < 100 * temp: # Avoid overflow in exp
                    prob = np.exp(-delta / temp)
                    if rng.random() < prob:
                        h_curr = h_new
                        val_curr = val_new
            
            # Update global best
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
                
    # Final Polish: High beta, low temperature (deterministic)
    if time.time() < deadline - 5.0:
        for beta in [500.0, 1000.0]:
            if time.time() > deadline - 2.0:
                break
            try:
                res = minimize(_objective_soft_max, best_h,
                              jac=lambda x: _soft_max_obj_and_grad(x, beta)[1],
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
                val = _objective_hard(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
