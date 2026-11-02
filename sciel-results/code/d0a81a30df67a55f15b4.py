import numpy as np
import time

N = 50
DELTA = 1.0 / N

def _build_overlap_matrices():
    num_lags = 2 * N - 1
    M = np.zeros((num_lags, N, N))
    for k in range(num_lags):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()
_NUM_LAGS = 2 * N - 1

def _autoconv_all(h):
    """Compute all autoconvolution values."""
    h = np.maximum(h, 0)
    return np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])

def _autoconv_sup(h):
    return np.max(_autoconv_all(h))

def _ratio(h):
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _dual_subgradient_optimize(h_init, start_time, deadline, max_iter=500):
    """
    Optimize the ratio using a dual subgradient approach.
    
    We view the problem as:
      max_h  ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    By fixing which lag achieves the supremum (dual variable), we can use
    subgradient methods. We alternate between:
      1. For a given dual distribution over lags, optimize h
      2. Update the dual distribution based on current h
    """
    h = np.maximum(h_init, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h
    
    # Normalize to ||h||_1 = 1 for optimization (scale-invariant)
    h = h / s
    
    # Dual variables: probability distribution over lags
    # Initialize uniformly
    dual_weights = np.ones(_NUM_LAGS) / _NUM_LAGS
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        # Step 1: Given dual weights, optimize h
        # The objective becomes: max_h ||h||_2^2 / sum_k(w_k * (h*M_k*h))
        # subject to ||h||_1 = 1, h >= 0
        #
        # Use multiplicative update / Sinkhorn-like approach
        h_new = h.copy()
        for inner_iter in range(50):
            if time.time() - start_time > deadline:
                break
            
            # Compute weighted autoconvolution
            ff_vals = _autoconv_all(h_new)
            weighted_ff = np.sum(dual_weights * ff_vals)
            
            if weighted_ff < 1e-15:
                break
            
            # Gradient of log objective w.r.t. h
            # d/dh [log(||h||_2^2) - log(weighted_ff) - log(||h||_1)]
            # = 2h/||h||_2^2 - (sum_k w_k * 2*M_k*h)/weighted_ff - 1/||h||_1
            
            sum_h2 = np.sum(h_new**2)
            sum_h = np.sum(h_new)
            if sum_h2 < 1e-15 or sum_h < 1e-15:
                break
            
            # Compute gradient
            grad_num = 2.0 * h_new / sum_h2
            grad_den = np.zeros(N)
            for k in range(_NUM_LAGS):
                if dual_weights[k] > 1e-10:
                    grad_den += dual_weights[k] * (2.0 * _M_matrices[k] @ h_new)
            grad_den = grad_den / weighted_ff
            grad_const = 1.0 / sum_h
            
            grad = grad_num - grad_den - grad_const
            
            # Multiplicative update: h_new = h * exp(eta * grad)
            eta = 0.1 / (1.0 + iteration * 0.01)
            h_new = h_new * np.exp(eta * np.clip(grad, -10, 10))
            h_new = np.maximum(h_new, 1e-15)
            
            # Renormalize to ||h||_1 = 1
            h_new = h_new / np.sum(h_new)
        
        # Step 2: Update dual weights based on which lags achieve max
        ff_vals = _autoconv_all(h_new)
        max_ff = np.max(ff_vals)
        if max_ff < 1e-15:
            break
        
        # Softmax update with temperature
        temp = 0.05
        shifted = ff_vals - max_ff
        exp_vals = np.exp(shifted / temp)
        dual_weights = exp_vals / np.sum(exp_vals)
        
        # Evaluate
        h_candidate = h_new
        sc = _ratio(h_candidate)
        if sc > best_score:
            best_score = sc
            best_h = h_candidate.copy()
        
        # Check convergence
        if iteration > 10:
            prev_score = _ratio(h)
            if abs(sc - prev_score) < 1e-8:
                break
            h = h_new
    
    return best_h

def _diverse_initializations(n_init=20):
    """Generate diverse initializations for the optimization."""
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Sparse random
    for _ in range(5):
        n_spikes = np.random.randint(5, 15)
        indices = np.random.choice(N, n_spikes, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        inits.append(h)
    
    # 3. Structured patterns
    # Alternating
    h = np.zeros(N)
    h[::2] = 1.0
    inits.append(h)
    
    # Every third
    h = np.zeros(N)
    h[::3] = 1.0
    inits.append(h)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    h = np.exp(-x**2 / (2 * 0.2**2))
    inits.append(h)
    
    # Bimodal
    x = np.linspace(-1, 1, N)
    h = np.exp(-(x - 0.5)**2 / (2 * 0.15**2)) + np.exp(-(x + 0.5)**2 / (2 * 0.15**2))
    inits.append(h)
    
    # Random smooth
    for _ in range(5):
        freqs = np.random.choice(N, 8, replace=False)
        coeffs = np.random.randn(8)
        h = np.zeros(N)
        for f, c in zip(freqs, coeffs):
            h += c * np.cos(2 * np.pi * f * np.arange(N) / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-15:
            h = h / np.sum(h)
            inits.append(h)
    
    # Normalize all
    for i, h in enumerate(inits):
        s = np.sum(h)
        if s > 1e-15:
            inits[i] = h / s
    
    return inits

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations
    inits = _diverse_initializations(n_init=20)
    
    # Sort initializations by initial score (best first)
    init_scores = [_ratio(h) for h in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    # Optimize from each initialization with time budget
    n_trials = len(inits)
    time_per_trial = (DEADLINE - start_time - 5.0) / max(n_trials, 1)
    
    for trial in range(n_trials):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        h_init = inits[trial]
        trial_deadline = min(DEADLINE - 5.0, start_time + time_per_trial * (trial + 1))
        
        h_optimized = _dual_subgradient_optimize(h_init, start_time, trial_deadline, max_iter=200)
        sc = _ratio(h_optimized)
        
        if sc > best_score:
            best_score = sc
            best_h = h_optimized.copy()
    
    # Final polish on best solution
    if time.time() - start_time < DEADLINE - 2.0:
        # Try a few more iterations from the best solution
        h_final = _dual_subgradient_optimize(best_h.copy(), start_time, DEADLINE - 1.0, max_iter=300)
        sc = _ratio(h_final)
        if sc > best_score:
            best_score = sc
            best_h = h_final.copy()
    
    return best_h.tolist()
