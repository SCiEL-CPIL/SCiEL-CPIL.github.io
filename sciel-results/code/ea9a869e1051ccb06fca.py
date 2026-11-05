import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

N = 400

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _soft_max_and_weights(x, beta):
    """
    Differentiable approximation to max using log-sum-exp.
    Returns the soft-max value and the weights (probabilities) of each element.
    """
    m = np.max(x)
    x_shifted = x - m
    # Clip to avoid overflow/underflow in exp
    x_shifted = np.clip(x_shifted, -50, 50)
    exp_vals = np.exp(beta * x_shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return m, np.zeros_like(x)
    weights = exp_vals / sum_exp
    val = m + (1.0 / beta) * np.log(sum_exp)
    return val, weights

def _evaluate_exact(h):
    """Compute the exact objective value as defined in the task."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _objective_and_grad(params, beta):
    """
    Soft-max objective with analytical gradient.
    Objective: abs(2 * N * soft_max(conv(h,h)) / sum(h)^2)
    Since we want to minimize the value, and the value is positive in our context 
    (max conv is usually positive for these signals, and we take abs), 
    we can just minimize 2 * N * soft_max / s^2.
    However, to be safe with the abs, let's look at the sign.
    For h with mean 0 or small mean, max conv is typically positive.
    If max conv is negative, abs flips the gradient.
    Given the structure of autoconvolution of real signals, c[0] = sum(h^2) >= 0.
    So max(c) is always >= 0. Thus abs() is redundant for the max term.
    We minimize f(h) = 2 * N * sm / s^2.
    """
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    c = _autoconv_fft(h)
    sm, weights = _soft_max_and_weights(c, beta)
    
    # Gradient of soft_max w.r.t h
    # d(sm)/d(h[i]) = sum_{k} weights[k] * d(c[k])/d(h[i])
    # d(c[k])/d(h[i]) = h[k-i] if 0<=k-i<N (from h[i]*h[k-i]) 
    #                  + h[k-i] if 0<=k-i<N and k-i != i (from h[k-i]*h[i])
    # Actually, c[k] = sum_j h[j]h[k-j].
    # Term with h[i]: 
    # 1. j=i -> h[i]h[k-i]. Deriv: h[k-i] if 0<=k-i<N.
    # 2. k-j=i -> j=k-i -> h[k-i]h[i]. Deriv: h[k-i] if 0<=k-i<N.
    # If i != k-i (i.e., k != 2i), these are distinct terms in the sum? 
    # No, the sum is over j.
    # If i != k-i, then j=i and j=k-i are two different indices in the summation range (provided both are valid).
    # So d(c[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] if k != 2i.
    # If k == 2i, j=i is the only term where h[i] appears (h[i]*h[i]). Deriv: 2*h[i]? 
    # Wait. d/dh_i (h_i^2) = 2 h_i.
    # Let's re-verify.
    # c[k] = sum_{j} h[j] h[k-j]
    # d(c[k])/d(h[i]) = sum_j [ delta_{j,i} h[k-j] + h[j] delta_{k-j,i} ]
    # = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N)
    # The second term comes from j = k-i.
    # If i = k-i (k=2i), the two terms are identical in the sum? 
    # No, in the sum over j, if j=i, we get h[i]h[i]. 
    # The derivative of h[i]h[i] w.r.t h[i] is 2h[i].
    # The formula sum_j [ ... ] gives:
    # Term 1 (j=i): h[k-i]
    # Term 2 (j=k-i): h[k-i]
    # If i = k-i, then j=i is the same index. 
    # Does the sum count it twice? 
    # The sum is linear. d/dx (x^2) = 2x.
    # If we write x^2 as sum of x*x, the derivative is x+x=2x.
    # So yes, d(c[k])/d(h[i]) = 2 * h[k-i] if 0 <= k-i < N.
    # Wait, is it always 2?
    # Let's check k=0. c[0] = sum h[j]h[-j] = h[0]^2 (since h is defined 0..N-1, h[-j]=0 for j>0).
    # d(c[0])/d(h[0]) = 2 h[0].
    # Formula: k=0, i=0. k-i=0. 2*h[0]. Correct.
    # Let's check k=1. c[1] = h[0]h[1] + h[1]h[0] = 2 h[0]h[1].
    # d(c[1])/d(h[0]) = 2 h[1].
    # Formula: k=1, i=0. k-i=1. 2*h[1]. Correct.
    # d(c[1])/d(h[1]) = 2 h[0].
    # Formula: k=1, i=1. k-i=0. 2*h[0]. Correct.
    
    # So d(c[k])/d(h[i]) = 2 * h[k-i] if 0 <= k-i < N, else 0.
    
    n_conv = 2 * N - 1
    grad_h = np.zeros(N)
    
    # We need grad_h[i] = sum_{k} weights[k] * 2 * h[k-i] (where 0<=k-i<N)
    # Let j = k-i. Then k = i+j.
    # j ranges from 0 to N-1.
    # k ranges from i to i+N-1.
    # This is a convolution of weights with h?
    # grad_h = 2 * convolve(weights, h, mode='valid')? 
    # convolve(w, h)[i] = sum_j w[i-j] h[j]? No.
    # Let's just loop or use FFT.
    # grad_h[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
    # This is a cross-correlation or convolution slice.
    # Let W be weights. Let H be h.
    # We want G[i] = sum_j W[i+j] H[j].
    # This is equivalent to (W * H)[i] if we reverse one?
    # Let's stick to the loop for clarity, N=400 is small enough for 400x400 ops per gradient eval?
    # 400*400 = 160k ops. Very fast in numpy.
    
    # Vectorized approach:
    # Create a matrix M where M[i, j] = weights[i+j]
    # Then grad_h[i] = 2 * dot(M[i], h)
    
    # Efficiently construct M using broadcasting or sliding_window_view
    # indices for weights: i+j for i in 0..N-1, j in 0..N-1
    # i+j ranges from 0 to 2N-2.
    
    # Let's use a simple loop for the gradient part as it's robust and N is small.
    # Or better, use FFT for the gradient calculation too?
    # grad_h = 2 * ifft(fft(weights, 2N) * fft(h, 2N)) ? No, it's a specific slice.
    
    # Let's use the direct sum with numpy vectorization per i? No, that's N calls.
    # Let's use the matrix approach.
    # idx_i = np.arange(N)
    # idx_j = np.arange(N)
    # K = idx_i[:, None] + idx_j[None, :]  # Shape (N, N)
    # W_mat = weights[K]
    # grad_h = 2.0 * (W_mat @ h)
    
    idx_i = np.arange(N)
    idx_j = np.arange(N)
    K = idx_i[:, None] + idx_j[None, :]
    W_mat = weights[K]
    grad_h = 2.0 * (W_mat @ h)
    
    # Objective: val = 2 * N * sm / s^2
    # d(val)/d(h[i]) = 2 * N * [ (d(sm)/d(h[i]) * s^2 - sm * 2 * s * 1) / s^4 ]
    # = 2 * N * [ (grad_sm_i * s - 2 * sm) / s^3 ]
    # = (2 * N / s^3) * (s * grad_sm_i - 2 * sm)
    
    coeff = 2.0 * N / (s ** 3)
    grad = coeff * (s * grad_h - 2.0 * sm)
    
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10, np.zeros_like(h)
    
    return val, grad

def _objective(params, beta):
    """Soft-max objective for full h vector."""
    val, _ = _objective_and_grad(params, beta)
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
        return val

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1a: Random signals with small mean
    for _ in range(10):
        h = rng.standard_normal(N)
        h = h - np.mean(h) + rng.uniform(-0.01, 0.01)
        candidates.append(h)
    
    # 1b: Structured patterns from current program
    m = N // 2
    h1 = np.zeros(N)
    h1[:m] = 1.0
    h1[m:] = -1.0
    candidates.append(h1)
    
    q = N // 4
    h2 = np.zeros(N)
    h2[0:q] = 1.0
    h2[q:2*q] = -1.0
    h2[2*q:3*q] = 1.0
    h2[3*q:4*q] = -1.0
    candidates.append(h2)
    
    idx = np.arange(N)
    h3 = np.cos(2.0 * np.pi * idx / N)
    h3 -= h3.mean()
    candidates.append(h3)
    
    h4 = np.sin(2.0 * np.pi * idx / N)
    candidates.append(h4)
    
    # Evaluate all candidates
    for h in candidates:
        try_h(h)
        
    # Phase 2: Gradient-based optimization with beta annealing
    # We start with a low beta (smooth) and increase it (sharp)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        # Try a few restarts from current best
        for restart in range(3):
            if time.time() > deadline - 15.0:
                break
                
            if restart == 0:
                init_h = current_h.copy()
            else:
                # Perturb
                scale = 0.1 * np.std(current_h) if np.std(current_h) > 0 else 0.1
                init_h = current_h + rng.standard_normal(N) * scale
                # Normalize sum to avoid issues with s^2
                s = np.sum(init_h)
                if abs(s) < 1e-6:
                    init_h += 0.01
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective,
                    init_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We need to return grad? 
                    # Wait, _objective only returns val. 
                    # We should define a function that returns val, grad
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                # Check if we found a better point
                if res.success and res.fun < best_val:
                    # Verify with exact evaluation
                    val_exact = _evaluate_exact(res.x)
                    if val_exact < best_val:
                        best_val = val_exact
                        best_h = res.x.copy()
                        current_h = best_h.copy()
                        
            except Exception as e:
                pass

    # Phase 3: Final local refinement (Coordinate-wise nudge)
    # This helps polish the result to the exact max
    if time.time() < deadline - 5.0:
        h_best = best_h.copy()
        cur_val = _evaluate_exact(h_best)
        
        # Simple hill climbing
        for step in [0.05, 0.01, 0.001]:
            if time.time() > deadline - 2.0:
                break
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 1.0:
                        break
                    old = h_best[i]
                    for delta in (step * max(1.0, abs(old)), -step * max(1.0, abs(old))):
                        h_try = h_best.copy()
                        h_try[i] += delta
                        val_try = _evaluate_exact(h_try)
                        if val_try < cur_val - 1e-10:
                            h_best = h_try
                            cur_val = val_try
                            improved = True
                            break
                    if time.time() > deadline - 1.0:
                        break
                if time.time() > deadline - 1.0:
                    break

        if cur_val < best_val:
            best_val = cur_val
            best_h = h_best

    return best_h

# Helper to make _objective work with jac=True
# We need a function that returns (val, grad)
def _objective_with_grad(params, beta):
    return _objective_and_grad(params, beta)

# Patch the minimize call to use the grad function properly
# The previous block used _objective which only returns val.
# We will re-define the construct function slightly to use the correct function.

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
        return val

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1a: Random signals with small mean
    for _ in range(10):
        h = rng.standard_normal(N)
        h = h - np.mean(h) + rng.uniform(-0.01, 0.01)
        candidates.append(h)
    
    # 1b: Structured patterns from current program
    m = N // 2
    h1 = np.zeros(N)
    h1[:m] = 1.0
    h1[m:] = -1.0
    candidates.append(h1)
    
    q = N // 4
    h2 = np.zeros(N)
    h2[0:q] = 1.0
    h2[q:2*q] = -1.0
    h2[2*q:3*q] = 1.0
    h2[3*q:4*q] = -1.0
    candidates.append(h2)
    
    idx = np.arange(N)
    h3 = np.cos(2.0 * np.pi * idx / N)
    h3 -= h3.mean()
    candidates.append(h3)
    
    h4 = np.sin(2.0 * np.pi * idx / N)
    candidates.append(h4)
    
    # Evaluate all candidates
    for h in candidates:
        try_h(h)
        
    # Phase 2: Gradient-based optimization with beta annealing
    # We start with a low beta (smooth) and increase it (sharp)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        # Try a few restarts from current best
        for restart in range(3):
            if time.time() > deadline - 15.0:
                break
                
            if restart == 0:
                init_h = current_h.copy()
            else:
                # Perturb
                scale = 0.1 * np.std(current_h) if np.std(current_h) > 0 else 0.1
                init_h = current_h + rng.standard_normal(N) * scale
                # Normalize sum to avoid issues with s^2
                s = np.sum(init_h)
                if abs(s) < 1e-6:
                    init_h += 0.01
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    _objective_with_grad,
                    init_h,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                # Check if we found a better point
                if res.success and res.fun < best_val:
                    # Verify with exact evaluation
                    val_exact = _evaluate_exact(res.x)
                    if val_exact < best_val:
                        best_val = val_exact
                        best_h = res.x.copy()
                        current_h = best_h.copy()
                        
            except Exception as e:
                pass

    # Phase 3: Final local refinement (Coordinate-wise nudge)
    # This helps polish the result to the exact max
    if time.time() < deadline - 5.0:
        h_best = best_h.copy()
        cur_val = _evaluate_exact(h_best)
        
        # Simple hill climbing
        for step in [0.05, 0.01, 0.001]:
            if time.time() > deadline - 2.0:
                break
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 1.0:
                        break
                    old = h_best[i]
                    for delta in (step * max(1.0, abs(old)), -step * max(1.0, abs(old))):
                        h_try = h_best.copy()
                        h_try[i] += delta
                        val_try = _evaluate_exact(h_try)
                        if val_try < cur_val - 1e-10:
                            h_best = h_try
                            cur_val = val_try
                            improved = True
                            break
                    if time.time() > deadline - 1.0:
                        break
                if time.time() > deadline - 1.0:
                    break

        if cur_val < best_val:
            best_val = cur_val
            best_h = h_best

    return best_h
