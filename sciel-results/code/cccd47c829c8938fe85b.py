"""Third autocorrelation inequality — continuous relaxation with projected gradient descent."""
import numpy as np
import time

N = 400
DEADLINE = 115.0  # seconds, safety margin below 150s evaluator limit


def _compute_c3(h):
    """
    Compute C3[k] = sum_i h[i] * h[i+k] * h[i+2k] for k=0,...,N-1.
    Uses FFT-based approach: C3 is the autoconvolution of f with f*f.
    For discrete signal h of length N:
      autoconv = conv(h, h)  (length 2N-1)
      C3_full = conv(autoconv, h)  (length 3N-2)
    C3[k] corresponds to the coefficient at lag 2k in C3_full.
    """
    h_float = h.astype(np.float64)
    size = 4 * N  # enough for linear convolution without wrap-around
    h_fft = np.fft.fft(h_float, n=size)
    # autoconv = conv(h, h)
    autoconv_fft = h_fft * h_fft
    # C3_full = conv(autoconv, h)
    c3_full_fft = autoconv_fft * h_fft
    c3_full = np.fft.ifft(c3_full_fft).real
    # C3[k] is at index 2*k in c3_full
    # Valid k: 0 to N-1, so indices 0, 2, 4, ..., 2*(N-1)
    c3 = c3_full[0:2*N:2]
    return c3


def _score(h):
    """The evaluator measures max |C3| (lower is better)."""
    c3 = _compute_c3(h)
    return np.max(np.abs(c3))


def _score_and_grad(x):
    """
    Compute S(x) = max_k |C3[k]| and its subgradient with respect to x.
    x is a continuous vector of length N.
    C3[k] = sum_i x[i] * x[i+k] * x[i+2k]
    dC3[k]/dx[j] = sum over valid (i,k) where j participates:
      If j = i: x[i+k]*x[i+2k]
      If j = i+k: x[i]*x[i+2k]
      If j = i+2k: x[i]*x[i+k]
    For efficiency, we compute the gradient via FFT-like structure.
    Actually, let's compute it directly but efficiently.
    
    For the subgradient of S(x) = max_k |C3[k]|:
    Let k* be the index achieving the max.
    Then subgrad = sign(C3[k*]) * dC3[k*]/dx
    
    But we need dC3[k*]/dx for all x[j].
    """
    c3 = _compute_c3(x)
    k_star = np.argmax(np.abs(c3))
    sign_k = np.sign(c3[k_star])
    if sign_k == 0:
        sign_k = 1.0
    
    k = k_star
    # dC3[k]/dx[j] for j = 0,...,N-1
    grad = np.zeros(N)
    # For each i such that i, i+k, i+2k are all in [0, N-1]:
    # i ranges from 0 to N-1-2k
    max_i = N - 1 - 2 * k
    if max_i < 0:
        # k is too large, C3[k] = 0, gradient is 0
        return np.abs(c3[k_star]), np.zeros(N)
    
    # Vectorized computation
    i_vals = np.arange(0, max_i + 1)
    a = x[i_vals]       # x[i]
    b = x[i_vals + k]   # x[i+k]
    c = x[i_vals + 2*k] # x[i+2k]
    
    # dC3[k]/dx[i] += b*c
    np.add.at(grad, i_vals, b * c)
    # dC3[k]/dx[i+k] += a*c
    np.add.at(grad, i_vals + k, a * c)
    # dC3[k]/dx[i+2k] += a*b
    np.add.at(grad, i_vals + 2*k, a * b)
    
    grad *= sign_k
    return np.abs(c3[k_star]), grad


def _project_to_sign(x):
    """Project continuous vector to ±1."""
    return np.sign(x)
    # np.sign(0) = 0, so handle that
    result = np.sign(x)
    result[result == 0] = 1.0
    return result


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Returns a sequence of signed step heights (±1).
    Score is the upper bound on C3 (max |C3|), lower is better.
    """
    start_time = time.time()
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Initialize with several different starting points
    # 1. Random initialization
    # 2. Sine-based initialization
    # 3. Block-based initialization
    
    best_h = None
    best_score = np.inf
    
    def try_candidate(h):
        nonlocal best_h, best_score
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Try some structured initializations
    # Random
    np.random.seed(42)
    h_rand = np.random.choice([-1.0, 1.0], size=N)
    try_candidate(h_rand)
    
    # Sine patterns with various frequencies
    x_grid = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            if not elapsed_ok():
                break
            h_sine = np.sign(np.sin(2*np.pi*freq*x_grid + phase))
            h_sine[h_sine == 0] = 1.0
            try_candidate(h_sine)
        if not elapsed_ok():
            break
    
    # Block patterns
    for period in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if not elapsed_ok():
            break
        for duty in range(1, period):
            if duty == period - duty:
                break  # skip symmetric
            pattern = np.ones(period)
            pattern[duty:] = -1
            h_block = np.tile(pattern, N // period + 1)[:N]
            try_candidate(h_block)
    
    if not elapsed_ok():
        return _project_to_sign(best_h) if best_h is not None else np.ones(N)
    
    # Continuous relaxation with projected gradient descent
    # Start from the best discrete solution found so far
    x = best_h.astype(np.float64)
    
    # Also try starting from a smooth continuous initialization
    # Use a low-frequency sine as continuous start
    x_smooth = np.sin(2*np.pi*2*x_grid)
    
    # Projected gradient descent on the continuous relaxation
    # We minimize S(x) = max_k |C3[k]|
    # Since S is convex in each variable (it's a max of linear functions in x for fixed k... 
    # actually C3[k] is cubic in x, so not convex. But we can still use subgradient method.)
    
    # Actually, C3[k] = sum_i x[i]x[i+k]x[i+2k] is a cubic polynomial, not quadratic.
    # So the "convex quadratic" claim in the directive is not quite right.
    # But we can still use subgradient descent on the max of absolute values.
    
    lr_schedule = lambda t: 1.0 / (1.0 + 0.01 * t)
    
    # Run projected subgradient descent from multiple starts
    starts = [best_h.astype(np.float64), x_smooth, np.zeros(N), np.random.randn(N)]
    
    for x0 in starts:
        if not elapsed_ok():
            break
        
        x = x0.copy()
        # Project to make it a valid starting point
        x = _project_to_sign(x).astype(np.float64)
        
        # Adaptive step size
        step = 0.1
        for iteration in range(5000):
            if not elapsed_ok():
                break
            
            val, grad = _score_and_grad(x)
            
            # Update
            x_new = x - step * grad
            # Project to feasible set (||x||_inf <= 1 or x in [-1,1])
            # Actually, for the sign problem, we want x in {-1, 1}
            # But for continuous relaxation, let's keep x in [-1, 1]
            x_new = np.clip(x_new, -1.0, 1.0)
            
            # Project to sign
            x_sign = _project_to_sign(x_new)
            s_sign = _score(x_sign)
            
            if s_sign < best_score:
                best_score = s_sign
                best_h = x_sign.copy()
            
            # Also check the continuous value
            if val < best_score:
                # The continuous value is lower, but we need a discrete answer
                # Still, use it to guide the search
                pass
            
            # Adaptive step size
            if s_sign > val + 0.1:
                step *= 0.95
            else:
                step = min(step * 1.02, 1.0)
            
            x = x_new
        
        # Final projection and evaluation
        x_final = _project_to_sign(x)
        try_candidate(x_final)
    
    # If time remains, do local greedy refinement on the best solution
    if elapsed_ok() and best_h is not None:
        h_work = best_h.copy().astype(np.float64)
        
        # Greedy coordinate descent: try flipping each element
        for sweep in range(50):
            if not elapsed_ok():
                break
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if not elapsed_ok():
                    break
                h_try = h_work.copy()
                h_try[idx] *= -1
                s_try = _score(h_try)
                if s_try < best_score - 1e-12:
                    best_score = s_try
                    best_h = h_try.copy()
                    h_work = h_try.copy()
                    improved = True
            if not improved:
                break
    
    # Ensure output is ±1
    if best_h is None:
        best_h = np.ones(N)
    best_h = _project_to_sign(best_h).astype(np.float64)
    
    return best_h


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
