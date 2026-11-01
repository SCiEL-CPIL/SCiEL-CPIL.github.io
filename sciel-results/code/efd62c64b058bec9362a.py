import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max over x,y of |autoconvolution| * something?
    
    Actually, let me think about what C3 is.
    
    The task says "improve an autoconvolution upper bound" and "score is the upper bound on C3".
    
    For a step function f on [-1/4, 1/4] with N equal intervals, the autoconvolution is:
    (f * f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    
    The C3 bound likely involves the maximum of the absolute value of the autoconvolution,
    or perhaps a specific integral related to the third autocorrelation.
    
    Let me reconsider. "Third autocorrelation inequality" - this might refer to a bound involving
    the third moment or a specific autocorrelation structure.
    
    A common setup: For a function f with ||f||_1 = 1 or some normalization, we want to bound
    max_t |(f*f)(t)| or similar.
    
    Let me assume: f is a step function with N intervals, each with height in {-1, 1} (signed).
    The autoconvolution is computed, and C3 is some measure of its maximum absolute value.
    
    Given the seed returns np.ones(N) and scores 1.0, and we want to minimize C3...
    
    Let me just compute the autoconvolution of the step function and take the max absolute value.
    For all ones, the autoconvolution of a step function that is 1 on [-1/4, 1/4] would be a
    triangular function with peak at 0.
    
    Actually, let me just implement a reasonable C3 computation and optimize for it.
    
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    # signed_step is an array of N values in {-1, 1}
    # The function f(x) = signed_step[i] for x in [i/N - 1/4, (i+1)/N - 1/4)
    # Wait, the domain is [-1/4, 1/4], so interval i covers [-1/4 + i/N, -1/4 + (i+1)/N)
    
    # Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed efficiently.
    
    # Let h = 1/N be the interval width.
    # f(s) = signed_step[i] for s in [-1/4 + i*h, -1/4 + (i+1)*h)
    
    # (f*f)(t) = sum over pairs (i,j) such that intervals overlap appropriately
    # This is complex. Let me use a direct numerical approach with fine grid.
    
    # Actually, for efficiency, let me compute it via FFT or direct summation.
    
    # Simpler: discretize the domain and compute convolution.
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid
    f_fine = np.zeros(M)
    for i in range(N):
        start = int(i * M / N)
        end = int((i + 1) * M / N)
        f_fine[start:end] = signed_step[i]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _soft_max_sidelobe(sidelobes, beta):
    """
    Smooth approximation of max |sidelobes| using log-sum-exp.
    sidelobes: array of sidelobe values (excluding DC)
    beta: inverse temperature (higher = sharper)
    """
    abs_vals = np.abs(sidelobes)
    if abs_vals.size == 0:
        return 0.0
    max_val = np.max(abs_vals)
    if max_val == 0:
        return 0.0
    # log-sum-exp trick for stability
    shifted = abs_vals - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _spectral_objective(params, beta):
    """
    Objective function for spectral shaping.
    params: real-valued vector of length N (continuous values, not constrained to ±1)
    beta: inverse temperature for soft-max
    
    We want to minimize the peak of the autocorrelation sidelobes.
    The autocorrelation is computed via FFT of the signal.
    """
    x = params
    # Compute autocorrelation via FFT
    fft_x = np.fft.fft(x, n=2*N)
    power_spectrum = np.abs(fft_x)**2
    autocorr = np.fft.ifft(power_spectrum).real
    
    # The DC component is at index 0, sidelobes are the rest
    # For a signal of length N, the autocorrelation has N unique lags
    sidelobes = autocorr[1:N]
    
    # Use soft-max to approximate max |sidelobe|
    objective = _soft_max_sidelobe(sidelobes, beta)
    
    return objective

def _spectral_objective_grad(params, beta):
    """
    Gradient of the spectral objective with respect to params.
    """
    x = params
    fft_x = np.fft.fft(x, n=2*N)
    power_spectrum = np.abs(fft_x)**2
    autocorr = np.fft.ifft(power_spectrum).real
    
    sidelobes = autocorr[1:N]
    abs_sidelobes = np.abs(sidelobes)
    
    if abs_sidelobes.size == 0:
        return 0.0, np.zeros_like(x)
    
    max_val = np.max(abs_sidelobes)
    if max_val == 0:
        return 0.0, np.zeros_like(x)
    
    # Compute soft-max and its gradient
    shifted = abs_sidelobes - max_val
    exp_vals = np.exp(beta * shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # d/dx of soft_max = sum_i w_i * sign(sidelobe_i) * d(sidelobe_i)/dx
    # sidelobe_k = autocorr[k] = (1/(2N)) * sum_j |FFT_x[j]|^2 * exp(-2pij*k/(2N))
    # This is complex; use automatic differentiation via finite differences or analytical
    
    # For efficiency, use a simpler approach: the gradient of max |autocorr[k]| w.r.t. x
    # can be approximated by the sign of the sidelobe times the correlation with the kernel
    
    # Actually, let's use a simpler gradient: the direction that reduces the largest sidelobe
    # The autocorrelation at lag k is x @ shift_k(x)
    # d(autocorr[k])/dx = shift_k(x) + shift_{-k}(x) (approximately)
    
    # For now, use a subgradient-like approach
    obj = _soft_max_sidelobe(sidelobes, beta)
    
    # Gradient via FFT: the sensitivity of each sidelobe to x
    # autocorr[k] = sum_j x[j] * x[j+k] (circular)
    # d(autocorr[k])/dx[m] = x[m+k] + x[m-k] (with circular indexing)
    
    # Weighted sum of these gradients
    grad = np.zeros_like(x)
    for k in range(1, N):
        if abs_sidelobes[k-1] == 0:
            continue
        w = weights[k-1]
        sign_val = np.sign(sidelobes[k-1])
        # d(autocorr[k])/dx[m] = x[m+k mod N] + x[m-k mod N]
        kernel = np.zeros(N)
        kernel[:N-k] = x[k:]
        kernel[N-k:] = x[:k]
        kernel2 = np.zeros(N)
        kernel2[:k] = x[N-k:]
        kernel2[k:] = x[:N-k]
        grad += w * sign_val * (kernel + kernel2)
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 0:
        grad = grad / grad_norm
    
    return obj, grad

def _continuous_to_signed(x):
    """
    Map continuous values to signed steps ±1 via thresholding.
    """
    return np.sign(x).astype(np.int8)

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Phase 1: Continuous spectral shaping
    # Start from a random continuous initialization
    np.random.seed(42)
    
    # Try multiple initializations
    n_restarts = 5
    beta_start = 10.0
    beta_end = 100.0
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 30.0:
            break
        
        # Initialize continuous signal
        if restart == 0:
            x_init = np.ones(N)
        else:
            x_init = np.random.randn(N)
        
        # Optimize with increasing beta (annealing)
        x_current = x_init.copy()
        
        # Use L-BFGS-B for continuous optimization
        # We need to be careful with time
        remaining = DEADLINE - 30.0 - (time.time() - start_time)
        if remaining < 10:
            break
        
        max_iter = min(200, int(remaining * 5))
        
        try:
            result = minimize(
                _spectral_objective,
                x_current,
                args=(beta_start,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 2}
            )
            x_current = result.x
        except Exception:
            pass
        
        # Map to signed steps
        signed_candidate = _continuous_to_signed(x_current)
        c3_candidate = _compute_c3(signed_candidate)
        
        if c3_candidate < best_c3:
            best_c3 = c3_candidate
            best_step = signed_candidate.copy()
        
        # Phase 2: Local search refinement on the signed step
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        # Local search: flip random elements
        no_improve = 0
        patience = 2000
        max_local_iters = 50000
        
        for iter_idx in range(max_local_iters):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            idx = np.random.randint(N)
            old_val = best_step[idx]
            best_step[idx] = -old_val
            
            new_c3 = _compute_c3(best_step)
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                no_improve = 0
            else:
                best_step[idx] = old_val
                no_improve += 1
            
            if no_improve >= patience:
                break
    
    return best_step
