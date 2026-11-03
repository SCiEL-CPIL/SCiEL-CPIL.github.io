import numpy as np
import time
import sys

N = 400
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _score(f):
    """
    Compute the score: max|autoconvolution| / (sum(f))^2.
    Lower is better.
    """
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac_and_grad(x, beta):
    """
    Compute soft-max approximation of max|ac| and its gradient w.r.t x.
    Returns (soft_val, grad_x).
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Handle numerical stability
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    
    # grad_ac[k] = weights[k] * signs[k]
    # d(ac[k])/dx[n] = x[n] + x[n-k] (for linear convolution)
    # This is a cross-correlation of grad_ac with x.
    # Specifically: grad_x[n] = sum_k grad_ac[k] * d(ac[k])/dx[n]
    # ac[k] = sum_i x[i] x[i+k]
    # d(ac[k])/dx[n] = x[n+k] + x[n-k] (with boundary conditions 0)
    
    # Efficiently compute via FFT:
    # grad_x = ifft( fft(grad_ac_padded) * conj(fft(x_padded)) )? No.
    # Let's use the property:
    # sum_k w_k sign(ac_k) (x[n+k] + x[n-k])
    # This is 2 * correlation(x, w*sign(ac)) if we treat w*sign(ac) as a filter?
    # Let g = w * sign(ac).
    # Term 1: sum_k g[k] x[n+k] -> Correlation of x and g?
    # Term 2: sum_k g[k] x[n-k] -> Correlation of x and g?
    
    # Let's compute:
    # y[n] = sum_k g[k] x[n+k]
    # This is cross-correlation.
    # Using FFT:
    # Y = fft(x) * conj(fft(g)) ? No, that's for correlation defined as sum x[n+k] g[k].
    # Standard correlation r_xy[n] = sum_k x[k] y[k-n].
    # Here we have sum_k g[k] x[n+k]. Let j = -k. sum_j g[-j] x[n-j].
    # So it's correlation of x and g_rev.
    
    # Easier:
    # grad_x[n] = sum_k g[k] (x[n+k] + x[n-k])
    # Let's compute c1[n] = sum_k g[k] x[n+k] and c2[n] = sum_k g[k] x[n-k].
    
    L_fft = 1
    while L_fft < 4 * N:
        L_fft *= 2
        
    # Pad g to length L_fft
    g_padded = np.zeros(L_fft)
    g_padded[:2*N-1] = weights * signs
    
    # Pad x to length L_fft
    x_padded = np.zeros(L_fft)
    x_padded[:N] = x
    
    G = np.fft.rfft(g_padded, n=L_fft)
    X = np.fft.rfft(x_padded, n=L_fft)
    
    # c1[n] = sum_k g[k] x[n+k]
    # This is equivalent to cross-correlation of x and g.
    # In FFT domain: ifft( conj(G) * X )? 
    # Let's check: 
    # ifft( conj(G) * X )[n] = sum_k g[k] x[n+k] (circular)
    # Yes, for zero-padded signals, circular = linear.
    
    c1 = np.fft.irfft(np.conj(G) * X, n=L_fft)[:N]
    
    # c2[n] = sum_k g[k] x[n-k]
    # This is sum_k g[k] x[n-k]. Let j = -k. sum_j g[-j] x[n-j].
    # This is cross-correlation of x and g_rev?
    # Or simply: ifft( G * conj(X) )?
    # ifft( G * conj(X) )[n] = sum_k g[k] x[n-k]?
    # Let's check. 
    # Correlation theorem: ifft( conj(G) * X ) is correlation.
    # Specifically, r_xy[n] = sum_k x[k] y[k-n].
    # Here we want sum_k g[k] x[n-k].
    # This is r_xg[n] = sum_k x[k] g[k-n]? No.
    # sum_k g[k] x[n-k] is the cross-correlation of x and g evaluated at n.
    # Standard definition: (x * g)[n] = sum_k x[k] g[n-k].
    # Cross-correlation: (x ⋆ g)[n] = sum_k x[k] g[k-n] = sum_k x[k] g^*[-k+n] (for real).
    # Wait, let's just use the symmetry.
    # sum_k g[k] x[n-k] is the same as sum_k x[k] g[n-k] if we swap indices?
    # Let j = n-k. k = n-j. sum_j g[n-j] x[j].
    # This is the convolution of g and x? No, convolution is sum g[k] x[n-k].
    # Yes! sum_k g[k] x[n-k] IS the convolution (g * x)[n].
    # So c2[n] = (g * x)[n].
    # FFT of convolution: G * X.
    # So c2 = ifft( G * X ).
    
    c2 = np.fft.irfft(G * X, n=L_fft)[:N]
    
    grad_x = 2.0 * (c1 + c2)
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """
    Compute objective (soft max |ac| / s^2) and gradient.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf, np.zeros(N)
    
    soft_val, grad_soft = _soft_max_abs_ac_and_grad(x, beta)
    
    # Objective: f(x) = soft_val / s^2
    # df/dx = (grad_soft * s^2 - soft_val * 2 * s * 1) / s^4
    #        = grad_soft / s^2 - 2 * soft_val / s^3
    
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    
    return val, grad

def _project_sum(x, target_sum):
    """
    Project x onto the hyperplane sum(x) = target_sum.
    Simplest: x + (target_sum - sum(x)) / N
    """
    current_sum = np.sum(x)
    delta = (target_sum - current_sum) / N
    return x + delta

def _optimize_phase_and_amplitude(x0, rng, max_time=5.0):
    """
    Optimize x using projected gradient descent on soft-max objective.
    Includes spectral initialization and local search.
    """
    start_time = time.time()
    x = x0.copy()
    s_target = np.sum(x)
    
    # Normalize x to have sum 1 for stability during optimization, then scale back?
    # The problem is scale invariant. Let's fix sum(x) = 1.
    if abs(s_target) < 1e-12:
        return x0, _score(x0)
    
    x = x / s_target
    s_target = 1.0
    
    best_x = x.copy()
    best_val = _score(best_x)
    
    beta = 50.0
    lr = 0.05
    
    # Adaptive Beta: start low, increase
    it = 0
    while time.time() - start_time < max_time:
        # Check deadline
        if time.time() > TIME_LIMIT:
            break
        
        val, grad = _objective_and_grad(x, beta)
        
        # Project gradient to sum(x)=1 subspace
        # grad_proj = grad - mean(grad)
        grad_mean = np.mean(grad)
        grad_proj = grad - grad_mean
        
        # Step
        x_new = x - lr * grad_proj
        
        # Project back to sum(x)=1
        x_new = _project_sum(x_new, s_target)
        
        # Evaluate true score
        true_val = _score(x_new)
        
        if true_val < best_val:
            best_val = true_val
            best_x = x_new.copy()
            # Decrease learning rate slightly on success? Or keep it?
            # Let's keep it constant or decrease slowly.
            lr = max(0.001, lr * 0.99)
        else:
            # Rejection
            lr = max(0.001, lr * 0.9)
            x = x_new # Accept the move even if worse? No, standard SGD accepts. 
            # But we are doing line search-ish. Let's just revert if worse?
            # Actually, soft max is smooth. Let's just accept and let it explore.
            # But if we accept bad moves, we might drift.
            # Let's try accepting always, but tracking best.
            x = x_new
        
        # Increase beta occasionally to sharpen max approximation
        if it % 100 == 0:
            beta = min(200.0, beta * 1.1)
            
        it += 1
        
    return best_x, best_val

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase/amplitude design.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    freqs = np.fft.rfftfreq(L_fft)
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        # Flat magnitude, random phase
        mag = np.ones(n_freq)
        phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        phase[0] = 0
    elif mode == 1:
        # Low-pass magnitude, random phase
        mag = np.exp(-5 * (freqs * L_fft) ** 2)
        phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        phase[0] = 0
    elif mode == 2:
        # Random magnitude, random phase
        mag = np.abs(rng.normal(0, 1, size=n_freq)) + 0.1
        phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        phase[0] = 0
    elif mode == 3:
        # "Ideal" magnitude for low autoconv: try to flatten |X|^2?
        # If |X|^2 is flat, ac is delta. But X must be real.
        # Let's try high-pass?
        mag = np.ones(n_freq)
        mag[1:] = 0.1
        phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        phase[0] = 0
    else:
        mag = np.ones(n_freq)
        phase = rng.uniform(0, 2 * np.pi, size=n_freq)
        phase[0] = 0
        
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
        
    return x

def construct():
    deadline = time.time() + TIME_LIMIT
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try several spectral initializations
    modes = [0, 1, 2, 3, 0, 1, 2] # More random restarts
    
    for i, mode in enumerate(modes):
        if time.time() > deadline:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Use a fraction of remaining time for this restart
        # Leave some time for final polish
        time_budget = min(remaining * 0.7, 10.0) # Cap per restart at 10s
        
        x_init = _spectral_init(rng, mode)
        
        # Optimize
        x_opt, score_opt = _optimize_phase_and_amplitude(x_init, rng, max_time=time_budget)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt.copy()
            
    # Final polish on the best solution found
    if time.time() < deadline - 2:
        x_final, score_final = _optimize_phase_and_amplitude(best_x, rng, max_time=deadline - time.time() - 1)
        if score_final < best_score:
            best_x = x_final
            best_score = score_final
            
    return best_x

# Self-test / Debug
if __name__ == "__main__":
    start = time.time()
    x = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Score: {_score(x):.6f}")
    print(f"Sum: {np.sum(x):.6f}")
    print(f"Max AC: {np.max(np.abs(_autoconv_fft(x))):.6f}")
