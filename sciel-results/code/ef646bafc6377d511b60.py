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
    
    # Clip for numerical stability
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        return 1e10, np.zeros_like(h)
    
    log_sum_exp = m + np.log(sum_exp) / beta
    softmax_val = log_sum_exp
    
    # Objective: 2*N*softmax_val / s^2
    obj = 2.0 * N * softmax_val / (s * s)
    
    # Gradient computation
    # d(softmax_val)/d(conv_i) = exp(beta*(conv_i - m)) / sum_exp
    weights = exp_shifted / sum_exp
    
    # d(conv_i)/d(h_j) for full convolution
    # conv[k] = sum_j h[j] * h[k-j] (with appropriate bounds)
    # Using FFT derivative or direct computation
    
    # For efficiency, compute gradient via FFT
    # conv = ifft(fft(h)^2)
    # d(conv)/d(h) = 2 * ifft(fft(h) * fft(dh))
    # But we need d(obj)/d(h) = d(obj)/d(conv) @ d(conv)/d(h)
    
    # Let's compute grad_conv = weights (shape 2N-1)
    # Then grad_h[j] = sum_k grad_conv[k] * d(conv_k)/d(h_j)
    # d(conv_k)/d(h_j) = h[k-j] + h[j-k] (with bounds checking)
    # This is equivalent to 2 * convolve(h, grad_conv) but with care for the diagonal
    
    # Actually, for real signals:
    # conv[k] = sum_i h[i] h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (if both indices valid)
    # Note: when k-j == j-k, i.e., 2j == k, we have 2*h[j]
    
    # We can compute this as:
    # grad_h = convolve(h, grad_conv) + convolve(grad_conv, h)
    # But this double counts the diagonal. Let's be careful.
    
    # Alternative: use the fact that conv = ifft(fft(h)^2)
    # grad_h = 2 * ifft(fft(h) * fft(grad_conv_padded))
    # where grad_conv_padded is grad_conv padded to fft_size
    
    fft_size = 1
    size = 2 * N - 1
    while fft_size < size:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    G = np.fft.rfft(np.concatenate([weights, np.zeros(fft_size - len(weights))]), n=fft_size)
    
    grad_h = 2.0 * np.fft.irfft(H * G, n=fft_size)[:N]
    
    # Chain rule: d(obj)/d(h) = d(obj)/d(softmax_val) * d(softmax_val)/d(conv) @ d(conv)/d(h)
    # + d(obj)/d(s) * d(s)/d(h)
    
    d_obj_d_softmax = 2.0 * N / (s * s)
    d_obj_d_s = -4.0 * N * softmax_val / (s * s * s)
    d_s_d_h = np.ones_like(h)
    
    grad = d_obj_d_softmax * grad_h + d_obj_d_s * d_s_d_h
    
    return obj, grad

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _spectral_flattening_obj(x):
    """
    Objective to minimize spectral variance (L4 norm of DCT coefficients).
    This promotes flat power spectrum, reducing peak-to-average ratio.
    """
    h = x
    # Compute DCT
    dct_coeffs = np.fft.rfft(h)
    power = np.abs(dct_coeffs) ** 2
    # Normalize by mean power
    mean_power = np.mean(power)
    if mean_power < 1e-10:
        return 1e10
    # L4 norm of normalized power spectrum
    normalized_power = power / mean_power
    return np.sum(normalized_power ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # List of initializations
    initializations = []
    
    # 1. Random normal with small mean
    for _ in range(10):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 2. Barker-like / Low autocorrelation sequences
    for _ in range(10):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.1 * rng.standard_normal()
        initializations.append(h0)
        
    # 3. Sine waves with random phase
    for freq in [1, 2, 5, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.5
        initializations.append(h0)

    # 4. Polynomial perturbations of 1s
    for _ in range(5):
        h0 = np.ones(N)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)

    # Optimization parameters
    betas = [50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        # Try different betas
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            # Use L-BFGS-B with custom gradient
            try:
                res = minimize(_soft_max_obj_and_grad, h_curr, 
                              method='L-BFGS-B',
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 20})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            # Evaluate hard objective
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Phase 5: Alternating spectral flattening and peak optimization
    if time.time() < deadline - 30.0:
        h_curr = best_h.copy()
        for phase in range(5):
            if time.time() > deadline - 20.0:
                break
            
            # Step 1: Spectral flattening
            try:
                res = minimize(_spectral_flattening_obj, h_curr,
                              method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-8})
                h_curr = res.x
            except Exception:
                pass
            
            # Step 2: Peak minimization
            for beta in [100.0, 200.0]:
                if time.time() > deadline - 20.0:
                    break
                try:
                    res = minimize(_soft_max_obj_and_grad, h_curr,
                                  method='L-BFGS-B',
                                  args=(beta,),
                                  jac=True,
                                  options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7})
                    h_curr = res.x
                except Exception:
                    pass
            
            val_curr = _objective_hard(h_curr)
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()

    # Final polish with high beta on the best solution
    if time.time() < deadline - 10.0:
        for beta in [200.0, 500.0]:
            try:
                res = minimize(_soft_max_obj_and_grad, best_h,
                              method='L-BFGS-B',
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
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
