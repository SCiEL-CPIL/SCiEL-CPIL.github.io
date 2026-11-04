import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=50.0):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) using soft-max for peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)  # prevent overflow
    logsum = np.log(np.sum(np.exp(shifted)))
    soft_max = max_val + np.log(np.sum(np.exp(shifted))) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=50.0):
    return -_ratio(h, beta)

def _ratio_and_grad(h, beta=50.0):
    """Compute ratio and its gradient with respect to h."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros_like(h)
    
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0, np.zeros_like(h)
    
    # Soft-max
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    logsum = np.log(sum_exp)
    soft_max = max_val + logsum / beta
    
    if soft_max <= 0:
        return 0.0, np.zeros_like(h)
    
    ratio = l2_sq / soft_max
    
    # Gradient computation using finite differences or analytic
    # For simplicity, use finite difference for the gradient
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(len(h)):
        h_plus = h.copy()
        h_plus[i] += eps
        r_plus = _ratio(h_plus, beta)
        h_minus = h.copy()
        h_minus[i] = max(h[i] - eps, 0)
        r_minus = _ratio(h_minus, beta)
        grad[i] = (r_plus - r_minus) / (2 * eps)
    
    return ratio, grad

def _dft_parameterize(z):
    """Convert complex z to non-negative h via |IFFT(z)|^2."""
    h = np.abs(np.fft.ifft(z)) ** 2
    return h

def _dft_inverse(h):
    """Find z such that |IFFT(z)|^2 ≈ h. Use z = FFT(sqrt(h))."""
    h = np.maximum(h, 0.0)
    sqrt_h = np.sqrt(h)
    z = np.fft.fft(sqrt_h)
    return z

def _optimize_from_z(z, start_time, max_iter=200, beta=50.0):
    """Optimize z to maximize ratio."""
    best_score = 0.0
    best_z = z.copy()
    
    for iteration in range(max_iter):
        if _time_left(start_time) < 5:
            break
        
        h = _dft_parameterize(z)
        l1 = np.sum(h)
        if l1 == 0:
            break
        h_norm = h / l1
        score = _ratio(h_norm, beta)
        
        if score > best_score:
            best_score = score
            best_z = z.copy()
        
        # Compute gradient of score with respect to z
        # score = f(h) where h = |IFFT(z)|^2
        # Use finite differences on z components
        eps = 1e-6
        grad_z = np.zeros_like(z, dtype=complex)
        
        # Only optimize a subset of z components for speed
        # Use real and imaginary parts
        n_components = len(z)
        
        for i in range(n_components):
            if _time_left(start_time) < 5:
                break
            
            # Real part
            z_plus = z.copy()
            z_plus[i] += eps
            h_plus = _dft_parameterize(z_plus)
            l1_plus = np.sum(h_plus)
            if l1_plus > 0:
                score_plus = _ratio(h_plus / l1_plus, beta)
            else:
                score_plus = 0.0
            
            z_minus = z.copy()
            z_minus[i] -= eps
            h_minus = _dft_parameterize(z_minus)
            l1_minus = np.sum(h_minus)
            if l1_minus > 0:
                score_minus = _ratio(h_minus / l1_minus, beta)
            else:
                score_minus = 0.0
            
            grad_z[i] += (score_plus - score_minus) / (2 * eps)
            
            # Imaginary part
            z_plus = z.copy()
            z_plus[i] += 1j * eps
            h_plus = _dft_parameterize(z_plus)
            l1_plus = np.sum(h_plus)
            if l1_plus > 0:
                score_plus = _ratio(h_plus / l1_plus, beta)
            else:
                score_plus = 0.0
            
            z_minus = z.copy()
            z_minus[i] -= 1j * eps
            h_minus = _dft_parameterize(z_minus)
            l1_minus = np.sum(h_minus)
            if l1_minus > 0:
                score_minus = _ratio(h_minus / l1_minus, beta)
            else:
                score_minus = 0.0
            
            grad_z[i] -= 1j * (score_plus - score_minus) / (2 * eps)
        
        # Update z
        step_size = 0.1
        z = z + step_size * grad_z
        
        # Ensure z stays in reasonable range
        z = np.clip(np.abs(z), 0, 10) * np.exp(1j * np.angle(z))
    
    # Final evaluation
    h_best = _dft_parameterize(best_z)
    l1_best = np.sum(h_best)
    if l1_best > 0:
        h_best = h_best / l1_best
        final_score = _ratio(h_best, beta)
        if final_score > best_score:
            best_score = final_score
    
    return h_best, best_score

def _gaussian_profile(sigma, center=0.0):
    """Create a Gaussian profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return h

def _lorentzian_profile(gamma, center=0.0):
    """Create a Lorentzian profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h

def _power_profile(k, center=0.0):
    """Create a power law profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.zeros(N)
    mask = x > center
    h[mask] = (x[mask] - center) ** k
    return h

def _two_gaussian(sigma1, center1, sigma2, center2):
    """Create a two-Gaussian profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h1 = np.exp(-((x - center1) ** 2) / (2 * sigma1 ** 2))
    h2 = np.exp(-((x - center2) ** 2) / (2 * sigma2 ** 2))
    return h1 + h2

def _sinc_like_profile(width, center=0.0):
    """Create a sinc-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = (x - center) / width
    h = np.sinc(t) ** 2
    return np.maximum(h, 0)

def construct():
    start_time = time.time()
    
    best_heights = [1.0] * N
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = []
    
    # Gaussian profiles
    for sigma in [0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _gaussian_profile(sigma, center)
            if np.sum(h) > 0:
                initializations.append(h)
    
    # Lorentzian profiles
    for gamma in [0.03, 0.05, 0.08, 0.1]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _lorentzian_profile(gamma, center)
            if np.sum(h) > 0:
                initializations.append(h)
    
    # Power law profiles
    for k in [2, 3, 4, 5]:
        for center in [0.0, 0.05]:
            h = _power_profile(k, center)
            if np.sum(h) > 0:
                initializations.append(h)
    
    # Two-Gaussian profiles
    for sigma1, center1, sigma2, center2 in [
        (0.05, 0.0, 0.05, 0.1),
        (0.05, -0.05, 0.05, 0.05),
        (0.1, 0.0, 0.05, 0.15),
        (0.05, -0.1, 0.05, 0.1),
    ]:
        h = _two_gaussian(sigma1, center1, sigma2, center2)
        if np.sum(h) > 0:
            initializations.append(h)
    
    # Sinc-like profiles
    for width in [0.05, 0.1, 0.15]:
        for center in [0.0, 0.05]:
            h = _sinc_like_profile(width, center)
            if np.sum(h) > 0:
                initializations.append(h)
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Optimize from each initialization
    for i, h_init in enumerate(initializations):
        if _time_left(start_time) < 10:
            break
        
        # Normalize
        l1 = np.sum(h_init)
        if l1 == 0:
            continue
        h_init = h_init / l1
        
        # Convert to z for DFT parameterization
        z_init = _dft_inverse(h_init)
        
        # Optimize
        h_opt, score_opt = _optimize_from_z(z_init, start_time, max_iter=150)
        
        if score_opt > best_score:
            best_score = score_opt
            best_heights = h_opt.tolist()
    
    return best_heights
