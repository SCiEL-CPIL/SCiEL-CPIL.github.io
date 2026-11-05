import numpy as np
from scipy.optimize import minimize

N = 400

def compute_value(h):
    """Compute the objective value for a given height vector."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s ** 2))

def spectral_init(freq_width, center=0):
    """
    Initialize h by designing a spectral mask.
    We want |H(w)|^2 to be a low-pass filter to make c_k flat?
    Actually, if |H|^2 is low-pass, h is a sinc-like pulse.
    Autocorrelation of sinc is triangular.
    Peak at 0.
    We want to MINIMIZE the peak.
    So we want |H|^2 to be HIGH-PASS?
    If |H|^2 is high-pass, h is a difference of sines?
    Let's try a band-pass filter.
    """
    w = np.linspace(0, 2 * np.pi, N)
    # Target power spectrum
    # We want c_k to be small.
    # c_k is IFFT of |H|^2.
    # If |H|^2 is constant, c_0 is large, others 0.
    # If |H|^2 is "comb", c is periodic.
    # Let's try to make |H|^2 proportional to 1/sqrt(1 + (w/width)^2) ?
    
    # A known good shape for low PAR autocorrelation is a "box" in time? No.
    # Let's use a Gaussian in frequency.
    h_freq = np.exp(-0.5 * ((w - center) / freq_width) ** 2)
    # Ensure real h
    h_freq = np.abs(h_freq)
    h = np.fft.irfft(h_freq, n=N)
    h = h / np.sum(h)
    return h

def compute_gradient_smooth(h, beta):
    """Compute gradient of smooth max (logsumexp) of autoconvolution."""
    c = np.convolve(h, h)
    max_c = np.max(c)
    shifted = c - max_c
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    L = 2 * N
    W = np.fft.fft(weights, n=L)
    H = np.fft.fft(h, n=L)
    grad = 2.0 * np.fft.ifft(W * np.conj(H), n=L).real[:N]
    
    return grad, c

def line_search(h, grad, eta, max_iter=10):
    """Backtracking line search to ensure descent."""
    h_current = h.copy()
    val_current = compute_value(h_current)
    
    for k in range(max_iter):
        h_try = h_current - eta * grad
        s_try = np.sum(h_try)
        if abs(s_try) < 1e-12:
            eta *= 0.5
            continue
        h_try = h_try / s_try
        
        if np.any(np.abs(h_try) > 50.0):
            eta *= 0.5
            continue
        
        val_try = compute_value(h_try)
        if val_try < val_current - 1e-10:
            return h_try, eta
        eta *= 0.5
    
    return h_current, eta

def gradient_descent(h_init, max_iter=500, verbose=False):
    """Projected gradient descent with line search and adaptive beta."""
    h = h_init.copy()
    s = np.sum(h)
    h = h / s
    
    best_h = h.copy()
    best_val = compute_value(h)
    
    eta = 0.1
    momentum = np.zeros(N)
    mu = 0.8
    
    for i in range(max_iter):
        beta = 3.0 + 12.0 * (i / max_iter) ** 0.5
        grad, c = compute_gradient_smooth(h, beta)
        
        grad -= np.mean(grad)
        
        momentum = mu * momentum - eta * grad
        h_new = h + momentum
        
        h_new, eta = line_search(h, momentum, eta)
        
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            break
        h_new = h_new / s_new
        
        if np.any(np.abs(h_new) > 50.0):
            eta *= 0.5
            continue
        
        h = h_new
        val = compute_value(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        if i > 100 and best_val < 1.01:
            break
        
        if i % 100 == 0 and i > 0:
            if best_val > 1.05:
                eta = min(0.2, eta * 1.2)
            else:
                eta = max(0.01, eta * 0.8)
        
        if verbose and i % 100 == 0:
            print(f"Iter {i}: val={best_val:.6f}, eta={eta:.4f}, beta={beta:.2f}")
    
    return best_h, best_val

def construct():
    """Construct optimal step heights using spectral initialization."""
    best_h = None
    best_val = np.inf
    
    # Try different spectral widths
    widths = [0.1, 0.2, 0.5, 1.0, 2.0]
    centers = [0, 0.5, 1.0] # in units of pi?
    
    inits = []
    for w in widths:
        for c in centers:
            inits.append(spectral_init(w, c))
            
    # Also add a few random inits as fallback
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_rand = 1.0 + 0.5 * rng.standard_normal(N)
        h_rand = h_rand / np.sum(h_rand)
        inits.append(h_rand)

    for h_init in inits:
        h, val = gradient_descent(h_init, max_iter=300, verbose=False)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    # Final polish
    h, val = gradient_descent(best_h, max_iter=200, verbose=False)
    if val < best_val:
        best_val = val
        best_h = h.copy()
        
    return best_h
