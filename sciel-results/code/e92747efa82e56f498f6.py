"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h using FFT for speed."""
    n = len(h)
    # overlap[k] = sum_i h[i] * (1 - h[(i+k) % n])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # The second term is the circular autocorrelation of h
    # Use FFT to compute all shifts at once
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i+k) % n]
    overlap = np.sum(h) - autocorr
    # Exclude shift 0
    return np.max(overlap[1:])

def _chirp_heights(beta, n=N):
    """
    Generate step heights from a complex chirp.
    h[i] = |exp(j * (i*beta + i^2 * beta^2 / (2*n)))|^2 scaled
    Actually, for a linear chirp: z[i] = exp(j * pi * beta * i^2 / n)
    Then h[i] = |z[i]|^2 = 1, which is constant. That won't work.
    
    Instead, use a windowed chirp or a sum of chirps.
    Let's try: h[i] = 0.5 + 0.5 * cos(2*pi*beta*i/n + phase)
    But this needs to stay in [0,1] and sum to N/2.
    
    Better: use a Fejer kernel or similar.
    
    Actually, let's try: h[i] = (1 + cos(2*pi*beta*i/n)) / 2
    This is in [0,1], and sum = N/2 (since the cos sums to ~0 for non-integer beta).
    For integer beta, the sum of cos is 0, so sum = N/2. Good.
    """
    i = np.arange(n)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * beta * i / n)
    # Clamp to [0,1] (should already be there)
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum is exactly n/2
    current_sum = np.sum(h)
    target_sum = n / 2.0
    if current_sum > 0:
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
    return h

def _score(h):
    """Compute the score (max overlap)."""
    return _compute_max_overlap(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = N
    best_h = np.full(n, 0.5)
    best_score = _score(best_h)
    
    # Try different chirp frequencies
    # beta is the frequency parameter
    # For beta=0, we get uniform 0.5
    # Try a range of beta values
    
    # First, try a grid of beta values
    beta_grid = np.linspace(0.1, n//2, 50)
    for beta in beta_grid:
        h = _chirp_heights(beta, n)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        if time.time() > deadline - 10:
            break
    
    # Now refine with optimization around the best beta
    best_beta = 0.5  # initial guess
    
    def obj(beta_arr):
        beta = float(beta_arr[0])
        h = _chirp_heights(beta, n)
        return _score(h)
    
    # Try local optimization from best grid point
    # Find which beta gave the best score
    for beta in beta_grid:
        h = _chirp_heights(beta, n)
        s = _score(h)
        if s <= best_score + 1e-9:
            best_beta = beta
            break
    
    if time.time() < deadline - 30:
        try:
            res = minimize(obj, np.array([best_beta]), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_score:
                h_opt = _chirp_heights(float(res.x[0]), n)
                s_opt = _score(h_opt)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Try multi-frequency chirps: h[i] = 0.5 + 0.25*cos(2*pi*b1*i/n) + 0.25*cos(2*pi*b2*i/n)
    if time.time() < deadline - 20:
        def _multi_chirp_heights(b1, b2, n=N):
            i = np.arange(n)
            h = 0.5 + 0.25 * np.cos(2.0 * np.pi * b1 * i / n) + 0.25 * np.cos(2.0 * np.pi * b2 * i / n)
            h = np.clip(h, 0.0, 1.0)
            current_sum = np.sum(h)
            target_sum = n / 2.0
            if current_sum > 0:
                h = h * (target_sum / current_sum)
                h = np.clip(h, 0.0, 1.0)
            return h
        
        # Grid search over two frequencies
        b1_grid = np.linspace(0.5, n//3, 10)
        b2_grid = np.linspace(n//3, n//2, 10)
        for b1 in b1_grid:
            for b2 in b2_grid:
                if time.time() > deadline - 15:
                    break
                h = _multi_chirp_heights(b1, b2, n)
                s = _score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
