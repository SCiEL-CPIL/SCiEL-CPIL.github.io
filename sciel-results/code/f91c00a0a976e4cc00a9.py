import numpy as np
import time
from scipy.optimize import minimize

N = 600
INTERVAL_WIDTH = 1.0 / N  # width of each step on [-1/4, 1/4]
TOTAL_WIDTH = 0.5
HALF_WIDTH = 0.25

def compute_autoconvolution_max(heights, interval_width=INTERVAL_WIDTH):
    """Compute the normalized maximum of the autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-12):
        return float('inf')
    heights = np.maximum(heights, 0.0)
    
    # Autoconvolution via FFT
    n_fft = 2 * N
    f_ext = np.zeros(n_fft)
    f_ext[:N] = heights
    conv = np.real(np.fft.ifft(np.fft.fft(f_ext) ** 2))
    
    # The autoconvolution on the full domain [-1/2, 1/2]
    # We need the max over the valid range
    # conv[0] is at t=0, conv[k] is at t = k*interval_width
    # Valid range: t in [-1/2, 1/2], so indices 0..N-1 and n_fft-N+1..n_fft-1
    # Actually for circular convolution of length N, we get values at t = 0, iw, 2iw, ..., (N-1)iw
    # and the negative ones wrap around
    
    # The autoconvolution (f*f)(t) for t in [-W, W] where W = TOTAL_WIDTH
    # With N intervals of width iw, t ranges from -N/2*iw to (N-1)/2*iw approximately
    # conv[0] = (f*f)(0), conv[1] = (f*f)(iw), ..., conv[N-1] = (f*f)((N-1)*iw)
    # conv[n_fft-N+1] = (f*f)(-iw), etc.
    
    # Max over all valid t
    max_val = np.max(conv[:N])  # t >= 0
    min_val = np.min(conv[:N])
    
    # Normalization: divide by (sum of heights * interval_width)^2 or ||f||_2^2
    # Let's use ||f||_2^2 = sum(h^2) * iw
    l2_sq = np.sum(heights ** 2) * interval_width
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_val / l2_sq

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    best_heights = np.ones(N, dtype=np.float64)
    best_score = compute_autoconvolution_max(best_heights)
    
    # Try several parametric families
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Cosine bump: h(x) = 1 + a*cos(2*pi*x/W) where W = TOTAL_WIDTH
    x = np.linspace(-HALF_WIDTH, HALF_WIDTH, N, endpoint=False) + INTERVAL_WIDTH / 2
    for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h = 1.0 + a * np.cos(2 * np.pi * x / TOTAL_WIDTH)
        h = np.maximum(h, 0.0)
        candidates.append(h)
    
    # 3. Parabolic: h(x) = 1 + a*(x^2 - mean(x^2))
    x2 = x ** 2
    x2_centered = x2 - np.mean(x2)
    for a in [0.5, 1.0, 2.0, 5.0]:
        h = 1.0 + a * x2_centered
        h = np.maximum(h, 0.0)
        candidates.append(h)
    
    # 4. Two-bump: high on sides, low in middle
    for gap in [0.05, 0.1, 0.15, 0.2]:
        h = np.ones(N)
        mask = np.abs(x) < gap
        h[mask] = 0.3
        candidates.append(h)
    
    # 5. Smooth two-bump
    for sigma in [0.05, 0.08, 0.1, 0.12]:
        h = 1.0 - 0.5 * np.exp(-x**2 / (2 * sigma**2))
        h = np.maximum(h, 0.1)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() - start_time > deadline:
            break
        score = compute_autoconvolution_max(h)
        if score < best_score:
            best_score = score
            best_heights = h.copy()
    
    # Try local optimization from the best candidate
    if time.time() - start_time < deadline - 20:
        # Use scipy to optimize a subset of parameters
        # Represent as: h_i = max(0, base + amplitude * shape(x_i))
        # Try optimizing a few Fourier coefficients
        
        def objective(params):
            if time.time() - start_time > deadline:
                return 1e10
            a0, a1, b1, a2, b2, a3, b3 = params
            h = (a0 + a1 * np.cos(2*np.pi*x/TOTAL_WIDTH) + b1 * np.sin(2*np.pi*x/TOTAL_WIDTH) +
                 a2 * np.cos(4*np.pi*x/TOTAL_WIDTH) + b2 * np.sin(4*np.pi*x/TOTAL_WIDTH) +
                 a3 * np.cos(6*np.pi*x/TOTAL_WIDTH) + b3 * np.sin(6*np.pi*x/TOTAL_WIDTH))
            h = np.maximum(h, 0.0)
            return compute_autoconvolution_max(h)
        
        # Start from best candidate's Fourier coefficients
        best_f = np.fft.fft(best_heights[:N])
        x0 = np.array([
            np.real(best_f[0]) / N,
            np.real(best_f[1]) / N,
            np.imag(best_f[1]) / N,
            np.real(best_f[2]) / N,
            np.imag(best_f[2]) / N,
            np.real(best_f[3]) / N,
            np.imag(best_f[3]) / N,
        ])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if time.time() - start_time < deadline:
                h_opt = (res.x[0] + res.x[1] * np.cos(2*np.pi*x/TOTAL_WIDTH) + res.x[2] * np.sin(2*np.pi*x/TOTAL_WIDTH) +
                         res.x[3] * np.cos(4*np.pi*x/TOTAL_WIDTH) + res.x[4] * np.sin(4*np.pi*x/TOTAL_WIDTH) +
                         res.x[5] * np.cos(6*np.pi*x/TOTAL_WIDTH) + res.x[6] * np.sin(6*np.pi*x/TOTAL_WIDTH))
                h_opt = np.maximum(h_opt, 0.0)
                score_opt = compute_autoconvolution_max(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_heights = h_opt
        except Exception:
            pass
    
    # Final check
    if time.time() - start_time > deadline:
        return [float(v) for v in best_heights]
    
    return [float(v) for v in best_heights]
