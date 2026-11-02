"""
Strategy: Spectral Coefficient Optimization via IDCT
Represents the step function h as the squared magnitude of the inverse discrete cosine transform (IDCT) of a non-negative vector c.
h_i = (IDCT(c))_i^2.
This parameterization enforces non-negativity of h (since it's a square) and allows us to optimize the spectral coefficients c.
The autoconvolution of h corresponds to the square of the power spectrum of the IDCT coefficients (via Parseval/Convolution Theorem properties, though we compute directly in time domain for simplicity and robustness with N=600).
We optimize the coefficients c to minimize the normalized peak autoconvolution.
This is a structural change from the "Two-Block" or "Cosine Modulation" approaches, as it operates in the frequency domain with a squared-magnitude constraint, potentially finding smoother, more "flat" autoconvolutions by shaping the spectrum.
"""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _construct_from_idct(c):
    """
    Construct h from IDCT coefficients c.
    c: array of length N (or subset).
    h_i = (IDCT(c))_i^2
    """
    # scipy.fft.idct expects a vector of coefficients.
    # We use the standard DCT-II / IDCT-II pair.
    # Note: scipy.fft.idct by default uses the "orthonormal" type II if not specified?
    # Actually, scipy.fft.idct(x, type=2, norm=None)
    # Let's use the standard definition:
    # f_n = sqrt(2/N) * sum_{k=1}^{N} sqrt(2/N) * c_k * cos(pi * (n-0.5) * k / N) ?
    # It's safer to use scipy.fft.idct with appropriate normalization.
    # Let's stick to a simple, robust implementation using numpy to avoid scipy version issues if possible, 
    # or just use scipy.fft.idct.
    
    # Using scipy.fft.idct
    # The IDCT of c gives a time-domain signal x.
    # h = x^2
    # We need to ensure x is real. c is real, so x is real.
    
    # Note: The length of c can be less than N, padded with zeros.
    # For simplicity, let's optimize all N coefficients.
    
    # scipy.fft.idct(c, type=2, norm='ortho')
    # Let's verify the scaling.
    # If c is all ones, what is the result?
    # It's safer to just use the function and normalize h later?
    # No, the score is normalized by (sum h)^2, so scaling h doesn't change the score.
    # So we can just compute x = idct(c) and h = x^2.
    
    try:
        x = scipy.fft.idct(c, type=2, norm='ortho')
    except Exception:
        # Fallback to manual if scipy.fft fails (unlikely)
        # But scipy is available.
        x = np.zeros(N)
        
    h = x ** 2
    return h

def _score_idct(c):
    """
    Score the IDCT coefficients c.
    """
    h = _construct_from_idct(c)
    if np.sum(h) == 0:
        return 1e10
    return compute_score(h)

import scipy.fft

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def check_time():
        return time.time() > deadline

    # Initial guess: Uniform function corresponds to only the DC component (k=0) being non-zero?
    # If c = [1, 0, 0, ...], then x = constant.
    # Let's check: idct([1, 0, ...]) -> constant?
    # With norm='ortho', the DC term is scaled.
    # Let's just start with a random non-negative vector c.
    
    np.random.seed(42)
    
    # We optimize c. c can be any real vector?
    # The prompt says "IDCT of a non-negative vector".
    # So c >= 0.
    # Let's initialize c with small positive values.
    
    c_init = np.ones(N) * 0.1
    c_init[0] = 1.0 # DC term
    
    # Objective function for optimization
    # We need to enforce c >= 0.
    # We can use a penalty or reparameterize c = exp(y) or c = y^2.
    # Let's use c = np.maximum(0, y) or c = np.exp(y).
    # c = np.exp(y) ensures c > 0.
    # However, the scale of c doesn't matter for the score.
    # So we can fix the norm of c to 1, or just let the optimizer handle it.
    # To avoid issues with scale, let's normalize c in the objective.
    
    def objective(y):
        if check_time():
            return 1e10
        # c = np.exp(y)
        # This can lead to overflow.
        # Let's use c = np.maximum(0, y)
        c = np.maximum(0, y)
        # Normalize c to have unit L2 norm to fix scale
        norm_c = np.linalg.norm(c)
        if norm_c == 0:
            return 1e10
        c = c / norm_c
        return _score_idct(c)
    
    # Initial y
    y_init = np.zeros(N)
    y_init[0] = 1.0
    
    # Use L-BFGS-B or Powell?
    # L-BFGS-B is good for large N with bounds.
    # Bounds: y >= 0.
    
    try:
        res = optimize.minimize(
            objective,
            y_init,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'maxls': 10} # Keep it fast
        )
        
        if not check_time():
            y_opt = res.x
            c_opt = np.maximum(0, y_opt)
            norm_c = np.linalg.norm(c_opt)
            if norm_c > 0:
                c_opt = c_opt / norm_c
                h_opt = _construct_from_idct(c_opt)
                score_opt = compute_score(h_opt)
                
                # Compare with uniform
                h_uniform = np.ones(N)
                score_uniform = compute_score(h_uniform)
                
                if score_opt < score_uniform:
                    return h_opt.tolist()
                else:
                    return h_uniform.tolist()
                    
    except Exception:
        # Fallback to uniform if optimization fails
        return np.ones(N).tolist()

# EVOLVE-BLOCK-END
