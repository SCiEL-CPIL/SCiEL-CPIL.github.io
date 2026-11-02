"""Program using model-and-optimize approach for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600

def _fft_convolve(a, b):
    """Fast linear convolution via FFT."""
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    return np.fft.irfft(fa * fb, size)[:n]

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h, h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_smooth(h, beta=100.0):
    """
    Smooth surrogate: use log-sum-exp to approximate max of convolution.
    This makes the objective differentiable everywhere.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h, h)
    # Soft max via log-sum-exp
    c = np.max(conv)
    log_sum_exp = c + np.log(np.sum(np.exp(conv - c)))
    soft_max = log_sum_exp / beta
    return soft_max / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We want to minimize max_k conv[k] / (sum h)^2 subject to h[i] >= 0.
    # This is scale invariant. We can fix sum(h) = 1.
    # The problem is equivalent to finding a probability mass function p on {0,...,N-1}
    # that minimizes the peak of its linear convolution (p * p).
    # Note: linear convolution of length N vectors has length 2N-1.
    
    # Let's use a parametric family of smooth, symmetric functions.
    # Due to symmetry of the objective (if we reverse h, conv is reversed, max is same),
    # we can restrict to symmetric h.
    # Also, the optimal solution is likely unimodal and symmetric.
    
    # Parameterization:
    # h[i] = (1 - (x_i / L)^p)^q for |x_i| < L, 0 otherwise.
    # L = 0.25 (half width of the interval).
    # p controls the "roundness" or spread.
    # q controls the peak sharpness.
    
    # Actually, let's just use a simpler, more flexible family:
    # h(x) = exp(- (x/L)^alpha ) for |x| < L, 0 otherwise.
    # Or a polynomial bump: (1 - (x/L)^2)^beta.
    
    # Let's try to optimize a set of parameters for a few basis functions.
    # Or better, use scipy.optimize.minimize on a low-dimensional parameter vector
    # defining the shape.
    
    # Let's define a basis:
    # h(x) = c * (1 - (2x/L)^2)^k  for |x| < L/2? No, support is [-L/2, L/2]?
    # The interval is [-1/4, 1/4], so length 1/2.
    # Let's map x in [-1/4, 1/4] to t in [-1, 1].
    # h(t) = (1 - |t|^a)^b for |t| < 1.
    
    # Parameters: a (exponent on |t|), b (exponent on (1-|t|^a)).
    # We can also allow for a mixture or a two-parameter family.
    
    # 3-parameter family: h(x) = exp(-(|x|/sigma)^p) * support(x)
    # support(x) = 1 for |x| < L, 0 otherwise, with L adjustable
    # Parameters: sigma (width), p (shape), L (support half-width)
    
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    def param_h3(sigma, p, L):
        """3-parameter family with adjustable support."""
        if sigma <= 0.001 or p <= 0.1 or L <= 0.01:
            return np.ones(N) / N
        L = min(L, 0.25)  # Clamp to domain
        h = np.exp(-np.abs(x / sigma) ** p)
        h[np.abs(x) >= L] = 0.0
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    def objective3(params, use_smooth=False):
        sigma, p, L = params
        # Bound parameters
        sigma = np.clip(sigma, 0.005, 0.5)
        p = np.clip(p, 0.5, 10.0)
        L = np.clip(L, 0.02, 0.25)
        
        h = param_h3(sigma, p, L)
        if use_smooth:
            return compute_score_smooth(h, beta=50.0)
        return compute_score(h)
    
    # Initial guesses
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Try a grid of initial values for the 3-parameter family
    sigmas = [0.03, 0.05, 0.08, 0.12, 0.15, 0.2]
    ps = [1, 2, 3, 4, 6]
    Ls = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    for s0 in sigmas:
        for p0 in ps:
            for L0 in Ls:
                if time.time() > deadline:
                    break
                h0 = param_h3(s0, p0, L0)
                score0 = compute_score(h0)
                if score0 < best_score:
                    best_score = score0
                    best_h = h0.copy()
    
    # Local optimization using scipy with smooth surrogate
    # Use multiple restarts from the grid points
    # The smooth surrogate makes the objective differentiable everywhere
    
    restarts = []
    for s0 in sigmas:
        for p0 in ps:
            for L0 in Ls:
                restarts.append((s0, p0, L0))
    
    # Limit number of optimization runs to avoid timeout
    # With FFT speedup, we can afford more evaluations per optimization
    max_opts = 20
    opt_count = 0
    
    for s0, p0, L0 in restarts:
        if time.time() > deadline or opt_count >= max_opts:
            break
        opt_count += 1
        
        bounds = [(0.005, 0.5), (0.5, 10.0), (0.02, 0.25)]
        
        try:
            # First pass: smooth surrogate for fast convergence
            res = optimize.minimize(
                lambda params: objective3(params, use_smooth=True),
                [s0, p0, L0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6}
            )
            # Second pass: exact objective for refinement
            res2 = optimize.minimize(
                lambda params: objective3(params, use_smooth=False),
                res.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            final_params = res2.x
            final_score = objective3(final_params, use_smooth=False)
            if final_score < best_score:
                best_score = final_score
                best_h = param_h3(*final_params)
        except Exception as e:
            pass
            
    # Second parametric family: polynomial bump with adjustable shape
    # h(x) = (1 - (|x|/L)^a)^b for |x| < L
    # 3 parameters: a (exponent on |x|/L), b (outer exponent), L (support)
    
    def param_h_poly(a, b, L):
        if a <= 0.1 or b <= 0.1 or L <= 0.01:
            return np.ones(N) / N
        L = min(L, 0.25)
        t = np.abs(x) / L
        t = np.clip(t, 0, 1)
        h = (1 - t ** a) ** b
        h[t >= 1] = 0
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    def objective_poly(params, use_smooth=False):
        a, b, L = params
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.1, 10.0)
        L = np.clip(L, 0.02, 0.25)
        
        h = param_h_poly(a, b, L)
        if use_smooth:
            return compute_score_smooth(h, beta=50.0)
        return compute_score(h)
    
    # Grid search for polynomial family
    a_vals = [0.5, 1, 2, 4, 8]
    b_vals = [0.5, 1, 2, 4, 8]
    Ls_poly = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    for a0 in a_vals:
        for b0 in b_vals:
            for L0 in Ls_poly:
                if time.time() > deadline:
                    break
                h0 = param_h_poly(a0, b0, L0)
                score0 = compute_score(h0)
                if score0 < best_score:
                    best_score = score0
                    best_h = h0.copy()
    
    # Optimize polynomial family
    restarts_poly = []
    for a0 in a_vals:
        for b0 in b_vals:
            for L0 in Ls_poly:
                restarts_poly.append((a0, b0, L0))
    
    max_opts_poly = 15
    opt_count_poly = 0
    
    for a0, b0, L0 in restarts_poly:
        if time.time() > deadline or opt_count_poly >= max_opts_poly:
            break
        opt_count_poly += 1
        
        bounds = [(0.1, 10.0), (0.1, 10.0), (0.02, 0.25)]
        
        try:
            # Smooth pass then exact pass
            res = optimize.minimize(
                lambda params: objective_poly(params, use_smooth=True),
                [a0, b0, L0],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6}
            )
            res2 = optimize.minimize(
                lambda params: objective_poly(params, use_smooth=False),
                res.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            final_params = res2.x
            final_score = objective_poly(final_params, use_smooth=False)
            if final_score < best_score:
                best_score = final_score
                best_h = param_h_poly(*final_params)
        except Exception as e:
            pass

    # Final check: ensure best_h is valid (non-negative) and normalized
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
