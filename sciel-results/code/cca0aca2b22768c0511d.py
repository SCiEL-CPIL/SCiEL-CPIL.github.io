"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA_T = (1/4 - (-1/4)) / N  # interval width

def _convolution(h):
    """Compute the autoconvolution of a piecewise-constant function on [-1/4, 1/4].
    
    The function is 1 on [-1/4, 1/4] with N intervals of width DELTA_T.
    The autoconvolution has support [-1/2, 1/2].
    """
    # For a piecewise-constant function with values h[i] on interval [x_i, x_{i+1}],
    # the convolution at point t is sum over i,j of h[i]*h[j] * overlap_length(interval_i, interval_j shifted by t)
    # 
    # A more efficient approach: use the fact that the convolution of two piecewise-constant
    # functions on a grid can be computed via discrete convolution with appropriate weights.
    #
    # For step functions on a uniform grid, the convolution at grid points can be approximated
    # by: (conv * delta_t) where conv is the discrete auto-convolution of h.
    # Specifically, if f(x) ≈ h[i] for x in [i*d, (i+1)*d), then
    # (f*f)(t) ≈ d * sum_{i,j: |i-j| such that overlap} h[i]*h[j] * (1 - |t - (i-j)*d|/d) for appropriate range
    #
    # A cleaner approach: the continuous convolution of two piecewise-constant functions
    # with values h[i] on [i*d, (i+1)*d) is a piecewise-linear function.
    # The peak occurs at t=0 (for symmetric h) and the value at t=0 is sum_i h[i]^2 * d.
    # More generally, for non-symmetric h, the peak of the autoconvolution may not be at 0.
    #
    # Let me compute it properly. The autoconvolution (f*f)(t) = ∫ f(s) f(t-s) ds.
    # For piecewise constant f on [-1/4, 1/4] divided into N equal parts:
    # Let x_i = -1/4 + i * DELTA_T for i=0..N, so the domain is [x_0, x_N].
    # f(x) = h[i] for x in [x_i, x_{i+1}).
    #
    # (f*f)(t) = sum_{i,j} h[i]*h[j] * measure({s: s in [x_i,x_{i+1}), t-s in [x_j,x_{j+1})})
    #          = sum_{i,j} h[i]*h[j] * measure([x_i,x_{i+1}) ∩ [t-x_{j+1}, t-x_j))
    #
    # This is a piecewise linear function in t. The support is [-1/2, 1/2].
    # For efficiency, I'll compute it on a fine grid.
    
    # Support of f*f is [-1/2, 1/2]. Let's sample on a fine grid.
    # Number of samples: use N^2 * some_factor or just a fine enough grid.
    n_samples = 1000
    t_grid = np.linspace(-0.5, 0.5, n_samples)
    dt = t_grid[1] - t_grid[0]
    
    x_left = -0.25
    x_right = 0.25
    
    # For each t, compute the overlap integral
    # f(s) is nonzero for s in [x_left, x_right]
    # f(t-s) is nonzero for t-s in [x_left, x_right], i.e., s in [t-x_right, t-x_left]
    # So the integral is over s in [max(x_left, t-x_right), min(x_right, t-x_left)]
    
    conv = np.zeros(n_samples)
    for idx, t in enumerate(t_grid):
        s_low = max(x_left, t - x_right)
        s_high = min(x_right, t - x_left)
        if s_low >= s_high:
            conv[idx] = 0.0
            continue
        # Now integrate piecewise
        # s ranges from s_low to s_high
        # f(s) = h[i] where i = floor((s - x_left) / DELTA_T)
        # f(t-s) = h[j] where j = floor((t-s - x_left) / DELTA_T)
        # We need to find the breakpoints where either i or j changes.
        # i changes at s = x_left + i*DELTA_T
        # j changes at t-s = x_left + j*DELTA_T => s = t - x_left - j*DELTA_T
        
        # Collect all breakpoints in [s_low, s_high]
        breakpoints = set()
        breakpoints.add(s_low)
        breakpoints.add(s_high)
        for i in range(N):
            bp = x_left + i * DELTA_T
            if s_low < bp < s_high:
                breakpoints.add(bp)
        for j in range(N):
            bp = t - x_left - j * DELTA_T
            if s_low < bp < s_high:
                breakpoints.add(bp)
        
        breakpoints = sorted(breakpoints)
        val = 0.0
        for k in range(len(breakpoints) - 1):
            a = breakpoints[k]
            b = breakpoints[k+1]
            if b <= a:
                continue
            mid = (a + b) / 2
            # Determine i for mid
            i_idx = int((mid - x_left) / DELTA_T)
            if i_idx >= N:
                i_idx = N - 1
            if i_idx < 0:
                i_idx = 0
            # Determine j for mid
            t_minus_mid = t - mid
            j_idx = int((t_minus_mid - x_left) / DELTA_T)
            if j_idx >= N:
                j_idx = N - 1
            if j_idx < 0:
                j_idx = 0
            val += h[i_idx] * h[j_idx] * (b - a)
        conv[idx] = val
    
    return t_grid, conv

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    For a piecewise-constant function with step heights h on N intervals of width DELTA_T:
    - ||f||_2^2 = sum h[i]^2 * DELTA_T
    - ||f||_1 = sum h[i] * DELTA_T  
    - ||f*f||_inf = max of the autoconvolution
    """
    if np.max(h) <= 0:
        return 0.0
    l2_sq = np.sum(h**2) * DELTA_T
    l1 = np.sum(h) * DELTA_T
    t_grid, conv = _convolution(h)
    conv_max = np.max(conv)
    if conv_max <= 0 or l1 <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Initial candidate: uniform
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_score = _score(h_uniform)
    
    # Analytical insight: for a two-spike distribution with equal weights,
    # the score can be higher. Let's try various structured starts.
    
    candidates = [
        np.ones(N),  # uniform
    ]
    
    # Two adjacent spikes
    for pos in [0, 1, 2, 5, 10, 24, 25]:
        h = np.zeros(N)
        if pos < N-1:
            h[pos] = 1.0
            h[pos+1] = 1.0
        candidates.append(h)
    
    # Single spike
    h = np.zeros(N)
    h[0] = 1.0
    candidates.append(h)
    
    # Three adjacent spikes
    for pos in [0, 1, 24]:
        h = np.zeros(N)
        if pos+2 < N:
            h[pos] = 1.0
            h[pos+1] = 1.0
            h[pos+2] = 1.0
        candidates.append(h)
    
    # Triangular-ish
    for center in [24, 25]:
        h = np.zeros(N)
        width = 5
        for i in range(max(0, center-width), min(N, center+width+1)):
            h[i] = 1.0 - abs(i - center) / width
        if np.max(h) > 0:
            candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local search using scipy minimize (minimize negative score)
    def neg_score(h_flat):
        h = np.maximum(h_flat, 0.0)  # enforce non-negativity
        if np.max(h) <= 0:
            return 0.0
        # Normalize to avoid scale issues - actually the ratio is scale-invariant!
        # ||f||_2^2 / (||f||_1 * ||f*f||_inf) is invariant to scaling of f.
        # So we can normalize to ||f||_1 = 1 or ||f||_2 = 1 for numerical stability.
        norm = np.sum(h)
        if norm <= 0:
            return 0.0
        h_norm = h / norm
        return -_score(h_norm)
    
    # Try gradient-based optimization from the best candidate
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        norm = np.sum(h0)
        if norm > 0:
            h0 = h0 / norm
        
        try:
            res = minimize(
                neg_score,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            if time.time() < deadline:
                h_opt = np.maximum(res.x, 0.0)
                norm_opt = np.sum(h_opt)
                if norm_opt > 0:
                    h_opt = h_opt / norm_opt
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try a few more Nelder-Mead runs from different starts if time permits
    if time.time() < deadline - 20:
        for start_idx, h_start in enumerate(candidates[:5]):
            if time.time() > deadline - 5:
                break
            h0 = h_start.copy()
            norm = np.sum(h0)
            if norm <= 0:
                continue
            h0 = h0 / norm
            try:
                res = minimize(
                    neg_score,
                    h0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                )
                h_opt = np.maximum(res.x, 0.0)
                norm_opt = np.sum(h_opt)
                if norm_opt > 0:
                    h_opt = h_opt / norm_opt
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Return as list of nonnegative heights
    result = np.maximum(best_h, 0.0).tolist()
    return result
# EVOLVE-BLOCK-END
