"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import sys

N = 50
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to reserve

# EVOLVE-BLOCK-START

def _autoconv_sup(h):
    """Compute the supremum norm of the autoconvolution of h."""
    # h is a 1D array of step heights on [-1/4, 1/4] with N equal intervals
    # The autoconvolution (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, we can compute this via convolution
    # The support of f is [-1/4, 1/4], so the autoconvolution support is [-1/2, 1/2]
    
    # Step function: h[i] on interval [x_i, x_{i+1}] where x_i = -1/4 + i*dx
    dx = (0.5) / N  # interval width = 1/2 / N = 1/(2N)
    
    # For step functions, the autoconvolution at a point t is:
    # (f*f)(t) = sum_i h[i] * integral over s where s in [x_i, x_{i+1}] and t-s in support
    # This is equivalent to the convolution of the piecewise constant function with itself.
    
    # Simpler approach: use discrete convolution with appropriate scaling
    # The continuous autoconvolution of step functions can be computed as:
    # (f*f)(t) = dx * (h * h)(k) where k corresponds to t, but we need to be careful
    
    # Actually, for step functions f(x) = h[i] for x in [x_i, x_{i+1}],
    # (f*f)(t) = integral_{-inf}^{inf} f(s) f(t-s) ds
    # = sum_i h[i] * integral_{s: s in [x_i, x_{i+1}], t-s in [-1/4, 1/4]} f(t-s) ds
    # = sum_i h[i] * integral_{max(x_i, t-1/4)}^{min(x_{i+1}, t+1/4)} f(t-s) ds
    
    # This is complex to compute analytically for all t. Let's use a numerical approach.
    # We'll evaluate the autoconvolution on a fine grid and take the max.
    
    # Alternative: use the fact that for nonnegative step functions, the autoconvolution
    # is a piecewise quadratic function. But let's just compute it numerically on a fine grid.
    
    # Grid for t in [-1/2, 1/2]
    n_grid = 2000
    t_grid = np.linspace(-0.5, 0.5, n_grid)
    
    # Compute (f*f)(t) for each t in t_grid
    # f(s) = h[i] where s in [x_i, x_{i+1}], x_i = -0.25 + i*dx
    # (f*f)(t) = integral f(s) f(t-s) ds over s where both s and t-s are in [-0.25, 0.25]
    
    # For each t, the integral is over s in [max(-0.25, t-0.25), min(0.25, t+0.25)]
    # Within this range, f(s) is piecewise constant with breakpoints at x_i
    # and f(t-s) is piecewise constant with breakpoints at x_j + t
    
    # This is getting complex. Let's use a simpler numerical method:
    # Sample f on a fine grid and use FFT-based convolution, then interpolate.
    
    # Actually, let's just do a direct numerical integration on a fine grid.
    n_fine = 10000
    s_fine = np.linspace(-0.25, 0.25, n_fine)
    ds = s_fine[1] - s_fine[0]
    
    # f(s) on the fine grid
    # s_fine[i] is in interval [x_i, x_{i+1}] where x_i = -0.25 + i*dx
    # index = floor((s_fine[i] + 0.25) / dx)
    idx = np.floor((s_fine + 0.25) / dx).astype(int)
    idx = np.clip(idx, 0, N-1)
    f_fine = h[idx]
    
    # For each t in t_grid, compute integral f(s) f(t-s) ds
    # t-s must be in [-0.25, 0.25], so s in [t-0.25, t+0.25]
    # Also s in [-0.25, 0.25]
    
    # This is O(n_grid * n_fine) which might be slow. Let's use a different approach.
    
    # Use FFT-based convolution:
    # f(s) * f(s) in continuous sense. Discretize f on a grid that's fine enough.
    # Let's use a grid of size 2^n for FFT efficiency.
    
    n_fft = 1 << 15  # 32768
    # Create a grid that covers [-0.25, 0.25] with n_fft points
    # But we need the convolution to cover [-0.5, 0.5]
    # Let's place the function on a grid and use FFT convolution
    
    # Grid for f: from -0.25 to 0.25, n_fft points
    # Actually, let's think about this differently.
    
    # The autoconvolution (f*f)(t) for t in [-0.5, 0.5] can be computed as:
    # (f*f)(t) = integral_{-0.25}^{0.25} f(s) f(t-s) ds
    # where f(t-s) = 0 if t-s not in [-0.25, 0.25]
    
    # Let's discretize s on a fine grid and for each t, do a trapezoidal rule.
    # To make this fast, we can use the fact that this is a convolution.
    
    # Let's use a grid of size M for s, and compute the convolution using FFT.
    M = 1 << 14  # 16384
    s_grid = np.linspace(-0.25, 0.25, M, endpoint=False)
    ds_grid = s_grid[1] - s_grid[0]
    
    # f on s_grid
    idx_s = np.floor((s_grid + 0.25) / dx).astype(int)
    idx_s = np.clip(idx_s, 0, N-1)
    f_s = h[idx_s]
    
    # For the convolution, we need f(t-s). If we define g(u) = f(u), then
    # (f*f)(t) = integral f(s) f(t-s) ds = (f * f_rev)(t) where f_rev(s) = f(-s)
    # Actually, (f*f)(t) = integral f(s) f(t-s) ds. Let u = t-s, then s = t-u, ds = -du
    # = integral f(t-u) f(u) du = (f * f)(t) (convolution)
    
    # So we just need the convolution of f with itself.
    # Use FFT:
    f_pad = np.zeros(n_fft)
    f_pad[:M] = f_s
    f_fft = np.fft.rfft(f_pad, n_fft)
    conv_fft = f_fft * f_fft
    conv = np.fft.irfft(conv_fft, n_fft) * ds_grid
    
    # The convolution result: conv[k] approximates (f*f)(t_k) where t_k corresponds to the k-th sample
    # The grid for t: t_k = (k - M) * ds_grid for k in [0, n_fft)
    # Actually, the convolution of two length-M sequences (zero-padded to n_fft) gives
    # a result where the k-th element corresponds to the sum over j of f[j] f[k-j]
    # which approximates (f*f)(t) where t = (k - M + 1) * ds_grid... let me think.
    
    # Standard convolution: if a and b are sampled at points x_i = i*ds, then
    # (a*b)[k] = sum_j a[j] b[k-j] ds approximates integral a(x) b(t-x) dx where t = k*ds
    # But we have a and b both being f sampled at s_grid points.
    
    # Let's just evaluate the autoconvolution on the t_grid using the fine grid directly.
    # To keep it fast, use vectorized operations.
    
    # For each t in t_grid, the integral is over s in [max(-0.25, t-0.25), min(0.25, t+0.25)]
    # Let's use the fine grid and mask.
    
    # Actually, let's just use the FFT result and interpolate.
    # The convolution conv[k] for k in [0, n_fft) corresponds to t values.
    # When we convolve f_pad (length n_fft) with itself, the result has length n_fft.
    # The k-th element of the linear convolution of two length-n_fft sequences is
    # sum_{j=0}^{n_fft-1} f_pad[j] f_pad[k-j] (with wraparound for circular, but we zero-padded)
    # Actually, FFT gives circular convolution. Since we zero-padded to n_fft >= 2*M-1,
    # the first M-1 + M-1 elements are the linear convolution.
    
    # The linear convolution c[k] = sum_{j=0}^{k} f[j] g[k-j] for k in [0, 2*M-2]
    # where f and g are the original sequences of length M.
    # In our case, f = g = f_s (length M).
    # c[k] = sum_{j=0}^{min(k, M-1)} f_s[j] f_s[k-j] (with appropriate bounds)
    
    # This approximates (f*f)(t_k) where t_k = (k - (M-1)) * ds_grid? 
    # Let me think: f_s[j] is f(s_j) where s_j = -0.25 + j*ds_grid
    # (f*f)(t) = integral f(s) f(t-s) ds
    # If we discretize: sum_j f(s_j) f(t - s_j) ds_grid
    # For t = t_k, we need f(t_k - s_j). 
    # t_k - s_j = t_k + 0.25 - j*ds_grid
    # We need this to be in [-0.25, 0.25], so 0 <= t_k + 0.25 - j*ds_grid < 0.5
    # i.e., (t_k + 0.25)/ds_grid - 0.5 < j <= (t_k + 0.25)/ds_grid
    
    # This is getting complicated. Let's just do a direct computation on a moderate grid.
    
    # Use a grid of size 2000 for t and 4000 for s, and do it with numpy operations.
    n_t = 2000
    n_s = 4000
    t_eval = np.linspace(-0.5, 0.5, n_t)
    s_eval = np.linspace(-0.25, 0.25, n_s)
    ds_eval = s_eval[1] - s_eval[0]
    
    # f(s_eval)
    idx_s_eval = np.floor((s_eval + 0.25) / dx).astype(int)
    idx_s_eval = np.clip(idx_s_eval, 0, N-1)
    f_s_eval = h[idx_s_eval]
    
    # For each t, compute integral f(s) f(t-s) ds
    # t-s must be in [-0.25, 0.25]
    # Create a 2D array: t_eval[:, None] - s_eval[None, :]
    ts = t_eval[:, None] - s_eval[None, :]
    # f(t-s) where t-s in [-0.25, 0.25]
    # index for t-s
    idx_ts = np.floor((ts + 0.25) / dx).astype(int)
    # Mask where t-s is in [-0.25, 0.25]
    mask = (ts >= -0.25) & (ts <= 0.25)
    idx_ts = np.where(mask, idx_ts, 0)
    idx_ts = np.clip(idx_ts, 0, N-1)
    f_ts = h[idx_ts] * mask
    
    # (f*f)(t) = sum_s f(s) f(t-s) ds
    autoconv = np.sum(f_s_eval[None, :] * f_ts, axis=1) * ds_eval
    sup_norm = np.max(autoconv)
    
    return sup_norm


def _score(h):
    """Compute the score: (L2 norm)^2 / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    sup_auto = _autoconv_sup(h)
    if sup_auto == 0:
        return 0.0
    return l2_sq / (l1 * sup_auto)


def _score_negative(h):
    """Negative score for minimization."""
    return -_score(h)


def construct():
    """Construct the optimal nonnegative step heights."""
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Initial guess: spike-like profile
    # Try a few initial profiles
    best_h = None
    best_score = -np.inf
    
    # Profile 1: uniform
    h_uniform = np.ones(N)
    s = _score(h_uniform)
    if s > best_score:
        best_score = s
        best_h = h_uniform.copy()
    
    # Profile 2: spike in center
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    s = _score(h_spike)
    if s > best_score:
        best_score = s
        best_h = h_spike.copy()
    
    # Profile 3: narrow block in center
    h_block = np.zeros(N)
    center = N // 2
    width = 3
    h_block[center - width//2 : center + width//2 + 1] = 1.0
    s = _score(h_block)
    if s > best_score:
        best_score = s
        best_h = h_block.copy()
    
    # Profile 4: Gaussian-like
    x = np.arange(N)
    h_gauss = np.exp(-0.5 * ((x - N/2) / (N/6)) ** 2)
    s = _score(h_gauss)
    if s > best_score:
        best_score = s
        best_h = h_gauss.copy()
    
    # Profile 5: triangular
    h_tri = np.zeros(N)
    h_tri[N//2] = 1.0
    for i in range(1, N//2 + 1):
        h_tri[N//2 - i] = 1.0 - i / (N//2)
        h_tri[N//2 + i] = 1.0 - i / (N//2)
    s = _score(h_tri)
    if s > best_score:
        best_score = s
        best_h = h_tri.copy()
    
    # Now try to optimize starting from the best initial profile
    if time.time() > deadline:
        return list(best_h)
    
    # Use Nelder-Mead optimization
    # Constraint: h >= 0 (nonnegative)
    # We'll use a penalty approach or project onto nonnegative orthant
    
    def objective_with_penalty(h):
        h_clipped = np.clip(h, 0, None)
        penalty = np.sum((np.clip(h, None, 0)) ** 2) * 100
        return _score_negative(h_clipped) + penalty
    
    # Try optimizing from the best starting point
    if best_h is not None and time.time() < deadline:
        try:
            res = minimize(
                objective_with_penalty,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            h_opt = np.clip(res.x, 0, None)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try coordinate descent refinement
    if time.time() < deadline:
        h_cd = best_h.copy()
        improved = True
        max_iters = 200
        it = 0
        while improved and it < max_iters and time.time() < deadline:
            improved = False
            it += 1
            for i in range(N):
                # Try adjusting h[i]
                old_val = h_cd[i]
                # Try a few values around the current one
                candidates = [max(0, old_val * 0.5), max(0, old_val * 0.8), old_val, 
                             old_val * 1.2, old_val * 1.5, old_val + 0.1, old_val + 0.5]
                # Also try 0
                candidates.append(0.0)
                # Filter out negative
                candidates = [max(0, c) for c in candidates]
                best_cand = old_val
                best_cand_score = _score(h_cd)
                for c in candidates:
                    h_test = h_cd.copy()
                    h_test[i] = c
                    s_test = _score(h_test)
                    if s_test > best_cand_score:
                        best_cand = c
                        best_cand_score = s_test
                if best_cand != old_val:
                    h_cd[i] = best_cand
                    improved = True
            if _score(h_cd) > best_score:
                best_score = _score(h_cd)
                best_h = h_cd.copy()
    
    # Final polish: use scipy minimize with tighter tolerances
    if time.time() < deadline - 5:
        try:
            def objective_smooth(h):
                h_clipped = np.clip(h, 0, None)
                return _score_negative(h_clipped)
            
            res2 = minimize(
                objective_smooth,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_polish = np.clip(res2.x, 0, None)
            s_polish = _score(h_polish)
            if s_polish > best_score:
                best_score = s_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.clip(best_h, 0, None)
    
    return list(best_h)

# EVOLVE-BLOCK-END
