import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    # Step function: width dw = 1/50, centered on [-1/4, 1/4]
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    centers = (x[:-1] + x[1:]) / 2.0
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution: (f * f)(t) = ∫ f(s) f(t-s) ds
    # For step functions, we can compute this on a fine grid
    # The support of f is [-1/4, 1/4], so support of f*f is [-1/2, 1/2]
    
    # Use FFT-based convolution for efficiency
    # Pad to avoid circular convolution
    n_fft = 2 * N + 100
    f = np.zeros(n_fft)
    f[:N] = h
    
    # Linear convolution via FFT
    F = np.fft.fft(f, n=2 * n_fft)
    conv = np.fft.ifft(F * F).real[:2 * N]
    
    # The convolution sum gives us Σ f_i f_j for each lag
    # For the step function, the actual continuous convolution at lag k is:
    # (f*f)(k*dw) = dw * Σ_i f_i * f_{i+k} (approximately, for small dw)
    # More precisely, for step functions of width dw:
    # (f*f)(t) = dw * Σ_i f_i f_j where |t - (x_i - x_j)| < dw/2
    
    # Actually, let's compute it more carefully.
    # f(x) = h_i for x in [x_i, x_{i+1})
    # (f*f)(t) = ∫ f(s) f(t-s) ds
    # For t = k*dw, this involves pairs (i,j) where the overlap of intervals matters.
    
    # Simpler approach: sample the autoconvolution on a fine grid
    t_fine = np.linspace(-0.5, 0.5, 4 * N)
    dt = t_fine[1] - t_fine[0]
    
    autoconv = np.zeros_like(t_fine)
    for idx, t in enumerate(t_fine):
        # f(s) * f(t-s): s ranges over [-1/4, 1/4], t-s ranges over [-1/4, 1/4]
        # So s ranges over [max(-1/4, t-1/4), min(1/4, t+1/4)]
        s_lo = max(-0.25, t - 0.25)
        s_hi = min(0.25, t + 0.25)
        if s_lo >= s_hi:
            autoconv[idx] = 0.0
            continue
        
        # Integrate by summing over the step intervals
        # For each interval i where f = h_i, find overlap with [s_lo, s_hi]
        # and where f(t-s) is nonzero, i.e., t-s in [-1/4, 1/4], i.e., s in [t-1/4, t+1/4]
        total = 0.0
        for i in range(N):
            a = x[i]
            b = x[i+1]
            # Overlap of [a,b] with [s_lo, s_hi] and [t-0.25, t+0.25]
            lo = max(a, s_lo, t - 0.25)
            hi = min(b, s_hi, t + 0.25)
            if lo < hi:
                # f(s) = h_i, f(t-s): t-s in [t-hi, t-lo]
                # Find which interval j has t-s in [x_j, x_{j+1})
                ts_lo = t - hi
                ts_hi = t - lo
                # Sum over j where [x_j, x_{j+1}] overlaps [ts_lo, ts_hi]
                for j in range(N):
                    c = x[j]
                    d = x[j+1]
                    olo = max(c, ts_lo)
                    ohi = min(d, ts_hi)
                    if olo < ohi:
                        total += h[i] * h[j] * (ohi - olo)
        autoconv[idx] = total
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)


def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start from known good solution
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _compute_score(best_h)
    
    # Local optimization using Nelder-Mead or BFGS
    # Use SLSQP for better handling of the non-smooth max
    
    try:
        # First, try BFGS with bounds
        bounds = [(0, 10)] * N
        result = minimize(_score_neg, h0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if time.time() < deadline:
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
    except Exception:
        pass
    
    # Try Nelder-Mead from multiple starting points
    if time.time() < deadline:
        starts = [
            h0,
            np.array([1.0/3.0, 1.0/3.0, 1.0/3.0] + [0.0]*(N-3)),
            np.array([0.5, 0.5] + [0.0]*(N-2)),
            np.array([1.0] + [0.0]*(N-1)),
        ]
        
        for s in starts:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_score_neg, s, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                h_trial = np.maximum(result.x, 0.0)
                score_trial = _compute_score(h_trial)
                if score_trial > best_score:
                    best_h = h_trial
                    best_score = score_trial
            except Exception:
                pass
    
    # Try optimizing with different support sizes
    if time.time() < deadline:
        for K in [2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            # Try consecutive support
            h_init = np.zeros(N)
            h_init[:K] = 1.0 / K
            try:
                result = minimize(_score_neg, h_init, method='L-BFGS-B',
                                 bounds=[(0, 10)] * N,
                                 options={'maxiter': 300, 'ftol': 1e-12})
                h_trial = np.maximum(result.x, 0.0)
                score_trial = _compute_score(h_trial)
                if score_trial > best_score:
                    best_h = h_trial
                    best_score = score_trial
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    return list(best_h)
