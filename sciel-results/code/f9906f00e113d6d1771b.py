"""
Second autocorrelation inequality:
Maximize (||h||_2^2) / (||h||_1 * sup_k |(h*h)_k|) over nonnegative step heights h of length N=50
on equal intervals of [-1/4, 1/4].

Strategy:
- The autoconvolution (h*h)_k = sum_i h_i h_{i+k} (for appropriate indexing).
- We want to maximize ||h||_2^2 / (||h||_1 * max_k |(h*h)_k|).
- Note that (h*h)_0 = sum h_i^2 = ||h||_2^2, so the ratio is at most 1 if the max is at lag 0.
- To exceed 1, we need the autoconvolution peak to be at a nonzero lag, or to reduce the relative
  contribution of lag 0 compared to other lags... wait, but (h*h)_0 is always the largest for
  nonnegative h? Actually no - for nonnegative h, (h*h)_0 = sum h_i^2, and (h*h)_k for k != 0
  involves cross terms. By Cauchy-Schwarz, (h*h)_k <= sqrt(sum_{i} h_i^2 * sum_{i+k} h_{i+k}^2) <= sum h_i^2.
  So for nonnegative h, the max of |h*h| is always at lag 0, equal to ||h||_2^2.
  
  Wait, that means the ratio is always exactly 1/||h||_1 * ||h||_2^2 / ||h||_2^2 = 1/||h||_1?
  
  No wait. Let me re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
  
  So it's: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
  
  For nonnegative h, (h*h)_0 = sum h_i^2 = ||h||_2^2. And for k != 0, (h*h)_k = sum_i h_i h_{i+k} <= (sum h_i^2)(sum h_{i+k}^2)^(1/2) by C-S, but more directly, since all terms are nonnegative, (h*h)_k <= (h*h)_0 because... actually not necessarily. 
  
  Example: h = [1, 1, 0, 0]. (h*h)_0 = 2, (h*h)_1 = 1, (h*h)_-1 = 1. Max is 2.
  h = [1, 0, 1, 0]. (h*h)_0 = 2, (h*h)_2 = 1, (h*h)_-2 = 1. Max is 2.
  
  Actually for nonnegative sequences, (h*h)_0 = sum h_i^2, and (h*h)_k for k≠0 involves products of different elements. By rearrangement or just noting that sum_i h_i h_{i+k} <= sum_i h_i^2 when... hmm, not always true.
  
  Consider h = [1, 2, 1, 0]. (h*h)_0 = 1+4+1 = 6. (h*h)_1 = 1*2 + 2*1 = 4. (h*h)_2 = 1*1 = 1. Max = 6.
  
  Consider h = [1, 1, 1, 1]. (h*h)_0 = 4. (h*h)_1 = 3, (h*h)_2 = 2, (h*h)_3 = 1. Max = 4.
  
  It seems for nonnegative h, (h*h)_0 is always the maximum. Let me verify: (h*h)_0 = sum h_i^2. For k > 0, (h*h)_k = sum_{i=0}^{N-1-k} h_i h_{i+k}. We need to show this <= sum h_i^2.
  
  sum_{i} h_i h_{i+k} <= (1/2) sum_i (h_i^2 + h_{i+k}^2) <= (1/2)(sum h_i^2 + sum h_{i+k}^2) <= sum h_i^2.
  
  Yes! By AM-GM, h_i h_{i+k} <= (h_i^2 + h_{i+k}^2)/2. Summing over i, we get (h*h)_k <= (1/2)(sum h_i^2 + sum h_{i+k}^2) <= sum h_i^2 = (h*h)_0.
  
  So for nonnegative h, the supremum of |h*h| is always (h*h)_0 = ||h||_2^2.
  
  Therefore the ratio simplifies to: ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
  
  To maximize this, we minimize ||h||_1 subject to... wait, there's no constraint on the scale? The problem says "nonnegative heights" but doesn't specify a normalization. If we can scale h arbitrarily, then ||h||_1 can be made arbitrarily small (approaching 0), making the ratio arbitrarily large.
  
  Hmm, but the seed program returns [1.0]*50, giving score 1.0. Let me reconsider.
  
  Actually, re-reading: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
  
  Maybe it's: ||h||_2^2 / (||h||_1 * ||h*h||_inf) where the autoconvolution is normalized somehow? Or maybe the heights are constrained to sum to 1 or have some fixed norm?
  
  Given the seed returns all ones and scores 1.0, and the observed best is ~1.014, the ratio must be bounded. Perhaps there's an implicit constraint like sum(h) = 1 or max(h) = 1 or ||h||_2 = 1.
  
  If ||h||_1 = 1 (probabilistic interpretation), then ratio = ||h||_2^2 / ||h*h||_inf. For h = [1/50]*50, ||h||_2^2 = 50*(1/50)^2 = 1/50, ||h*h||_inf = (h*h)_0 = 1/50. Ratio = 1. That matches score 1.0!
  
  So the constraint is likely ||h||_1 = 1 (or equivalently, the ratio is scale-invariant and we can assume ||h||_1 = 1).
  
  With ||h||_1 = 1, ratio = ||h||_2^2 / ||h*h||_inf = (sum h_i^2) / max_k (sum_i h_i h_{i+k}).
  
  Since max_k is at k=0 for nonnegative h, ratio = (sum h_i^2) / (sum h_i^2) = 1. Always 1?!
  
  That can't be right either since we see 1.014. Let me reconsider the autoconvolution definition.
  
  Maybe the autoconvolution is circular? Or maybe it's defined differently. Or maybe the "supremum norm of autoconvolution" excludes lag 0?
  
  Or perhaps the autoconvolution is (h*h)_k = sum_i h_i h_{i+k} but with the understanding that for the "second autocorrelation inequality," we're looking at a specific lag or a normalized version.
  
  Given the task name "Second autocorrelation inequality" and that we need to maximize the ratio, and the seed gives 1.0, let me just implement a direct optimization that computes the ratio as stated and search for good configurations.

  Let me just code it up with a reasonable interpretation and use optimization.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _autoconv(h):
    """Compute full autoconvolution of h (linear, not circular)."""
    n = len(h)
    result = np.zeros(2*n - 1)
    for k in range(2*n - 1):
        # (h*h)_k = sum_i h_i h_{i+k} where indices are valid
        if k < n:
            # positive lag k: sum_{i=0}^{n-1-k} h[i] * h[i+k]
            result[k + n - 1] = np.dot(h[:n-k], h[k:])
        else:
            # negative lag: k >= n, let lag = k - (n-1), which is negative
            lag = k - (n - 1)  # negative
            result[k] = np.dot(h[-lag:], h[:lag])
    return result

def _autoconv_fast(h):
    """Faster autoconvolution using FFT."""
    n = len(h)
    size = 2*n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * np.conj(H), n=fft_size)[:size]
    return conv

def _score(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = _autoconv_fast(h)
    conv_max = np.max(np.abs(conv))
    if conv_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_max)

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try various structured initializations
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    tri = np.zeros(N)
    mid = N // 2
    for i in range(N):
        tri[i] = 1.0 - abs(i - mid) / mid
    candidates.append(np.maximum(tri, 0))
    
    # Cosine bump
    theta = np.linspace(-np.pi/2, np.pi/2, N)
    candidates.append(np.maximum(np.cos(theta), 0))
    
    # Two peaks
    two_peak = np.zeros(N)
    two_peak[:N//4] = 1.0
    two_peak[3*N//4:] = 1.0
    candidates.append(two_peak)
    
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    candidates.append(alt)
    
    # Random phase cosine
    for theta_val in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for phi in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            h = np.maximum(np.cos(theta_val * np.arange(N) + phi), 0)
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 30:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best candidate
    if time.time() < deadline - 40:
        # Normalize to l1=1 for optimization, since ratio is scale-invariant
        h0 = best_h / np.sum(best_h)
        
        def neg_score(x):
            x = np.maximum(x, 0)
            if np.sum(x) < 1e-15:
                return 0.0
            x = x / np.sum(x)
            return -_score(x)
        
        try:
            res = minimize(neg_score, h0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.success or time.time() < deadline - 20:
                h_opt = np.maximum(res.x, 0)
                if np.sum(h_opt) > 0:
                    s = _score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try a few more random restarts with local opt
    if time.time() < deadline - 30:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            h_rand = np.random.dirichlet(np.ones(N))
            def neg_score_r(x):
                x = np.maximum(x, 0)
                if np.sum(x) < 1e-15:
                    return 0.0
                x = x / np.sum(x)
                return -_score(x)
            try:
                res = minimize(neg_score_r, h_rand, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-9})
                h_opt = np.maximum(res.x, 0)
                if np.sum(h_opt) > 0:
                    s = _score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 10:
        h_polish = best_h / np.sum(best_h)
        def neg_score_p(x):
            x = np.maximum(x, 0)
            if np.sum(x) < 1e-15:
                return 0.0
            x = x / np.sum(x)
            return -_score(x)
        try:
            res = minimize(neg_score_p, h_polish, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 0:
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
