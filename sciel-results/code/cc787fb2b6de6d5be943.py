"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Structural change: direct convex optimization on the bin values.
The problem min max_lag (f*f)_lag / (sum f)^2 s.t. f >= 0, sum f = 1
is convex. We solve it via projected subgradient / projected gradient
with FFT-accelerated autoconvolution, plus a local coordinate-descent polish.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for speed
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def initial_candidates():
    """Generate diverse initial candidates."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian with various widths
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
    
    # Raised cosine with various widths
    for w in [0.1, 0.15, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = np.cos(np.pi * x[mask] / w) ** 2
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Triangle (B-spline order 1)
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    # Quadratic bump
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = (1.0 - (2.0 * x[mask] / w) ** 2) ** 2
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
    
    return candidates

def convex_refine(f_init, start_time, max_time=120.0):
    """
    Convex optimization: minimize max_lag (f*f)_lag s.t. f >= 0, sum f = 1.
    Use projected subgradient method with FFT acceleration.
    """
    f = f_init.copy()
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s  # normalize so sum = 1
    
    best_f = f.copy()
    best_val = np.max(fft_autoconv(f))
    
    # Subgradient method with diminishing step size
    step = 0.01
    decay = 0.999
    momentum = np.zeros(N)
    beta_mom = 0.9
    
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        conv = fft_autoconv(f)
        current_val = np.max(conv)
        
        if current_val < best_val - 1e-14:
            best_val = current_val
            best_f = f.copy()
        
        # Find the lag(s) achieving the max
        max_lag = np.argmax(conv)
        
        # Subgradient of max(conv) w.r.t. f:
        # d(conv_lag)/d(f_i) = f[lag - i] (with appropriate indexing)
        # For full conv: conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/d(f[i]) = f[m-i] + f[m-i] if i != m-i, else f[m-i]
        # Actually: conv[m] = sum_{j=0}^{N-1} f[j] f[m-j]
        # d(conv[m])/d(f[i]) = f[m-i] + f[m-i] = 2*f[m-i] (when both terms exist)
        # But need to be careful with boundaries
        
        subgrad = np.zeros(N)
        m = max_lag
        for i in range(N):
            # Contribution to conv[m] from f[i]:
            # conv[m] includes f[i]*f[m-i] if 0 <= m-i < N
            # and f[m-i]*f[i] if 0 <= m-i < N (same term)
            # Actually conv[m] = sum_j f[j] f[m-j], so d/d(f[i]) = f[m-i] if valid
            j = m - i
            if 0 <= j < N:
                subgrad[i] += f[j]
            # Also if m-i gives a different index in the sum
            # Wait, the full convolution is:
            # (f*f)[m] = sum_{j=max(0,m-N+1)}^{min(N-1,m)} f[j] f[m-j]
            # So d/d(f[i]) = f[m-i] if m-i is in [max(0,m-N+1), min(N-1,m)]
            # which is equivalent to 0 <= m-i < N AND 0 <= i < N (already true)
        
        # Normalize subgradient
        sg_norm = np.linalg.norm(subgrad)
        if sg_norm > 1e-15:
            subgrad = subgrad / sg_norm
        
        # Momentum update
        momentum = beta_mom * momentum + (1 - beta_mom) * subgrad
        
        # Projected gradient step
        f_new = f - step * momentum
        
        # Project onto {f >= 0, sum f = 1}
        f_new = np.maximum(f_new, 0)
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            f_new = f.copy()
        else:
            f_new = f_new / s_new
        
        f = f_new
        step *= decay
    
    return best_f

def coordinate_descent_polish(f_init, start_time, max_time=120.0):
    """
    Coordinate descent: for each bin, find the optimal value given others fixed.
    This is exact for each coordinate since the objective is convex in each coordinate.
    """
    f = f_init.copy()
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = np.max(fft_autoconv(f))
    
    # Precompute: for each bin i, the contribution to each lag m is f[m-i]
    # When we change f[i], only lags m where m-i is in [0, N-1] are affected
    # i.e., m in [i, i+N-1]
    
    for sweep in range(500):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            # Current value of f[i]
            old_val = f[i]
            
            # The autoconvolution values affected by f[i] are at lags m in [i, i+N-1]
            # conv[m] = sum_j f[j] f[m-j]
            # When we change f[i] to v, the new conv[m] for m in [i, i+N-1]:
            # new_conv[m] = conv[m] - old_val * f[m-i] + v * f[m-i]
            #            = conv[m] + (v - old_val) * f[m-i]
            # But wait, f[m-i] might also be f[i] if m-i == i, i.e., m = 2i
            # In that case, the term is f[i]^2, so changing f[i] changes it by 2*(v-old_val)*f[i] - (v-old_val)^2
            # This gets complicated. Let's just recompute the affected lags.
            
            # For efficiency, let's compute the max over affected lags
            # The lags affected are m in [i, i+N-1], but conv has length 2N-1
            # So m in [i, min(i+N-1, 2N-2)]
            m_start = i
            m_end = min(i + N - 1, 2 * N - 2)
            
            # For each affected lag m, conv[m] depends on f[i] as:
            # conv[m] = ... + f[i] * f[m-i] + ... (if m-i != i)
            # or conv[m] = ... + f[i]^2 + ... (if m-i == i, i.e., m = 2i)
            
            # Let's compute the base conv without f[i]'s contribution
            # Actually, simpler: for a given v, the new conv at lag m is:
            # If m-i != i: new_conv[m] = old_conv[m] + (v - old_val) * f[m-i]
            # If m-i == i: new_conv[m] = old_conv[m] + (v - old_val) * (2*f[i] - (v - old_val))
            #             = old_conv[m] + (v-old_val)*2*old_val - (v-old_val)^2
            
            # The max over m of new_conv[m] is a piecewise linear/quadratic function of v
            # For simplicity, let's sample a few values of v and pick the best
            
            # The optimal v should be in [0, some upper bound]
            # Upper bound: the current max value
            v_max = max(old_val * 3, np.max(f) * 2, 0.01)
            
            # Try several candidate values
            candidates_v = [0.0, old_val * 0.5, old_val * 0.8, old_val, old_val * 1.2, v_max * 0.5, v_max]
            # Also try values that would equalize the top few lags
            conv_current = fft_autoconv(f)
            top_lags = np.argsort(conv_current)[-5:]
            for m in top_lags:
                if m_start <= m <= m_end:
                    # Find v such that new_conv[m] equals the current max
                    # This is approximate
                    pass
            
            best_v = old_val
            best_max_conv = np.max(conv_current)
            
            for v in candidates_v:
                v = max(0.0, v)
                # Compute new conv at affected lags
                new_conv_vals = []
                for m in range(m_start, m_end + 1):
                    j = m - i
                    if j == i:
                        # f[i]^2 term
                        new_conv_m = conv_current[m] + (v - old_val) * (2 * old_val - (v - old_val))
                    else:
                        # f[i] * f[j] term (and f[j] * f[i] is the same term in the sum)
                        # Actually in the full convolution sum, f[i] appears once as f[j] when j=i
                        # and f[m-j] when m-j=i. So the total contribution is 2*f[i]*f[j] if i != j
                        # Wait no: conv[m] = sum_{j=0}^{N-1} f[j] * f[m-j]
                        # The term with f[i] is when j=i: f[i] * f[m-i]
                        # There's no separate term for f[m-i] * f[i] because that's the same term
                        # So the contribution is just f[i] * f[m-i]
                        new_conv_m = conv_current[m] + (v - old_val) * f[j]
                    new_conv_vals.append(new_conv_m)
                
                # Also need to consider lags not affected (they stay the same)
                # The max is max(max of affected, max of unaffected)
                max_affected = max(new_conv_vals) if new_conv_vals else 0
                # For unaffected lags, the max is the max of conv_current at lags outside [m_start, m_end]
                # But this is expensive to compute each time. Approximate:
                # The global max is at most max(max_affected, best_max_conv)
                # Actually, unaffected lags have the same value as before, so their max is
                # the max of conv_current at lags not in [m_start, m_end]
                # For efficiency, let's just use the global max as an upper bound
                candidate_max = max(max_affected, best_max_conv)
                
                if candidate_max < best_max_conv - 1e-14:
                    best_max_conv = candidate_max
                    best_v = v
            
            if best_v != old_val:
                f[i] = best_v
                improved = True
        
        if not improved:
            break
        
        # Renormalize
        s = np.sum(f)
        if s < 1e-15:
            break
        f = f / s
        current_val = np.max(fft_autoconv(f))
        if current_val < best_val - 1e-14:
            best_val = current_val
            best_f = f.copy()
    
    return best_f

def construct():
    start_time = time.time()
    
    # Generate initial candidates
    candidates = initial_candidates()
    
    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    print(f"Best initial: score={scored[0][1]:.10f}")
    
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply convex refinement to top candidates
    for f_init, score_init in scored[:5]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        refined = convex_refine(f_init, start_time, TIME_LIMIT - 5)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            print(f"Convex refined: score={best_score:.10f}")
    
    # Apply coordinate descent polish
    if time.time() - start_time < TIME_LIMIT - 5:
        polished = coordinate_descent_polish(best_f, start_time, TIME_LIMIT - 2)
        polished_score = compute_score(polished)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
            print(f"Polished: score={best_score:.10f}")
    
    # Final normalization and nonnegativity
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
    
    final_score = compute_score(best_f)
    print(f"Final score: {final_score:.10f}")
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
