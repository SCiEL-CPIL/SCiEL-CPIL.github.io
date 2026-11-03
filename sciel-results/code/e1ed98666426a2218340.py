import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _smooth_max(conv, beta):
    """Log-sum-exp smooth approximation of max."""
    m = np.max(conv)
    if m < 0:
        return 0.0
    shifted = conv / beta
    shifted -= np.max(shifted)
    return beta * (np.max(conv) / beta + np.log(np.sum(np.exp(shifted))))

def _smooth_max_grad(conv, beta):
    """Gradient of smooth max w.r.t. conv."""
    m = np.max(conv)
    if m < 0:
        return np.zeros_like(conv)
    shifted = conv / beta
    shifted -= np.max(shifted)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)

def _smooth_obj_and_grad(heights, beta):
    """Compute smooth objective and gradient for optimization.
    Uses log-sum-exp to smooth the max in autoconvolution.
    """
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf, np.zeros(n)
    
    # Normalize to sum=1 for scale-invariant computation
    h_norm = heights / total
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h_norm
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Smooth max
    sm = _smooth_max(conv, beta)
    sm_grad = _smooth_max_grad(conv, beta)
    
    # Score for normalized heights
    score = sm * n / w
    
    # Gradient of smooth max w.r.t. h_norm
    # d(sm)/d(h_norm[i]) = sum_k sm_grad[k] * d(conv[k])/d(h_norm[i])
    # d(conv[k])/d(h_norm[i]) = h_norm[k-i] + h_norm[i-k] (with bounds)
    
    # Vectorized: for each i, sum over k of sm_grad[k] * (h_norm[k-i] + h_norm[i-k])
    # This is essentially a convolution of sm_grad with h_norm
    grad_h_norm = np.zeros(n)
    
    # conv[k] = sum_j h_norm[j] * h_norm[k-j]
    # d(conv[k])/d(h_norm[i]) = h_norm[k-i] (if valid) + h_norm[i-k] (if valid)
    # sum_k sm_grad[k] * h_norm[k-i] = (sm_grad * h_norm)[i] in circular conv sense
    # But we need linear, so let's do it via FFT or direct
    
    # Using the fact that sum_k sm_grad[k] * h_norm[k-i] for k-i in [0,n-1]
    # = correlation of sm_grad and h_norm
    
    # Direct computation: grad_h_norm[i] = sum_{k: 0<=k-i<n} sm_grad[k] * h_norm[k-i] + sum_{k: 0<=i-k<n} sm_grad[k] * h_norm[i-k]
    
    # First term: for fixed i, k ranges from i to i+n-1, and k-i from 0 to n-1
    # So sum_{j=0}^{n-1} sm_grad[i+j] * h_norm[j] for i+j < len(sm_grad)
    # sm_grad has length M_FFT, but only first 2n-1 entries matter for linear conv
    
    sg = sm_grad[:2*n-1]  # relevant part
    
    # Term 1: sum_j h_norm[j] * sg[i+j] for valid indices
    for i in range(n):
        j_max = min(n, len(sg) - i)
        if j_max > 0:
            grad_h_norm[i] += np.dot(h_norm[:j_max], sg[i:i+j_max])
    
    # Term 2: sum_j h_norm[j] * sg[i-j] for valid indices where i-j >= 0
    # i-j >= 0 => j <= i, and i-j < len(sg)
    for i in range(n):
        j_max = min(n, i + 1)
        if j_max > 0:
            # sg[i-j] for j=0..j_max-1, so sg[i], sg[i-1], ..., sg[i-j_max+1]
            idx = np.arange(i, i - j_max, -1)
            valid = (idx >= 0) & (idx < len(sg))
            if np.any(valid):
                grad_h_norm[i] += np.dot(h_norm[:j_max][valid], sg[idx[valid]])
    
    # d(score)/d(h_norm) = (n/w) * grad_h_norm
    grad_norm = (n / w) * grad_h_norm
    
    # Chain rule: h_norm = heights / total
    sum_grad_h = np.dot(grad_norm, heights)
    grad = grad_norm / total - heights / (total * total) * sum_grad_h
    
    return score, grad

def _two_block_optimize():
    """Optimize a two-block step function: [a]*[h1]*[0]*[h2]*[a] with variable params."""
    # Parameters: left_len, right_len, gap_len, h1, h2
    # Total N = left_len + gap_len + right_len
    # We'll optimize over continuous parameters then round
    
    def obj(params):
        left_frac, gap_frac, h1, h2 = params
        left_frac = np.clip(left_frac, 0.01, 0.45)
        gap_frac = np.clip(gap_frac, 0.0, 0.4)
        h1 = max(h1, 0.01)
        h2 = max(h2, 0.01)
        
        left_len = int(left_frac * N)
        gap_len = int(gap_frac * N)
        right_len = N - left_len - gap_len
        if right_len <= 0:
            return 1e6
        
        h = np.zeros(N)
        h[:left_len] = h1
        h[left_len + gap_len:] = h2
        
        return compute_score(h)
    
    # Try multiple starting points
    best_val = 1e6
    best_params = None
    
    starts = [
        [0.3, 0.1, 1.0, 1.0],
        [0.2, 0.2, 1.0, 1.0],
        [0.4, 0.05, 1.0, 1.0],
        [0.25, 0.15, 0.8, 1.2],
        [0.35, 0.1, 1.2, 0.8],
        [0.2, 0.3, 1.0, 1.0],
    ]
    
    for s in starts:
        try:
            res = minimize(obj, s, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is not None:
        left_frac, gap_frac, h1, h2 = best_params
        left_frac = np.clip(left_frac, 0.01, 0.45)
        gap_frac = np.clip(gap_frac, 0.0, 0.4)
        h1 = max(h1, 0.01)
        h2 = max(h2, 0.01)
        
        left_len = int(left_frac * N)
        gap_len = int(gap_frac * N)
        right_len = N - left_len - gap_len
        if right_len > 0:
            h = np.zeros(N)
            h[:left_len] = h1
            h[left_len + gap_len:] = h2
            return h, best_val
    return None, 1e6

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0

    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(10):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # 6. Smooth bumps
    for center in [0.2, 0.3, 0.5, 0.7, 0.8]:
        for width in [0.05, 0.1, 0.2]:
            hb = np.exp(-((x - center) / width) ** 2)
            candidates.append(hb)

    # 7. Two-block optimized
    if not deadline_reached():
        tb_h, tb_score = _two_block_optimize()
        if tb_h is not None:
            candidates.append(tb_h)
    
    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Multi-start gradient optimization with smooth max
    # Use L-BFGS-B with the smooth surrogate
    
    num_starts = min(6, len(candidates))
    if time_left() > 30:
        # Pick the best few candidates for refinement
        scored_candidates = []
        for hc in candidates:
            hc = np.maximum(hc, 0.0)
            if np.sum(hc) <= 0:
                continue
            sc = compute_score(hc)
            scored_candidates.append((sc, hc))
        scored_candidates.sort(key=lambda x: x[0])
        
        for sc, hc in scored_candidates[:num_starts]:
            if deadline_reached():
                break
            
            # Use smooth objective with moderate beta for optimization
            beta = 50.0
            
            def obj_and_grad(h):
                h = np.maximum(h, 0.0)
                s, g = _smooth_obj_and_grad(h, beta)
                return s, g
            
            try:
                res = minimize(obj_and_grad, hc.copy(), jac=True, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6,
                                      'maxls': 20})
                new_h = np.maximum(res.x, 0.0)
                new_score = compute_score(new_h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h.copy()
            except Exception:
                pass
    
    # Final polish: Increase beta for sharper max approximation
    if time_left() > 15:
        current_h = best_h.copy()
        current_score = best_score
        
        # Progressive beta increase
        for beta in [100.0, 200.0, 500.0]:
            if deadline_reached():
                break
            
            def obj_and_grad(h):
                h = np.maximum(h, 0.0)
                s, g = _smooth_obj_and_grad(h, beta)
                return s, g
            
            try:
                res = minimize(obj_and_grad, current_h.copy(), jac=True, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxls': 30})
                new_h = np.maximum(res.x, 0.0)
                new_score = compute_score(new_h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h.copy()
                    current_h = new_h.copy()
            except Exception:
                pass
        
        # Final exact gradient descent with true max
        if time_left() > 5:
            for _ in range(500):
                if deadline_reached():
                    break
                
                # Use true max for final polish
                h_norm = current_h / np.sum(current_h)
                f = np.zeros(M_FFT, dtype=np.float64)
                f[:N] = h_norm
                F = np.fft.rfft(f)
                conv = np.fft.irfft(F * F, n=M_FFT)
                k_star = np.argmax(conv)
                
                grad_conv = np.zeros(N)
                idx1 = k_star - np.arange(N)
                valid1 = (idx1 >= 0) & (idx1 < N)
                grad_conv[valid1] += h_norm[idx1[valid1]]
                idx2 = np.arange(N) - k_star
                valid2 = (idx2 >= 0) & (idx2 < N)
                grad_conv[valid2] += h_norm[idx2[valid2]]
                
                w = 0.5 / N
                grad_norm = (N / w) * grad_conv
                total = np.sum(current_h)
                sum_grad_h = np.dot(grad_norm, current_h)
                grad = grad_norm / total - current_h / (total * total) * sum_grad_h
                
                grad_norm_val = np.linalg.norm(grad)
                if grad_norm_val < 1e-10:
                    break
                grad = grad / grad_norm_val
                
                step_size = 0.05
                improved = False
                for _ in range(10):
                    new_h = np.maximum(current_h - step_size * grad, 0.0)
                    if np.sum(new_h) <= 0:
                        step_size *= 0.5
                        continue
                    new_score = compute_score(new_h)
                    if new_score < current_score:
                        current_h = new_h
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = current_h.copy()
                        improved = True
                        break
                    step_size *= 0.5
                
                if not improved:
                    break

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
