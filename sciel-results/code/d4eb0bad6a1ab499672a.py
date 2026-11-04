import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _objective(x):
    """
    Minimize log(||h||_1) + log(||h*h||_inf) subject to ||h||_2 = 1.
    Scale invariant: fix ||h||_2 = 1, then score = 1/(||h||_1 * ||h*h||_inf).
    """
    h = np.maximum(x, 0)
    norm = np.linalg.norm(h)
    if norm < 1e-15:
        return 1e10
    h = h / norm  # normalize to L2=1
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 1e10
        
    return np.log(l1) + np.log(max_conv)

def _objective_grad(x):
    """
    Gradient of log(||h||_1) + log(||h*h||_inf) with h normalized to L2=1.
    Uses subgradient for the max term.
    """
    h = np.maximum(x, 0)
    norm = np.linalg.norm(h)
    if norm < 1e-15:
        return np.zeros(N)
    h = h / norm
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return np.zeros(N)
    
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return np.zeros(N)
    
    # Gradient of log(l1) w.r.t. h (before normalization)
    # After normalization h/||h||, the gradient transforms
    # Let's compute gradient w.r.t. unnormalized x, then project
    
    # Actually, let's think more carefully.
    # f(x) = log(sum(max(x,0)/||x||)) + log(max(autoconv(max(x,0)/||x||)))
    # This is complex. Let's use a simpler approach.
    
    # With h = x/||x|| (assuming x >= 0), 
    # d/dx_i [sum h] = d/dx_i [sum x_j/||x||] = (delta_i - x_i/||x||^2) * ||x||... 
    # This gets messy. Let's use finite differences as fallback, or a simpler analytic form.
    
    # Alternative: since we fix ||h||_2=1, let's parametrize differently.
    # Or just use scipy's numerical gradient.
    
    # For now, use a simple subgradient approach:
    # The peak of autoconv is at some index k*. 
    # conv_k = sum_j h_j h_{j+k} (with appropriate bounds for full conv)
    
    # Let's find which index achieves the max
    k_star = np.argmax(conv)
    
    # Gradient of conv_k_star w.r.t. h_m:
    # conv_k_star = sum_j h_j h_{j+k_star} for j where both indices valid
    # d(conv_k)/dh_m = h_{m+k_star} (if m+k_star < N) + h_{m-k_star} (if m-k_star >= 0)
    # But we need to be careful with the full convolution indexing.
    
    # For full conv of length 2N-1:
    # conv[t] = sum_j h[j] * h[j + t - (N-1)] ... let me re-derive.
    # Actually, np.convolve(h, h, 'full') gives:
    # conv[t] = sum_j h[j] * h[t-j] for j where 0<=j<N and 0<=t-j<N
    
    # So d(conv[t])/dh[m] = h[t-m] (if 0<=t-m<N) + h[m-t] (if 0<=m-t<N)
    # Wait, that's not right either. Let me be more careful.
    
    # conv[t] = sum_j h[j] h[t-j]
    # If we vary h[m], it appears when j=m: term is h[t-m]
    # and when t-j=m, i.e., j=t-m: term is h[t-(t-m)] = h[m]... no.
    
    # conv[t] = sum_j h[j] * h[t-j]
    # d(conv[t])/dh[m] = (if j=m is in range) h[t-m] + (if t-j=m, i.e., j=t-m, is in range) h[t-(t-m)]
    # = h[t-m] (if 0<=m<N and 0<=t-m<N) + h[m] (if 0<=t-m<N and 0<=m<N)
    # Wait, the second term: when j = t-m, the term is h[j]*h[t-j] = h[t-m]*h[m]. 
    # So d/dh[m] of this is h[t-m].
    # And the first term: when j=m, the term is h[m]*h[t-m], so d/dh[m] is h[t-m].
    # So d(conv[t])/dh[m] = 2*h[t-m] if 0<=t-m<N, else 0? 
    # No wait, that double counts. Let me redo.
    
    # conv[t] = sum_j h[j] * h[t-j]
    # The term involving h[m] appears in two ways:
    # 1. When j=m: h[m]*h[t-m], contributes h[t-m] to d(conv[t])/dh[m]
    # 2. When t-j=m, i.e., j=t-m: h[t-m]*h[m], contributes h[t-m] to d(conv[t])/dh[m]
    # But these are the same term if we're summing over all j. Actually no, they're different j values unless m = t-m.
    # If m != t-m, both terms appear separately. If m = t-m, it's one term.
    
    # So d(conv[t])/dh[m] = h[t-m] + h[t-m] = 2*h[t-m] if 0<=t-m<N and t-m != m
    # And = h[t-m] if t-m == m (i.e., t=2m)
    # Hmm, this is getting complicated. Let me just use the fact that for the full convolution,
    # the derivative can be computed as a convolution.
    
    # Simpler: use numerical gradient since N=50 is small.
    # But that's expensive per iteration. Let me try a different approach.
    
    # Let's just return zeros and rely on scipy's numerical gradient.
    # Actually, let me try a cleaner approach.
    
    # Return subgradient based on which bin achieves max
    grad = np.zeros(N)
    
    # For the L1 part: d/dh[m] log(sum h) = 1/sum(h) (for h>0)
    grad[:] = 1.0 / l1
    
    # For the max_conv part: subgradient is concentrated at the peak index
    # d/dh[m] log(conv[k_star]) = (1/conv[k_star]) * d(conv[k_star])/dh[m]
    
    # Compute d(conv[k_star])/dh[m] for all m
    t = k_star
    for m in range(N):
        # h[t-m] contributes if 0 <= t-m < N
        if 0 <= t - m < N:
            # From j=m: h[t-m]
            grad[m] += h[t - m] / max_conv
            # From j=t-m: also h[t-m] if t-m != m
            if t - m != m:
                grad[m] += h[t - m] / max_conv
    
    return grad

def _objective_and_grad(x):
    f = _objective(x)
    g = _objective_grad(x)
    return f, g

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Generate diverse starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N))
    
    # 2. Centered Gaussian
    x = np.arange(N)
    starts.append(np.exp(-((x - N//2)**2) / (2 * (N/6)**2)))
    
    # 3. Sparse random
    rng = np.random.RandomState(42)
    for _ in range(8):
        h = np.zeros(N)
        idx = rng.choice(N, rng.randint(3, 15), replace=False)
        h[idx] = rng.uniform(0.5, 1.0, len(idx))
        starts.append(h)
        
    # 4. Dense random
    for _ in range(8):
        h = rng.uniform(0.1, 1.0, N)
        starts.append(h)
        
    # 5. Bimodal
    for _ in range(4):
        h = np.zeros(N)
        center1 = N // 4
        center2 = 3 * N // 4
        h[center1] = 1.0
        h[center2] = 1.0
        h += rng.uniform(0, 0.3, N)
        starts.append(h)
        
    # 6. Triangular
    for sigma in [3, 5, 8, 12]:
        h = np.exp(-np.abs(x - N//2) / sigma)
        starts.append(h)
        
    # 7. Sinc-like
    for a in [0.5, 1.0, 2.0]:
        h = np.abs(np.sinc(a * (x - N//2) / N))
        starts.append(h)
        
    # 8. Power law
    for p in [0.5, 1.0, 2.0]:
        h = (x + 1) ** (-p)
        h = h / np.max(h)
        starts.append(h)
        starts.append(h[::-1])
        
    # 9. Step functions
    for k in [5, 10, 15, 20, 25]:
        h = np.zeros(N)
        h[:k] = 1.0
        starts.append(h)
        h2 = np.zeros(N)
        h2[k:] = 1.0
        starts.append(h2)
        
    # 10. Cosine windows
    for beta in [0.5, 1.0, 2.0, 3.0]:
        h = np.cos(np.pi * (x - N//2) / (N/2 * beta)) ** 2
        h[h < 0] = 0
        if np.max(h) > 0:
            h = h / np.max(h)
            starts.append(h)
        
    # Shuffle starts for diversity
    rng.shuffle(starts)
    
    for s in starts:
        if time.time() > deadline - 15:
            break
            
        s = np.asarray(s, dtype=float)
        norm = np.linalg.norm(s)
        if norm < 1e-15:
            continue
        x0 = s / norm
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        h_opt = np.maximum(res.x, 0)
        h_norm = np.linalg.norm(h_opt)
        if h_norm < 1e-15:
            continue
        h_opt = h_opt / h_norm
        
        score = _score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Local refinement from this result
        if time.time() > deadline - 10:
            break
        try:
            res2 = minimize(
                _objective, 
                h_opt, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt2 = np.maximum(res2.x, 0)
            h_norm2 = np.linalg.norm(h_opt2)
            if h_norm2 > 1e-15:
                h_opt2 = h_opt2 / h_norm2
                score2 = _score(h_opt2)
                if score2 > best_score:
                    best_score = score2
                    best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Final polish with best found
    if best_score > 0 and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_final = np.maximum(res.x, 0)
            h_norm = np.linalg.norm(h_final)
            if h_norm > 1e-15:
                h_final = h_final / h_norm
                score_final = _score(h_final)
                if score_final > best_score:
                    best_h = h_final
        except Exception:
            pass
    
    # Ensure non-negative and valid
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
