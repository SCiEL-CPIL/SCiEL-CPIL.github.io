import numpy as np
import time
import warnings

N = 50

def _score(h):
    """Compute the ratio sum(h^2) / (sum(h) * max_conv)."""
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return l2_sq / (l1 * max_conv)


def _spectral_waterfilling(w, alpha=1.0, max_iter=2000, tol=1e-10):
    """
    Water-filling in the spectral domain to minimize the autoconvolution peak
    relative to energy.
    
    We work in the Fourier domain. Let h be the time-domain signal.
    The autoconvolution peak is bounded by the inverse Fourier transform of |H|^2.
    We want to minimize max_k |IFFT(|H|^2)| subject to ||h||_2 = 1, h >= 0.
    
    This is a non-convex problem. We use a projected gradient / water-filling approach.
    """
    n = len(w)
    # Initialize with uniform weights
    h = np.ones(n) / n
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_val = 1.0 / n  # Lower bound for uniform
    
    for it in range(max_iter):
        # Compute autoconvolution
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            break
        
        # The peak location in the autoconvolution
        peak_idx = np.argmax(conv)
        
        # Gradient of max_conv with respect to h
        # conv[k] = sum_i h[i] * h[k-i]
        # d(conv[peak_idx])/d(h[j]) = h[peak_idx - j] + h[j] (for valid indices)
        # Actually, conv = h * h, so d(conv[m])/d(h[j]) = h[m-j] for m-j in range, plus symmetry
        # More carefully: conv[m] = sum_j h[j] * h[m-j]
        # d(conv[m])/d(h[j]) = h[m-j] + (if m-j != j) h[m-j] ... no
        # d(conv[m])/d(h[j]) = sum over k where h[k]*h[m-k]: derivative is h[m-j] + h[j] if j != m-j, else 2*h[j]
        # Actually: conv[m] = sum_j h[j] h[m-j]
        # d/dh[j] conv[m] = h[m-j] + (if m-j in [0,n-1] and m-j != j) ... 
        # Let's just compute: for each j, d conv[m]/dh[j] = h[m-j] if 0<=m-j<n, plus h[j] if m-j=j i.e. m=2j
        # Wait, that's not right either. Let me think again.
        # conv[m] = sum_j h[j] * h[m-j]
        # Partial derivative wrt h[j]: 
        #   Term where first factor is h[j]: h[m-j] (if m-j in [0,n-1])
        #   Term where second factor is h[j]: h[m-j] where m-k=j, i.e., k=m-j, so h[m-(m-j)] = h[j]... no
        #   Second factor is h[k] where k = m-j, so the term is h[j]*h[m-j], derivative wrt h[j] is h[m-j]
        #   But also, when k=j, the term is h[j]*h[m-j], and derivative wrt h[j] gives h[m-j]
        #   When k=j means m-j=j, i.e., m=2j. In that case, the term is h[j]^2, derivative is 2*h[j]
        #   So: d conv[m]/dh[j] = h[m-j] (if 0<=m-j<n) + h[m-j] (if 0<=m-j<n and m-j != j) ... 
        #   No, let me be more careful.
        #   conv[m] = sum_k h[k] * h[m-k]
        #   d/dh[j] conv[m] = sum_k [delta(k,j) * h[m-k] + h[k] * delta(m-k, j)]
        #                    = h[m-j] (if 0 <= m-j < n) + h[m-j] (if 0 <= m-j < n and m-j is the k such that m-k=j, i.e., k=m-j)
        #   Wait, the second term: delta(m-k, j) means m-k = j, so k = m-j. Then h[k] = h[m-j].
        #   So d/dh[j] conv[m] = h[m-j] + h[m-j] = 2*h[m-j] if 0 <= m-j < n
        #   But if m-j = j (i.e., m = 2j), then both terms are the same term h[j]^2, and the derivative is 2*h[j], which equals 2*h[m-j]. So it's consistent.
        #   Actually wait: when k=j in the first sum, we get h[m-j]. When m-k=j in the second, k=m-j, we get h[m-j].
        #   These are the same contribution only when j = m-j, i.e., m=2j. In that case, the term in the sum is h[j]*h[j] = h[j]^2, and its derivative is 2*h[j].
        #   The formula 2*h[m-j] gives 2*h[j] when m=2j. So it's correct.
        #   But we need to be careful: if m-j is out of bounds, that term is 0.
        
        grad = np.zeros(n)
        for j in range(n):
            idx = peak_idx - j
            if 0 <= idx < n:
                grad[j] += 2.0 * h[idx]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Projected gradient step: we want to decrease max_conv while maintaining ||h||_1 = 1, h >= 0
        # Use a simple approach: move in direction -grad, then project
        lr = 0.01
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0)
        l1_new = np.sum(h_new)
        if l1_new < 1e-15:
            h_new = h
        else:
            h_new = h_new / l1_new
        
        # Check if this is better
        conv_new = np.convolve(h_new, h_new, mode='full')
        max_conv_new = np.max(conv_new)
        l2_new = np.sum(h_new ** 2)
        ratio_new = l2_new / max_conv_new
        
        conv_curr = np.convolve(h, h, mode='full')
        max_conv_curr = np.max(conv_curr)
        l2_curr = np.sum(h ** 2)
        ratio_curr = l2_curr / max_conv_curr
        
        if ratio_new > ratio_curr:
            h = h_new
            if ratio_new > best_val:
                best_val = ratio_new
                best_h = h.copy()
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
        
        if it % 100 == 0:
            pass  # No print
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Strategy: Use spectral water-filling with various initializations
    # and support sets
    
    np.random.seed(42)
    
    # 1. Try different support sizes with spectral water-filling
    for k in range(2, min(N, 15)):
        if time.time() > deadline - 15.0:
            break
        
        # Generate a random support set
        for trial in range(5):
            if time.time() > deadline - 15.0:
                break
            
            S = np.random.choice(N, k, replace=False)
            S = np.sort(S)
            
            # Initialize with uniform on support
            h_init = np.zeros(N)
            h_init[S] = 1.0 / k
            
            # Run spectral water-filling
            h_opt = _spectral_waterfilling(h_init, max_iter=1000)
            
            # Clip to support (water-filling may spread mass)
            # Actually, let's try both: restricted to support and full
            s1 = _score(h_opt)
            if s1 > best_score:
                best_score = s1
                best_h = h_opt.copy()
            
            # Also try restricting to original support
            h_restricted = np.zeros(N)
            h_restricted[S] = h_opt[S]
            s2 = _score(h_restricted)
            if s2 > best_score:
                best_score = s2
                best_h = h_restricted.copy()
    
    # 2. Try structured supports (arithmetic progressions, etc.)
    if time.time() < deadline - 20.0:
        for start in range(0, N, 5):
            for step in range(1, 8):
                if time.time() > deadline - 15.0:
                    break
                
                S = list(range(start, N, step))
                if len(S) < 2:
                    continue
                
                h_init = np.zeros(N)
                h_init[S] = 1.0 / len(S)
                
                h_opt = _spectral_waterfilling(h_init, max_iter=800)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
    
    # 3. Polish the best result with more iterations
    if best_score > 0 and time.time() < deadline - 5.0:
        h_polished = _spectral_waterfilling(best_h, max_iter=2000)
        s = _score(h_polished)
        if s > best_score:
            best_score = s
            best_h = h_polished.copy()
    
    # 4. Final local optimization on the best support
    if time.time() < deadline - 3.0:
        support_idx = np.where(best_h > 1e-8)[0]
        if len(support_idx) >= 2:
            from scipy.optimize import minimize
            
            def neg_ratio(x):
                h_full = np.zeros(N)
                h_full[support_idx] = x
                s = _score(h_full)
                if s < 1e-12:
                    return 1e10
                return -np.log(s)
            
            x0 = best_h[support_idx]
            x0 = x0 / np.sum(x0)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            ]
            bounds = [(0, 1)] * len(support_idx)
            
            try:
                res = minimize(neg_ratio, x0, method='SLSQP', bounds=bounds, 
                              constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
                if res.success:
                    h_opt = np.zeros(N)
                    h_opt[support_idx] = res.x
                    s = _score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    return list(best_h)
