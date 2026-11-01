"""Rewritten program: continuous relaxation + FFT-based discrete refinement for C3 upper bound."""
import time
import numpy as np

N = 400
DEADLINE_SECONDS = 115  # safety margin before 120s kill

def _autoconv_peak(s):
    """Compute the peak absolute autoconvolution of signed step function s."""
    # Autoconvolution via FFT
    f = np.fft.fft(s, n=2 * N)
    ac = np.fft.ifft(f * np.conj(f)).real
    # The autoconvolution has 2N-1 meaningful points; center is at index N-1
    # We want the maximum absolute value away from the peak (which is at lag 0)
    # Actually for C3 inequality, we care about the upper bound which involves
    # the peak sidelobe of the autoconvolution
    # The DC component (lag 0) is sum(s^2) = N for signed steps
    # We look at the maximum of |ac| excluding the main lobe
    # The autoconvolution array from ifft has length 2N; lags 0..N-1 are in first half
    # Let's get the circular autoconvolution properly
    # Actually, linear autoconvolution:
    ac_lin = np.convolve(s, s[::-1], mode='full')
    # The peak is at the center
    center = N - 1
    # The C3 upper bound relates to the ratio of peak sidelobe to main lobe
    # Main lobe value = sum(s_i^2) = N (since s_i = ±1)
    # Peak sidelobe = max of |ac_lin| away from center
    # For signed steps, ac_lin[center] = N
    # We want to minimize the peak sidelobe relative to N
    # The score is the upper bound on C3, which is related to peak_sidelobe / N
    # Lower is better, so we want to minimize peak_sidelobe
    sidelobe = np.delete(ac_lin, center)
    peak_sidelobe = np.max(np.abs(sidelobe))
    return peak_sidelobe

def _score_from_sidelobe(peak_sidelobe):
    """Convert peak sidelobe to C3 upper bound score."""
    # The C3 upper bound is peak_sidelobe / N (normalized)
    # Lower is better
    return peak_sidelobe / N

def _gradient_descent_continuous(s0, max_iter=5000, lr=0.01, deadline=None):
    """
    Continuous relaxation: treat s as real-valued, optimize to minimize peak autoconvolution.
    Then round to ±1.
    """
    s = s0.astype(np.float64).copy()
    best_s = s.copy()
    best_score = _autoconv_peak(s)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        ac = np.convolve(s, s[::-1], mode='full')
        center = N - 1
        main = ac[center]
        
        # Gradient of peak sidelobe w.r.t. s
        # This is tricky because of the max operation. Use a soft-max approximation
        # or just use the gradient at the argmax
        sidelobe = np.delete(ac, center)
        abs_sidelobe = np.abs(sidelobe)
        idx_max = np.argmax(abs_sidelobe)
        
        # Map idx_max back to the original ac indices
        if idx_max < center:
            ac_idx = idx_max
        else:
            ac_idx = idx_max + 1
        
        # Gradient of ac[ac_idx] w.r.t. s
        # ac[k] = sum_i s[i] * s[k-i] (with appropriate indexing for full convolve)
        # For full convolve of s with s[::-1]:
        # ac[j] = sum_i s[i] * s_rev[j-i] = sum_i s[i] * s[N-1-(j-i)]
        # = sum_i s[i] * s[N-1-j+i]
        
        # d(ac[ac_idx])/d(s[m]) = s[N-1-ac_idx+m] + s[ac_idx-m] (if indices valid)
        # This is getting complex. Let's use numerical gradient or a simpler approach.
        
        # Alternative: use FFT-based gradient
        # f = FFT(s), ac = IFFT(|f|^2)
        # d(ac[k])/d(s[m]) = 2*Re(IFFT(f * conj(f) * exp(2*pi*i*k*m/N)))... 
        # This is also complex.
        
        # Simpler: just do a small perturbation-based gradient estimate
        # Or use the fact that for the peak, we can use subgradient
        # Let's try a different approach: optimize a smooth proxy
        
        # Use a soft-max of the absolute sidelobes as a smooth objective
        beta = 50.0
        sm = np.logsumexp(beta * abs_sidelobe) / beta
        # Gradient of sm w.r.t. ac values
        weights = np.exp(beta * abs_sidelobe) / np.sum(np.exp(beta * abs_sidelobe))
        
        # Gradient of ac[ac_idx] w.r.t. s
        # For full convolution ac = convolve(s, s_rev):
        # ac[j] = sum_{i=0}^{N-1} s[i] * s[N-1-j+i] where N-1-j+i is in [0, N-1]
        # d(ac[j])/d(s[m]) = s[N-1-j+m] (if 0<=N-1-j+m<N) + s[j-m] (if 0<=j-m<N)
        
        # Let's compute gradient numerically for simplicity and correctness
        eps = 1e-6
        grad = np.zeros(N)
        for m in range(N):
            s_plus = s.copy()
            s_plus[m] += eps
            ac_plus = np.convolve(s_plus, s_plus[::-1], mode='full')
            sm_plus = np.logsumexp(beta * np.abs(np.delete(ac_plus, center))) / beta
            grad[m] = (sm_plus - sm) / eps
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Update
        s -= lr * grad
        # Project back to roughly unit magnitude (soft projection)
        s = np.sign(s) * np.maximum(0.1, np.minimum(1.0, np.abs(s)))
        
        score = _autoconv_peak(s)
        if score < best_score:
            best_score = score
            best_s = s.copy()
    
    return best_s, best_score

def _discrete_refinement(s, max_iters=20000, deadline=None):
    """
    Discrete local search: flip individual bits to improve the peak sidelobe.
    """
    s = np.sign(s).copy()
    s[s == 0] = 1  # ensure ±1
    best_score = _autoconv_peak(s)
    best_s = s.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping each bit and keep the best improvement
        best_improvement = None
        best_flip_idx = -1
        best_new_score = best_score
        
        # Compute current autoconvolution
        ac = np.convolve(s, s[::-1], mode='full')
        center = N - 1
        sidelobe = np.delete(ac, center)
        peak_sidelobe = np.max(np.abs(sidelobe))
        
        # For efficiency, try a subset of flips per iteration
        # Random sample of indices to try
        n_tries = min(50, N)
        indices = np.random.choice(N, size=n_tries, replace=False)
        
        for m in indices:
            s_new = s.copy()
            s_new[m] = -s_new[m]
            ac_new = np.convolve(s_new, s_new[::-1], mode='full')
            sidelobe_new = np.delete(ac_new, center)
            peak_new = np.max(np.abs(sidelobe_new))
            
            if peak_new < best_new_score:
                best_new_score = peak_new
                best_flip_idx = m
                best_improvement = s_new
        
        if best_flip_idx >= 0:
            s = best_improvement
            best_score = best_new_score
            best_s = s.copy()
        else:
            # No improvement found in this round; try with different random indices
            # or break if we've been stuck for a while
            pass
    
    return best_s, best_score

def construct():
    """Construct a signed step function with 400 intervals to minimize C3 upper bound."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with random initialization
    np.random.seed(42)
    s0 = np.random.choice([-1.0, 1.0], size=N)
    
    best_s = s0.copy()
    best_score = _autoconv_peak(s0)
    
    # Phase 1: Continuous relaxation from multiple random starts
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            s_init = s0.copy()
        else:
            s_init = np.random.choice([-1.0, 1.0], size=N)
        
        s_cont, score_cont = _gradient_descent_continuous(s_init, max_iter=3000, deadline=deadline - 30)
        if score_cont < best_score:
            best_score = score_cont
            best_s = s_cont.copy()
    
    # Phase 2: Discrete refinement
    s_rounded = np.sign(best_s)
    s_rounded[s_rounded == 0] = 1
    s_refined, score_refined = _discrete_refinement(s_rounded, max_iters=50000, deadline=deadline - 10)
    if score_refined < best_score:
        best_score = score_refined
        best_s = s_refined.copy()
    
    # Ensure final output is ±1
    result = np.sign(best_s)
    result[result == 0] = 1
    
    # Final check
    final_score = _autoconv_peak(result)
    if final_score < best_score:
        best_score = final_score
    
    return result
