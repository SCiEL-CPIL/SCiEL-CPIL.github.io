import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

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
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    grad = np.zeros(len(heights), dtype=np.float64)
    for i in range(len(heights)):
        j = peak_idx - i
        if 0 <= j < len(heights):
            grad[i] = 2.0 * heights[j]
    
    return conv, grad, peak_idx

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # Use log-sum-exp smooth surrogate for the max
    def smooth_max(conv, alpha):
        """Smooth approximation of max(conv) using log-sum-exp."""
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        return m + np.log(np.sum(exp_conv)) / alpha
    
    def smooth_obj_and_grad(heights, alpha):
        """Compute smooth objective and its gradient."""
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Gradient of smooth_max
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        weights = exp_conv / np.sum(exp_conv)  # softmax weights
        
        # Compute gradient of smooth_max w.r.t. heights using FFT correlation
        W_full = np.fft.rfft(weights)
        F_full = np.fft.rfft(f)
        corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
        grad_sm = 2.0 * corr[:N]
        
        # Chain rule for normalized objective
        S = np.sum(heights)
        w = 0.5 / N
        if S <= 1e-10:
            return 1e10, np.zeros(N)
        
        sm = smooth_max(conv, alpha)
        obj = sm / (S * w) ** 2 * (N * w)
        
        # d(obj)/d(f_i) = grad_sm[i] * N/(S^2*w) - sm * 2*N/(S^3*w)
        obj_grad = grad_sm * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
        
        return obj, obj_grad
    
    # Try several structured initial candidates
    x = np.linspace(0, 1, N)
    candidates = [np.ones(N)]
    
    # Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.4]:
        hc = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            hc[:tl] = ramp
            hc[-tl:] = ramp[::-1]
        candidates.append(hc)
    
    # Gaussian bumps
    for width in [0.3, 0.5, 0.7]:
        candidates.append(np.exp(-((x - 0.5) / width) ** 2))
    
    # Parabolic
    for a in [1.0, 2.0, 3.0]:
        hp = np.maximum(1.0 - a * (2 * x - 1) ** 2, 0.0)
        if np.max(hp) > 0:
            candidates.append(hp)
    
    # Evaluate candidates
    for hc in candidates:
        if time_left() < 5:
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()
    
    # Use scipy.optimize with L-BFGS-B for gradient-based optimization
    # We'll do this from multiple starting points
    
    def objective_wrapper(h_flat):
        if time_left() < 5:
            return 1e10
        h = np.maximum(h_flat, 0.0)
        obj, _ = smooth_obj_and_grad(h, alpha=100.0)
        return obj
    
    def gradient_wrapper(h_flat):
        if time_left() < 5:
            return np.zeros(N)
        h = np.maximum(h_flat, 0.0)
        _, grad = smooth_obj_and_grad(h, alpha=100.0)
        return grad
    
    # Optimize from best candidate and a few others
    starts = [best_h.copy()]
    for hc in candidates[:5]:
        starts.append(np.maximum(hc, 0.0).copy())
    
    for h_start in starts:
        if time_left() < 15:
            break
        
        try:
            res = minimize(
                objective_wrapper,
                h_start,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success and res.fun < best_score:
                h_res = np.maximum(res.x, 0.0)
                actual_score = compute_score(h_res)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time_left() < 15:
            break
        
        h_rand = np.ones(N) * rng.uniform(0.5, 1.5)
        # Add some smooth noise
        for k in rng.integers(1, 10, size=3):
            h_rand += rng.uniform(0.1, 0.3) * np.cos(2 * np.pi * k * x + rng.uniform(0, 2 * np.pi))
        h_rand = np.maximum(h_rand, 0.0)
        
        try:
            res = minimize(
                objective_wrapper,
                h_rand,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success and res.fun < best_score:
                h_res = np.maximum(res.x, 0.0)
                actual_score = compute_score(h_res)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Final polish with smaller alpha for sharper max approximation
    if time_left() > 10:
        def objective_polish(h_flat):
            if time_left() < 3:
                return 1e10
            h = np.maximum(h_flat, 0.0)
            obj, _ = smooth_obj_and_grad(h, alpha=300.0)
            return obj
        
        def gradient_polish(h_flat):
            if time_left() < 3:
                return np.zeros(N)
            h = np.maximum(h_flat, 0.0)
            _, grad = smooth_obj_and_grad(h, alpha=300.0)
            return grad
        
        try:
            res = minimize(
                objective_polish,
                best_h.copy(),
                jac=gradient_polish,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                h_res = np.maximum(res.x, 0.0)
                actual_score = compute_score(h_res)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_res.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
