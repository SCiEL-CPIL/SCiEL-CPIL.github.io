import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    n_fft = 2 * N - 1
    F = np.fft.rfft(f, n=n_fft)
    ac = np.fft.irfft(F * F, n=n_fft)
    return ac

def _soft_max_abs_ac(f, beta=50.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient) using FFT-based gradient computation.
    """
    n_fft = 2 * N - 1
    F = np.fft.rfft(f, n=n_fft)
    ac = np.fft.irfft(F * F, n=n_fft)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    
    shifted = beta * (abs_ac - max_ac)
    log_sum = np.log(np.sum(np.exp(shifted))) / beta + max_ac
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    # Gradient via FFT: d|ac[k]|/df[i] = sign(ac[k]) * 2 * f[k-i] (valid range)
    # grad[i] = 2 * sum_k sign(ac[k]) * w[k] * f[k-i] for valid k-i
    # This is a cross-correlation of (sign(ac) * w) with f
    sgn_w = np.sign(ac) * weights  # length n_fft
    # For each i, grad[i] = 2 * sum_{k: 0<=k-i<N} sgn_w[k] * f[k-i]
    # = 2 * (sgn_w * f)[i] where * is cross-correlation
    # Cross-correlation: (a * b)[i] = sum_k a[k+i] * b[k]
    # We want: sum_k sgn_w[k] * f[k-i] = sum_k sgn_w[k+i] * f[i] ... no
    # Let's use: sum_k sgn_w[k] * f[k-i] for 0<=k-i<N, i.e., k from i to i+N-1
    # = (sgn_w ⊛ f)[i] where ⊛ denotes cross-correlation
    # np.correlate(sgn_w, f, mode='full')[N-1+i] gives sum_k sgn_w[k] * f[k-i]
    # Actually: np.correlate(a, b, 'full')[m] = sum_k a[k+m] * b[k]
    # We want sum_k sgn_w[k] * f[k-i] = sum_j sgn_w[j] * f[j-i] (j=k)
    # = np.correlate(sgn_w, f, 'full')[i] ... let me verify
    # np.correlate(a, b, 'full')[n] = sum_k a[k+n] * b[k] for valid k
    # So np.correlate(sgn_w, f, 'full')[i] = sum_k sgn_w[k+i] * f[k]
    # That's not what we want. We want sum_k sgn_w[k] * f[k-i]
    # = sum_j sgn_w[j] * f[j-i] where j=k
    # Let m = j-i, so j = m+i: sum_m sgn_w[m+i] * f[m]
    # = np.correlate(sgn_w, f, 'full')[i]
    # Wait, that IS what we want! sum_m sgn_w[m+i] * f[m]
    # But we need 0 <= m < N (since f is length N) and 0 <= m+i < n_fft (since sgn_w is length n_fft)
    # For i in [0, N-1], m in [0, N-1], m+i in [i, i+N-1] which is in [0, 2N-2] = [0, n_fft-1]. Good.
    
    corr = np.correlate(sgn_w, f, mode='full')
    # corr[i] = sum_k sgn_w[k+i] * f[k] for k=0..N-1 (when i is in valid range)
    # We want grad[i] = 2 * sum_{k=i}^{i+N-1} sgn_w[k] * f[k-i]
    # Let j = k-i: sum_{j=0}^{N-1} sgn_w[j+i] * f[j] = corr[i]
    grad = 2.0 * corr[:N]
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if abs(s) < 1e-30 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(abs(s))
    
    return obj, g1 + g2

def _sign_flip_search(f, deadline, max_iters=200000):
    """
    Local search using sign flips. Exploits scale invariance:
    flipping signs changes the autoconvolution pattern while the
    objective ratio max|ac|/(sum f)^2 can decrease if sidelobes cancel.
    """
    best = f.copy()
    ac_best = _autoconv(best)
    s_best = np.sum(best)
    if abs(s_best) < 1e-12:
        return best, np.inf
    best_score = np.max(np.abs(ac_best)) / (s_best * s_best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try flipping 1-3 random signs
        n_flips = np.random.randint(1, 4)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        s_cand = np.sum(candidate)
        if abs(s_cand) < 1e-12:
            continue
        
        ac_cand = _autoconv(candidate)
        score = np.max(np.abs(ac_cand)) / (s_cand * s_cand)
        
        if score < best_score:
            best = candidate
            ac_best = ac_cand
            s_best = s_cand
            best_score = score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_f = np.ones(N)
    best_score = 1.0 / N  # Theoretical lower bound achieved by ones
    
    np.random.seed(42)
    
    # Phase 1: Gradient-based optimization from diverse starts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.random.randn(N)
        elif r == 2:
            f0 = np.random.choice([-1.0, 1.0], size=N)
            if np.sum(f0) < 0:
                f0 = -f0
        elif r == 3:
            t = np.linspace(0, 10, N)
            f0 = np.cos(t**2)
        elif r == 4:
            f0 = np.ones(N)
            f0[::2] = -1.0
        elif r == 5:
            f0 = np.random.randn(N // 2)
            f0 = np.repeat(f0, 2)
        elif r == 6:
            t = np.linspace(0, 4 * np.pi, N)
            f0 = np.sin(t)
        elif r == 7:
            # Block pattern
            f0 = np.ones(N)
            block = N // 4
            f0[block:2*block] = -1.0
        elif r == 8:
            # Low-frequency modulation
            t = np.linspace(0, 2*np.pi, N)
            f0 = 1.0 + 0.5 * np.cos(t)
        else:
            f0 = np.random.randn(N)
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            candidate = res.x
            ac = _autoconv(candidate)
            s = np.sum(candidate)
            if abs(s) > 1e-12:
                score = np.max(np.abs(ac)) / (s * s)
                if score < best_score:
                    best_f = candidate
                    best_score = score
                    
        except Exception:
            pass
    
    # Phase 2: Sign-flip local search from best configuration
    if time.time() < deadline - 5.0:
        polished, polished_score = _sign_flip_search(best_f, deadline - 5.0, max_iters=300000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    return best_f
