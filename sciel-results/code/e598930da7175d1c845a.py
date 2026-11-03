"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs(h):
    """Compute all linear (non-circular) cross-correlations R_k for k=1..n-1.
    R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    Uses FFT for efficiency: pad to length 2n, compute autocorrelation via FFT.
    """
    n = len(h)
    # Pad to length 2n to avoid circular wrap-around
    h_pad = np.zeros(2 * n)
    h_pad[:n] = h
    H = np.fft.fft(h_pad)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_{i=0}^{2n-k-1} h_pad[i] * h_pad[i+k]
    # For k=1..n-1, this equals sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # (since h_pad[i]=0 for i>=n, the extra terms vanish)
    corrs = (2.0 / n) * autocorr[1:n]
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of R(s) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1."""
    corrs = _compute_all_corrs(h)
    return np.min(corrs)

def _project_box_sum(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum} using iterative bisection on threshold."""
    n = len(v)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, 0.0, 1.0)
        s = np.sum(w)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - (lo + hi) / 2.0, 0.0, 1.0)
    # Fine-tune to hit exact sum
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-14:
        mask = (w > 0) & (w < 1)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0.0, 1.0)
    return w

def _soft_min_objective_and_grad(h, tau=0.05):
    """Compute smooth approximation of min_k R_k and its gradient.
    
    Uses log-sum-exp: soft_min(R) = -tau * log(sum(exp(-R_k/tau)))
    Gradient: d(soft_min)/dh = -sum_k w_k * dR_k/dh where w_k = exp(-R_k/tau)/sum(exp(-R_j/tau))
    
    For R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]:
    dR_k/dh[j] = (2/n)*h[j+k] (if j < n-k) + (2/n)*h[j-k] (if j >= k)
    
    We compute the weighted sum of gradients using FFT-based approach.
    """
    n = len(h)
    corrs = _compute_all_corrs(h)
    
    # Soft min via log-sum-exp
    neg_R_over_tau = -corrs / tau
    max_val = np.max(neg_R_over_tau)
    exp_vals = np.exp(neg_R_over_tau - max_val)
    log_sum_exp = max_val + np.log(np.sum(exp_vals))
    soft_min_val = -tau * log_sum_exp
    
    # Weights w_k = exp(-R_k/tau) / sum(exp(-R_j/tau))
    w = exp_vals / np.sum(exp_vals)
    
    # Compute gradient: sum_k w_k * dR_k/dh
    # dR_k/dh[j] = (2/n)*h[j+k] for j < n-k, plus (2/n)*h[j-k] for j >= k
    # grad[j] = sum_k w_k * [(2/n)*h[j+k] (if j<n-k) + (2/n)*h[j-k] (if j>=k)]
    # 
    # Let's compute this efficiently:
    # grad[j] = (2/n) * sum_{k: j<n-k} w_k * h[j+k] + (2/n) * sum_{k: j>=k} w_k * h[j-k]
    #          = (2/n) * sum_{k=1}^{n-1-j} w_k * h[j+k] + (2/n) * sum_{k=1}^{j} w_k * h[j-k]
    
    grad = np.zeros(n)
    for k in range(1, n):
        wk = w[k-1]
        if wk < 1e-10:
            continue
        factor = (2.0 / n) * wk
        # Contribution from first term: j < n-k, grad[j] += factor * h[j+k]
        if n - k > 0:
            grad[:n-k] += factor * h[k:]
        # Contribution from second term: j >= k, grad[j] += factor * h[j-k]
        if k < n:
            grad[k:] += factor * h[:n-k]
    
    return soft_min_val, grad

def _fista_smooth(h_init, target_sum, max_iters=3000, tau_schedule=None):
    """FISTA for minimizing soft_min(R) subject to 0<=h<=1, sum=h_target.
    
    Uses smooth log-sum-exp approximation of the min correlation.
    """
    n = len(h_init)
    h = h_init.copy()
    h_prev = h_init.copy()
    t_curr = 1.0
    
    # Lipschitz constant estimate for the smooth objective
    # The gradient of soft_min has norm bounded by sum_k w_k * ||dR_k/dh||
    # ||dR_k/dh||_2 <= (2/n) * sqrt(2*(n-k)) approximately
    # Conservative: mu = (2/n) * sqrt(2*n) * 2
    mu = (2.0 / n) * 2.0 * np.sqrt(2.0 * n)
    mu = max(mu, 1.0)
    
    best_h = h.copy()
    best_obj = _compute_cross_correlation_min(h)
    
    # Adaptive tau: start large, decrease over time
    if tau_schedule is None:
        tau_schedule = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    tau_idx = 0
    tau = tau_schedule[0]
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
        
        # Adapt tau: decrease every ~max_iters/len(tau_schedule) iterations
        if tau_idx < len(tau_schedule) - 1:
            switch_iter = int(max_iters / len(tau_schedule))
            if (it + 1) % switch_iter == 0:
                tau_idx += 1
                tau = tau_schedule[tau_idx]
        
        # FISTA momentum
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_curr * t_curr)) / 2.0
        y = h + ((t_curr - 1.0) / t_next) * (h - h_prev)
        y = np.clip(y, 0.0, 1.0)  # keep in box for gradient computation
        
        # Compute smooth objective and gradient at y
        _, grad_y = _soft_min_objective_and_grad(y, tau)
        
        # Gradient step
        z = y - (1.0 / mu) * grad_y
        
        # Proximal step: project onto {0<=x<=1, sum=target_sum}
        h_new = _project_box_sum(z, target_sum)
        
        # Update
        h_prev = h
        h = h_new
        t_curr = t_next
        
        # Check improvement using true min
        obj = _compute_cross_correlation_min(h)
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
    
    return best_h, best_obj

def _local_polish(h_init, target_sum, max_iters=500):
    """Local polishing using subgradient descent with adaptive step size on the hard min."""
    n = len(h_init)
    h = h_init.copy()
    best_obj = _compute_cross_correlation_min(h)
    best_h = h.copy()
    step = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
        
        # Find argmin correlation
        corrs = _compute_all_corrs(h)
        k_star = np.argmin(corrs) + 1  # k_star is 1-indexed
        
        # Subgradient direction to increase R_{k_star}
        grad = np.zeros(n)
        ks = k_star
        grad[:n-ks] += (2.0 / n) * h[ks:]
        grad[ks:] += (2.0 / n) * h[:n-ks]
        
        # Move in +grad direction (to increase min correlation)
        h_new = h + step * grad
        h_new = _project_box_sum(h_new, target_sum)
        
        new_obj = _compute_cross_correlation_min(h_new)
        if new_obj < best_obj - 1e-10:
            best_obj = new_obj
            best_h = h_new.copy()
            h = h_new
            no_improve = 0
            step = min(step * 1.05, 0.05)
        else:
            h = h_new  # still update to keep momentum
            no_improve += 1
            step *= 0.85
            if step < 1e-5:
                break
    
    return best_h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = _compute_cross_correlation_min(best_x)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(n, 0.5))
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = max(1, int(n * frac))
        if k < n:
            val = target_sum / (n - k)
            if 0 <= val <= 1:
                restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, val)]))
                restarts.append(np.concatenate([np.full(k, val), np.full(n-k, 0.0)]))
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    for period in [3, 4, 5, 6, 7]:
        for offset in range(period):
            pattern = np.array([1.0 if (i + offset) % period < period // 2 else 0.0 for i in range(n)])
            mean_val = np.mean(pattern)
            if 0.01 < mean_val < 0.99:
                pattern = pattern / (2.0 * mean_val)
                if np.all((pattern >= 0) & (pattern <= 1)):
                    restarts.append(pattern)
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    # Sinusoidal
    x = np.linspace(0, 2 * np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for amp in [0.2, 0.3, 0.4]:
            restarts.append(np.clip(0.5 + amp * np.sin(freq * x), 0, 1))
            restarts.append(np.clip(0.5 + amp * np.cos(freq * x), 0, 1))
    # Multi-frequency
    for _ in range(5):
        freqs = np.random.choice([1,2,3,4,5,6,7,8], size=3, replace=False)
        amps = np.random.uniform(0.1, 0.3, size=3)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        sig = np.zeros(n)
        for f, a, p in zip(freqs, amps, phases):
            sig += a * np.sin(f * x + p)
        restarts.append(np.clip(0.5 + sig, 0, 1))
    
    # Run smooth FISTA on each restart
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x0_proj = _project_box_sum(x0, target_sum)
        try:
            h_opt, obj = _fista_smooth(x0_proj, target_sum, max_iters=2000)
            if obj < best_obj:
                best_obj = obj
                best_x = h_opt
        except Exception:
            pass
    
    # Polish the best solution
    if _time_remaining() > 8.0:
        h_pol, obj_pol = _local_polish(best_x, target_sum, max_iters=500)
        if obj_pol < best_obj:
            best_obj = obj_pol
            best_x = h_pol
    
    # Final smooth FISTA polish with more iterations and finer tau
    if _time_remaining() > 8.0:
        h_opt2, obj2 = _fista_smooth(best_x, target_sum, max_iters=5000, 
                                      tau_schedule=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001])
        if obj2 < best_obj:
            best_obj = obj2
            best_x = h_opt2
    
    # Ensure exact sum and bounds
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        cnt = np.sum(mask)
        if cnt > 0:
            best_x[mask] += diff / cnt
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
