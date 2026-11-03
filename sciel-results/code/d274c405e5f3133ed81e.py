"""Combined Erdos minimum overlap: FISTA + FFT + piecewise-constant parameterization."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1 using FFT."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    corr_full = np.fft.irfft(f * np.conj(f), size)
    corrs = corr_full[1:n] * (2.0 / n)
    return corrs

def _compute_min_corr(h):
    """Compute min over shifts of cross-correlation."""
    corrs = _compute_all_corrs(h)
    return np.min(corrs)

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0 - 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-15:
        mask = (w > 1e-15) & (w < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            w[mask] += diff / np.sum(mask)
            w = np.clip(w, 0, 1)
    
    return w

def _soft_min_obj_grad(h, beta):
    """Compute soft-min objective and gradient using FFT for correlations."""
    n = len(h)
    corrs = _compute_all_corrs(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    weights = exp_terms / sum_exp
    
    # Gradient: d(soft_min)/d(h[j]) = sum_k weights[k] * d(corrs_k)/d(h[j])
    # d(corrs_k)/d(h[j]) = (2/n) * (h[j+k] if j < n-k) + (2/n) * (h[j-k] if j >= k)
    # Vectorized: for each k, add weights[k]*(2/n)*h[j+k] to grad[j] for valid j
    grad_sm = np.zeros(n)
    active = np.where(weights > 1e-15)[0]
    if len(active) > 0:
        for k_idx in active:
            k = k_idx + 1
            w = weights[k_idx] * (2.0 / n)
            grad_sm[:n-k] += w * h[k:]
            grad_sm[k:] += w * h[:n-k]
    
    grad_obj = -grad_sm
    return obj, grad_obj

def _fista_optimize(h0, beta, max_iter=3000, tol=1e-9):
    """FISTA for constrained soft-min optimization."""
    n = len(h0)
    target_sum = n / 2.0
    
    x = h0.copy()
    x_prev = h0.copy()
    t = 1.0
    
    L = 50.0
    best_x = x.copy()
    best_obj = _soft_min_obj_grad(x, beta)[0]
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        if it > 0:
            t_prev = t
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x + (t_prev - 1.0) / (t + 1.0) * (x - x_prev)
        else:
            y = x.copy()
        
        obj_y, grad_y = _soft_min_obj_grad(y, beta)
        
        alpha = 1.0
        while alpha > 1e-5:
            z = y - alpha * grad_y
            z_proj = _project_to_simplex(z, target_sum)
            obj_z, _ = _soft_min_obj_grad(z_proj, beta)
            if obj_z <= obj_y + np.dot(grad_y, z_proj - y) + (1.0/(2.0*alpha)) * np.sum((z_proj - y)**2):
                break
            alpha *= 0.5
        
        x_new = _project_to_simplex(y - alpha * grad_y, target_sum)
        
        if np.sum((x_new - x)**2) < tol * tol:
            x = x_new
            break
        
        x_prev = x
        x = x_new
        
        obj_x, _ = _soft_min_obj_grad(x, beta)
        if obj_x < best_obj:
            best_obj = obj_x
            best_x = x.copy()
    
    return best_x, best_obj

def _pw_constant_from_impulses(n, impulses):
    """Build piecewise-constant signal from impulse positions and magnitudes.
    
    impulses: array of (position, magnitude) pairs.
    The signal starts at 0 and each impulse adds magnitude to all subsequent samples.
    """
    h = np.zeros(n)
    for pos, mag in impulses:
        pos = int(round(pos))
        if 0 <= pos < n:
            h[pos:] += mag
    return h

def _pw_constant_optimize(n, n_jumps, target_sum, max_iter=2000):
    """Optimize piecewise-constant signal with n_jumps discontinuities.
    
    Parameters: jump positions (n_jumps floats in [0,n]) and jump magnitudes (n_jumps+1 floats).
    The signal is h[i] = base + sum_{j: pos_j <= i} mag_j
    Constraint: sum(h) = target_sum, 0 <= h[i] <= 1
    """
    np.random.seed(42)
    
    # Parameterize: positions in [0, n], magnitudes
    # h[i] = m_0 + sum_{j: pos_j <= i} m_{j+1}
    # sum(h) = n*m_0 + sum_j m_{j+1} * (n - pos_j)
    
    best_h = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_h)
    
    # Multiple restarts with different initial jump structures
    for restart in range(8):
        if _time_remaining() < 10.0:
            break
        
        # Random initial positions
        if restart == 0:
            positions = np.linspace(0, n, n_jumps + 1)[1:-1] if n_jumps > 1 else np.array([n/2.0])
        else:
            positions = np.sort(np.random.uniform(0, n, n_jumps))
        
        # Initialize magnitudes to satisfy sum constraint
        # sum(h) = n*m_0 + sum_j m_{j+1} * (n - pos_j) = target_sum
        # Let's set m_0 = 0.5 and solve for others
        # Actually, let's parameterize differently:
        # h[i] = sum of step functions. Let's use a simpler approach:
        # h has n_jumps+1 constant segments. Optimize segment heights and boundaries.
        
        # Alternative: just optimize the full h vector but with a smoothness penalty
        # that encourages piecewise-constant structure
        pass
    
    # Simpler approach: use the full h vector but add a total variation penalty
    # to encourage piecewise-constant solutions
    x0 = np.full(n, 0.5)
    x0 = _project_to_simplex(x0, target_sum)
    
    # Add TV penalty to objective to encourage piecewise-constant
    def _obj_with_tv(h, beta, tv_weight):
        obj_sm, grad_sm = _soft_min_obj_grad(h, beta)
        # TV penalty: sum |h[i+1] - h[i]|
        diffs = np.diff(h)
        tv = np.sum(np.abs(diffs))
        # Gradient of TV (subgradient)
        grad_tv = np.zeros(n)
        for i in range(n-1):
            if diffs[i] > 0:
                grad_tv[i] -= 1.0
                grad_tv[i+1] += 1.0
            elif diffs[i] < 0:
                grad_tv[i] += 1.0
                grad_tv[i+1] -= 1.0
            else:
                grad_tv[i] += 0.0
                grad_tv[i+1] += 0.0
        total_obj = obj_sm + tv_weight * tv
        total_grad = grad_sm + tv_weight * grad_tv
        return total_obj, total_grad
    
    # Use FISTA with TV-regularized objective
    for tv_w in [0.01, 0.05, 0.1]:
        if _time_remaining() < 10.0:
            break
        
        # Modify FISTA to use TV-regularized objective
        x = x0.copy()
        x_prev = x.copy()
        t = 1.0
        beta = 50.0
        L = 50.0
        
        best_x_local = x.copy()
        best_obj_local = _obj_with_tv(x, beta, tv_w)[0]
        
        for it in range(1500):
            if _time_remaining() < 5.0:
                break
            
            if it > 0:
                t_prev = t
                t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                y = x + (t_prev - 1.0) / (t + 1.0) * (x - x_prev)
            else:
                y = x.copy()
            
            obj_y, grad_y = _obj_with_tv(y, beta, tv_w)
            
            alpha = 1.0
            while alpha > 1e-5:
                z = y - alpha * grad_y
                z_proj = _project_to_simplex(z, target_sum)
                obj_z, _ = _obj_with_tv(z_proj, beta, tv_w)
                if obj_z <= obj_y + np.dot(grad_y, z_proj - y) + (1.0/(2.0*alpha)) * np.sum((z_proj - y)**2):
                    break
                alpha *= 0.5
            
            x_new = _project_to_simplex(y - alpha * grad_y, target_sum)
            
            if np.sum((x_new - x)**2) < 1e-12:
                x = x_new
                break
            
            x_prev = x
            x = x_new
            
            obj_x, _ = _obj_with_tv(x, beta, tv_w)
            if obj_x < best_obj_local:
                best_obj_local = obj_x
                best_x_local = x.copy()
        
        # Evaluate with pure objective (no TV)
        x_eval = _project_to_simplex(best_x_local, target_sum)
        obj_pure = 1.0 - _compute_min_corr(x_eval)
        if obj_pure < best_obj:
            best_obj = obj_pure
            best_h = x_eval
    
    return best_h, best_obj

def _coordinate_polish(h, max_iter=150):
    """Coordinate descent polish for the hard min objective."""
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    
    best_obj = 1.0 - _compute_min_corr(h)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for j in range(n):
            if _time_remaining() < 5.0:
                break
            
            old_val = h[j]
            
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.002, -0.002]:
                h[j] = np.clip(old_val + delta, 0, 1)
                
                diff = target_sum - np.sum(h)
                if abs(diff) > 1e-10:
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    cnt = np.sum(mask)
                    if cnt > 0:
                        h[mask] += diff / cnt
                        h = np.clip(h, 0, 1)
                        diff2 = target_sum - np.sum(h)
                        if abs(diff2) > 1e-10:
                            mask2 = (h > 0) & (h < 1)
                            cnt2 = np.sum(mask2)
                            if cnt2 > 0:
                                h[mask2] += diff2 / cnt2
                                h = np.clip(h, 0, 1)
                
                obj = 1.0 - _compute_min_corr(h)
                if obj < best_obj - 1e-10:
                    best_obj = obj
                    improved = True
                    break
            
            if not improved:
                h[j] = old_val
        
        if not improved:
            break
    
    return h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_x)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(n, 0.5))
    
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        if k > 0 and (n-k) > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    for period in [3, 4, 5, 7]:
        pattern = [0.0 if (i % period) < period//2 else 1.0 for i in range(n)]
        mean_val = np.mean(pattern)
        if mean_val > 0:
            pattern = [p / (2*mean_val) for p in pattern]
        restarts.append(np.array(pattern))
    
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    x_grid = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5]:
        restarts.append(0.5 + 0.3*np.sin(freq*x_grid))
        restarts.append(0.5 + 0.3*np.cos(freq*x_grid))
    
    for ratio in [0.2, 0.3, 0.4]:
        k = int(n * ratio)
        if k > 0 and (n-k) > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 1.0/(1-ratio))]))
    
    # Optimize each restart with FISTA on soft-min objective
    for x0 in restarts:
        if _time_remaining() < 15.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        try:
            x_opt, obj_opt = _fista_optimize(x, beta=50.0, max_iter=2000, tol=1e-9)
            x_opt = _project_to_simplex(x_opt, target_sum)
            obj_hard = 1.0 - _compute_min_corr(x_opt)
            
            if obj_hard < best_obj:
                best_obj = obj_hard
                best_x = x_opt
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 20.0:
        for beta in [100.0, 200.0, 400.0]:
            if _time_remaining() < 10.0:
                break
            try:
                x_opt, obj_opt = _fista_optimize(best_x, beta=beta, max_iter=3000, tol=1e-10)
                x_opt = _project_to_simplex(x_opt, target_sum)
                obj_hard = 1.0 - _compute_min_corr(x_opt)
                if obj_hard < best_obj:
                    best_obj = obj_hard
                    best_x = x_opt
            except Exception:
                pass
    
    # Try piecewise-constant optimization
    if _time_remaining() > 25.0:
        try:
            pw_h, pw_obj = _pw_constant_optimize(n, 5, target_sum, max_iter=1500)
            if pw_obj < best_obj:
                best_obj = pw_obj
                best_x = pw_h
        except Exception:
            pass
    
    # Final coordinate descent polish
    if _time_remaining() > 10.0:
        best_x, best_obj = _coordinate_polish(best_x, max_iter=100)
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
