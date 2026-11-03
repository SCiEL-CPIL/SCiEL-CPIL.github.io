"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1
    using FFT for O(n log n) complexity."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    corr_full = np.fft.irfft(f * np.conj(f), size)
    corrs = corr_full[1:n] * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _compute_max_overlap(h):
    """Compute max overlap: max_k (1 - R_k) = 1 - min_k R_k."""
    return 1.0 - _compute_cross_correlation_min(h)

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

def _soft_min_and_grad_fft(h, beta):
    """Compute soft_min of correlations and its gradient w.r.t. h, using FFT for O(n log n)."""
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    weights = exp_terms / sum_exp  # shape (n-1,)
    
    # Compute gradient via FFT: grad[j] = (2/n) * sum_k weights[k-1] * (h[j+k] + h[j-k])
    # This is equivalent to a weighted autocorrelation
    # Use FFT: if we define w_ext[k] = weights[k-1] for k=1..n-1, w_ext[0]=0, w_ext[n-k]=weights[k-1]
    # Then grad = (2/n) * ifft(fft(h) * fft(w_ext)) where w_ext is the kernel
    
    # Build the kernel for convolution
    kernel = np.zeros(n)
    # For each shift k (1-indexed), weight is weights[k-1]
    # grad[j] = (2/n) * sum_{k=1}^{n-1} weights[k-1] * h[(j+k) % n] (circular)
    # But we need linear correlation, not circular
    # For linear: grad[j] = (2/n) * sum_{k: j+k < n} weights[k-1]*h[j+k] + (2/n) * sum_{k: j-k >= 0} weights[k-1]*h[j-k]
    
    # Let's use the direct approach but vectorized with matrix operations
    # For n=100, this is 100*99/2 ~ 5000 operations per call, which is fine
    # But let's try to make it faster with FFT-based approach
    
    # Actually, for the gradient of soft_min:
    # d(soft_min)/d(h[j]) = sum_k weights[k] * d(corrs_k)/d(h[j])
    # d(corrs_k)/d(h[j]) = (2/n) * (h[j+k] if j < n-k else 0) + (2/n) * (h[j-k] if j >= k else 0)
    
    # We can compute this as: grad = (2/n) * (convolve(h, w_kernel) + convolve(h, w_kernel_rev))
    # where w_kernel[k] = weights[k-1] for k=1..n-1
    
    # Let's just do it directly - for n=100 it's fast enough
    grad = np.zeros(n)
    active_idx = np.where(weights > 1e-15)[0]
    if len(active_idx) > 0:
        for k_idx in active_idx:
            k = k_idx + 1
            w = weights[k_idx] * (2.0 / n)
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    return soft_min, grad

def _soft_min_obj_grad(h, beta):
    """Objective and gradient for soft-min formulation."""
    soft_min, grad_h = _soft_min_and_grad_fft(h, beta)
    obj = 1.0 - soft_min
    grad = -grad_h
    return obj, grad

def _fixed_point_iteration(h, target_sum, beta, max_iters=100, tol=1e-10):
    """Fixed-point iteration based on optimality conditions.
    
    The optimality condition for minimizing max_k (1 - R_k) subject to constraints is:
    At the optimum, the gradient of the active constraints should be balanced.
    
    We use a soft-min regularization: minimize (1/beta) * log(sum_k exp(beta*(1-R_k)))
    The gradient w.r.t. h is: -sum_k w_k * grad(R_k) where w_k = softmax(-beta*(1-R_k))
    
    Fixed-point: h_new = project(h - lr * grad_soft_max)
    """
    n = len(h)
    h = h.copy()
    h = _project_to_simplex(h, target_sum)
    
    best_obj = _compute_max_overlap(h)
    lr = 0.01
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Compute soft-max of (1 - R_k) and its gradient
        corrs = _compute_all_corrs_fft(h)
        vals = 1.0 - corrs  # shape (n-1,)
        
        max_val = np.max(vals)
        shifted = vals - max_val
        exp_terms = np.exp(beta * shifted)
        sum_exp = np.sum(exp_terms)
        log_sum = np.log(sum_exp)
        soft_max = max_val + log_sum / beta
        
        weights = exp_terms / sum_exp  # softmax weights
        
        # Compute gradient of soft_max w.r.t. h
        # d(soft_max)/d(h[j]) = sum_k weights[k] * d(1-R_k)/d(h[j])
        # = -sum_k weights[k] * d(R_k)/d(h[j])
        # d(R_k)/d(h[j]) = (2/n) * (h[j+k] if j < n-k else 0) + (2/n) * (h[j-k] if j >= k else 0)
        
        grad = np.zeros(n)
        active_idx = np.where(weights > 1e-15)[0]
        if len(active_idx) > 0:
            for k_idx in active_idx:
                k = k_idx + 1
                w = weights[k_idx] * (2.0 / n)
                grad[:n-k] += w * h[k:]
                grad[k:] += w * h[:n-k]
        
        # Gradient of soft_max is -grad (since we're minimizing soft_max of 1-R)
        # So descent direction is -(-grad) = grad... wait
        # We want to minimize soft_max = (1/beta)*log(sum exp(beta*(1-R_k)))
        # d(soft_max)/d(h[j]) = sum_k weights[k] * d(1-R_k)/d(h[j]) = -sum_k weights[k] * d(R_k)/d(h[j])
        # So the gradient IS -grad (where grad was computed as sum_k weights[k]*d(R_k)/d(h[j]))
        # Descent: h_new = h - lr * (-grad) = h + lr * grad
        
        h_new = h + lr * grad
        h_new = _project_to_simplex(h_new, target_sum)
        
        new_obj = _compute_max_overlap(h_new)
        
        if new_obj < best_obj - 1e-12:
            h = h_new
            best_obj = new_obj
            # Adapt learning rate
            if it % 10 == 0 and it > 0:
                lr = min(lr * 1.2, 0.05)
        else:
            # Reduce learning rate
            lr *= 0.7
            if lr < 1e-6:
                break
        
        # Check convergence
        if it > 5 and abs(best_obj - _compute_max_overlap(h)) < tol:
            break
    
    return h, best_obj

def _coordinate_polish(h, target_sum, max_iters=80):
    """Coordinate-wise descent to polish the solution."""
    n = len(h)
    h = h.copy()
    h = _project_to_simplex(h, target_sum)
    
    best_obj = _compute_max_overlap(h)
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for i in range(n):
            if _time_remaining() < 5.0:
                break
            
            old_val = h[i]
            found_improvement = False
            for delta in step_sizes:
                for sign in [1, -1]:
                    new_val = old_val + sign * delta
                    if new_val < 0 or new_val > 1:
                        continue
                    
                    h[i] = new_val
                    diff = target_sum - np.sum(h)
                    if abs(diff) > 1e-15:
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        cnt = np.sum(mask)
                        if cnt == 0:
                            h[i] = old_val
                            continue
                        h[mask] += diff / cnt
                        h = np.clip(h, 0, 1)
                        h = _project_to_simplex(h, target_sum)
                    
                    new_obj = _compute_max_overlap(h)
                    
                    if new_obj < best_obj - 1e-13:
                        best_obj = new_obj
                        improved = True
                        found_improvement = True
                        break
                    else:
                        h[i] = old_val
                if found_improvement:
                    break
            
            if not improved and it > 30:
                break
    
    return h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = _compute_max_overlap(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if k > 0 and n - k > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # More bang-bang patterns
    for period in [3, 4, 5, 7]:
        pattern = [0.0 if (i // period) % 2 == 0 else 1.0 for i in range(n)]
        arr = np.array(pattern)
        curr_sum = np.sum(arr)
        if curr_sum > 0:
            arr = arr * (target_sum / curr_sum)
            arr = np.clip(arr, 0, 1)
            arr = _project_to_simplex(arr, target_sum)
            restarts.append(arr)
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Near-optimal structured patterns
    for p in [3, 4, 5, 6, 7, 8, 9, 10]:
        for a in [0.1, 0.2, 0.3, 0.4, 0.45]:
            pattern = 0.5 + a * np.cos(2 * np.pi * np.arange(n) / p)
            pattern = np.clip(pattern, 0, 1)
            pattern = _project_to_simplex(pattern, target_sum)
            restarts.append(pattern)
            pattern2 = 0.5 + a * np.sin(2 * np.pi * np.arange(n) / p)
            pattern2 = np.clip(pattern2, 0, 1)
            pattern2 = _project_to_simplex(pattern2, target_sum)
            restarts.append(pattern2)
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for amp in [0.2, 0.3, 0.4, 0.5]:
        for freq in [1, 2, 3, 4]:
            restarts.append(0.5 + amp*np.sin(freq*x))
            restarts.append(0.5 + amp*np.cos(freq*x))
    
    # Phase 1: Optimize each restart with soft-min objective (moderate beta)
    for x0 in restarts:
        if _time_remaining() < 15.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        try:
            res = minimize(
                _soft_min_obj_grad,
                x,
                args=(20.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = _compute_max_overlap(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Phase 2: Fixed-point iteration on best solution
    if _time_remaining() > 25.0:
        for beta in [30.0, 60.0, 100.0, 200.0]:
            if _time_remaining() < 10.0:
                break
            try:
                fp_x, fp_obj = _fixed_point_iteration(best_x, target_sum, beta, max_iters=80)
                if fp_obj < best_obj:
                    best_obj = fp_obj
                    best_x = fp_x
            except Exception:
                pass
    
    # Phase 3: Polish with higher beta L-BFGS-B
    if _time_remaining() > 15.0:
        for beta in [50.0, 100.0, 200.0, 400.0]:
            if _time_remaining() < 8.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = _compute_max_overlap(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 4: Final coordinate-wise polish
    if _time_remaining() > 8.0:
        best_x, best_obj = _coordinate_polish(best_x, target_sum, max_iters=100)
    
    # Phase 5: One more high-beta pass if time allows
    if _time_remaining() > 10.0:
        try:
            res = minimize(
                _soft_min_obj_grad,
                best_x,
                args=(500.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            res_x = _project_to_simplex(res.x, target_sum)
            res_obj = _compute_max_overlap(res_x)
            if res_obj < best_obj:
                best_obj = res_obj
                best_x = res_x
        except Exception:
            pass
    
    # Phase 6: Final coordinate polish
    if _time_remaining() > 5.0:
        best_x, best_obj = _coordinate_polish(best_x, target_sum, max_iters=50)
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-14:
        mask = (best_x > 1e-15) & (best_x < 1.0 - 1e-15)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
