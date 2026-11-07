"""Second autocorrelation inequality - gradient-based optimization with smooth surrogates."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        return -np.inf
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)  # spacing from linspace(-0.5, 0.5, m+2)
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_objective(params, sharpness):
    """Negative of smooth approximation of the value.
    params: (a, b) where h = a + b * shape, shape fixed pattern.
    Actually we optimize h directly in log-space or with soft constraints.
    """
    h = np.exp(params)  # ensures h > 0
    if np.any(~np.isfinite(h)):
        return 1e10
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    abs_g = np.abs(g)
    norm_1 = np.sum(abs_g) / (m + 1)
    # Smooth max via log-sum-exp
    max_val = np.max(abs_g)
    if max_val == 0:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(abs_g - max_val)))
    norm_inf_smooth = log_sum_exp / sharpness  # soft-max approximation
    # Actually log-sum-exp gives sum of exp, not max. Let me use a proper soft-max.
    # soft-max = (1/beta) * log(sum(exp(beta*x)))
    beta = sharpness
    norm_inf_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * abs_g / max_val))) * max_val
    # Hmm, let me reconsider. For soft-max of values x_i:
    # soft_max = (1/beta) * log(sum(exp(beta * x_i)))
    # As beta -> inf, this approaches max(x_i)
    norm_inf_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * abs_g)))
    # But this can overflow. Let's use the stable version:
    max_abs = np.max(abs_g)
    norm_inf_smooth = max_abs + (1.0 / beta) * np.log(np.sum(np.exp(beta * (abs_g - max_abs))))
    
    if norm_1 == 0:
        return 1e10
    value = l2 / (norm_1 * norm_inf_smooth)
    return -value

def _try_optimize(h0, max_iter, tol):
    """Run gradient optimization starting from h0."""
    if _deadline_hit():
        return h0, _compute_value(h0)
    
    h0 = np.asarray(h0, dtype=float)
    if np.any(h0 < 0):
        h0 = np.abs(h0)
    
    # Work in log-space to maintain nonnegativity: h = exp(log_h)
    h0_safe = np.maximum(h0, 1e-10)
    log_h0 = np.log(h0_safe)
    
    best_h = h0
    best_val = _compute_value(h0)
    
    # Try a few different sharpness levels for the soft-max
    sharpnesses = [5.0, 10.0, 20.0, 50.0]
    
    for beta in sharpnesses:
        if _deadline_hit():
            break
        try:
            def obj(x):
                return _smooth_objective(x, beta)
            
            def grad(x):
                # Numerical gradient
                eps = 1e-7
                g = np.zeros_like(x)
                for i in range(len(x)):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    g[i] = (obj(xp) - obj(xm)) / (2 * eps)
                return g
            
            res = minimize(obj, log_h0, method='L-BFGS-B', jac=grad,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': max_iter * 2})
            
            if res.success or res.fun < obj(log_h0):
                h_new = np.exp(res.x)
                if np.all(np.isfinite(h_new)) and np.all(h_new >= 0):
                    val_new = _compute_value(h_new)
                    if val_new > best_val:
                        best_val = val_new
                        best_h = h_new
                        log_h0 = res.x  # warm-start for next beta
        except Exception:
            pass
    
    # Final polish with direct finite-difference on h (constrained)
    if not _deadline_hit() and best_val > 0:
        try:
            def direct_obj(x):
                h = np.maximum(x, 0.0)
                v = _compute_value(h)
                return -v if np.isfinite(v) else 1e10
            
            res2 = minimize(direct_obj, best_h, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if np.all(np.isfinite(res2.x)):
                h_new = np.maximum(res2.x, 0.0)
                val_new = _compute_value(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    global _START
    _START = time.time()
    
    # Seed: 50 equal heights
    h_seed = np.ones(N)
    seed_val = _compute_value(h_seed)
    
    best_h = h_seed.copy()
    best_val = seed_val
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starting_points = []
    # 1. Uniform
    starting_points.append(np.ones(N))
    # 2. Sine wave (low frequency)
    x = np.linspace(0, np.pi, N)
    starting_points.append(1.0 + 0.5 * np.sin(x))
    # 3. Cosine bell
    starting_points.append(1.0 + 0.5 * np.cos(2 * np.pi * x / N))
    # 4. Random smooth
    for _ in range(5):
        noise = rng.standard_normal(N)
        # Smooth it
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(noise, sigma=2)
        h = 1.0 + 0.3 * smooth / (np.max(np.abs(smooth)) + 1e-10)
        h = np.maximum(h, 0.1)
        starting_points.append(h)
    # 5. Triangular
    starting_points.append(1.0 + 0.4 * np.abs(2.0 * np.arange(N) / N - 1.0))
    # 6. Flat top with edges
    h_flat = np.ones(N)
    h_flat[:5] = 1.3
    h_flat[-5:] = 1.3
    starting_points.append(h_flat)
    
    # Evaluate all starting points
    for hp in starting_points:
        if _deadline_hit():
            break
        hp = np.maximum(hp, 0.0)
        val = _compute_value(hp)
        if val > best_val:
            best_val = val
            best_h = hp.copy()
    
    # Optimize from the best few starting points
    # Sort starting points by value
    scored = []
    for hp in starting_points:
        hp = np.maximum(hp, 0.0)
        scored.append((_compute_value(hp), hp))
    scored.sort(key=lambda x: -x[0])
    
    # Optimize top 3 starting points
    time_budget_per_opt = 30  # seconds
    for i, (val_init, hp) in enumerate(scored[:3]):
        if _deadline_hit():
            break
        if val_init <= best_val and i > 0:
            continue
        h_opt, val_opt = _try_optimize(hp, max_iter=100, tol=1e-8)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # If we still have time, do one more round from the best
    if not _deadline_hit() and best_val > seed_val:
        h_opt, val_opt = _try_optimize(best_h, max_iter=150, tol=1e-10)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final check
    final_h = np.maximum(best_h, 0.0)
    final_val = _compute_value(final_h)
    if final_val < seed_val:
        final_h = h_seed
        final_val = seed_val
    
    return list(final_h)
