import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

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

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    return m + np.log(np.sum(exp_conv)) / alpha

def _smooth_max_grad(conv, alpha):
    """Gradient of smooth max w.r.t. conv elements."""
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    weights = exp_conv / np.sum(exp_conv)
    return weights

def _parametric_heights(params):
    """Generate heights from parametric representation.
    
    params: [center, width, taper_power, amplitude]
    - center: center position in [0, 1] (normalized)
    - width: effective width in [0, 1]
    - taper_power: exponent for super-ellipse taper (higher = more rectangular)
    - amplitude: overall scale
    
    The shape is: amplitude * (1 - (|x - center| / width)^p)^(1/p) for |x-center| < width, else 0
    This is a super-ellipse / generalized cosine window.
    """
    center = params[0]
    width = params[1]
    taper_power = params[2]
    amplitude = params[3]
    
    x = np.linspace(0, 1, N)
    d = np.abs(x - center)
    
    # Super-ellipse shape
    ratio = d / max(width, 1e-10)
    mask = ratio < 1.0
    heights = np.zeros(N)
    # Use (1 - ratio^p)^(1/p) which gives:
    # p=1: triangular
    # p=2: parabolic
    # p->inf: rectangular (top-hat)
    p = max(taper_power, 0.1)
    heights[mask] = amplitude * (1.0 - ratio[mask] ** p) ** (1.0 / p)
    
    return np.maximum(heights, 0.0)

def _parametric_score(params):
    """Score for parametric representation."""
    h = _parametric_heights(params)
    if np.sum(h) <= 0:
        return 1e10
    return compute_score(h)

def _smooth_obj_and_grad(heights, alpha, M=None):
    """Compute smooth objective and gradient for L-BFGS-B."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    sm = _smooth_max(conv, alpha)
    total = np.sum(heights)
    w = 0.5 / len(heights)
    if total <= 1e-10:
        return 1e10, np.zeros(len(heights))
    
    # Gradient of smooth_max via log-sum-exp
    weights = _smooth_max_grad(conv, alpha)
    
    # Use FFT for efficient correlation
    W_full = np.fft.rfft(weights)
    F_full = np.fft.rfft(f)
    corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
    grad_f = 2.0 * corr[:len(heights)]
    
    S = total
    obj = sm * len(heights) / (S * S * w)
    obj_grad = grad_f * (len(heights) / (S * S * w)) - sm * (2.0 * len(heights) / (S * S * S * w))
    
    return obj, obj_grad

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
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Parametric search with Nelder-Mead
    # Parameters: [center, width, taper_power, amplitude]
    # center in [0, 1], width in [0.1, 1.0], taper_power in [0.5, 20], amplitude in [0.1, 2.0]
    
    param_bounds = [
        (0.2, 0.8),   # center
        (0.1, 1.0),   # width
        (0.5, 20.0),  # taper_power
        (0.1, 2.0)    # amplitude
    ]
    
    # Grid search over center and width first
    center_vals = np.linspace(0.3, 0.7, 5)
    width_vals = np.linspace(0.3, 1.0, 6)
    taper_vals = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    for c in center_vals:
        if time_left() < 10:
            break
        for wdt in width_vals:
            if time_left() < 5:
                break
            for tp in taper_vals:
                if time_left() < 3:
                    break
                for amp in [0.5, 1.0, 1.5]:
                    params = [c, wdt, tp, amp]
                    score = _parametric_score(params)
                    if score < best_score:
                        best_score = score
                        best_h = _parametric_heights(params).copy()
    
    # Nelder-Mead refinement on best parametric region
    if time_left() > 20:
        # Find best parametric point from grid
        best_params = None
        best_param_score = best_score
        for c in center_vals:
            for wdt in width_vals:
                for tp in taper_vals:
                    for amp in [0.5, 1.0, 1.5]:
                        params = [c, wdt, tp, amp]
                        s = _parametric_score(params)
                        if s < best_param_score:
                            best_param_score = s
                            best_params = params
        
        if best_params is not None:
            def param_obj(p):
                return _parametric_score(p)
            
            try:
                result = minimize(
                    param_obj,
                    best_params,
                    method='Nelder-Mead',
                    bounds=param_bounds,
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.x is not None:
                    h_new = _parametric_heights(result.x)
                    score = compute_score(h_new)
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
            except Exception:
                pass
    
    # Strategy 2: Gradient-based optimization with smooth surrogate
    alpha_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for alpha_val in alpha_schedule:
        if time_left() < 15:
            break
        
        # Start from best so far
        h_opt = best_h.copy()
        
        def objective_wrapper(h):
            obj, _ = _smooth_obj_and_grad(h, alpha_val, M)
            return obj
        
        def gradient_wrapper(h):
            _, grad = _smooth_obj_and_grad(h, alpha_val, M)
            return grad
        
        try:
            result = minimize(
                objective_wrapper,
                h_opt,
                jac=gradient_wrapper,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.x is not None and np.all(result.x >= -1e-10):
                h_new = np.maximum(result.x, 0.0)
                score = compute_score(h_new)
                if score < best_score:
                    best_score = score
                    best_h = h_new.copy()
        except Exception:
            pass
        
        # Random restarts
        for restart in range(4):
            if time_left() < 10:
                break
            
            freqs = np.arange(N)
            h_init = np.ones(N)
            num_modes = rng.integers(3, 8)
            for _ in range(num_modes):
                k = rng.integers(1, N // 4)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.05, 0.25)
                h_init += amp * np.cos(2 * np.pi * k * freqs / N + phase)
            h_init = np.maximum(h_init, 0.0)
            if np.sum(h_init) <= 0:
                continue
            
            def objective_wrapper_r(h):
                obj, _ = _smooth_obj_and_grad(h, alpha_val, M)
                return obj
            
            def gradient_wrapper_r(h):
                _, grad = _smooth_obj_and_grad(h, alpha_val, M)
                return grad
            
            try:
                result_r = minimize(
                    objective_wrapper_r,
                    h_init,
                    jac=gradient_wrapper_r,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                if result_r.x is not None and np.all(result_r.x >= -1e-10):
                    h_new_r = np.maximum(result_r.x, 0.0)
                    score_r = compute_score(h_new_r)
                    if score_r < best_score:
                        best_score = score_r
                        best_h = h_new_r.copy()
            except Exception:
                pass
    
    # Strategy 3: Final high-precision polish
    if time_left() > 15:
        h_final = best_h.copy()
        alpha_f = 500.0
        
        def objective_final(h):
            obj, _ = _smooth_obj_and_grad(h, alpha_f, M)
            return obj
        
        def gradient_final(h):
            _, grad = _smooth_obj_and_grad(h, alpha_f, M)
            return grad
        
        try:
            result_f = minimize(
                objective_final,
                h_final,
                jac=gradient_final,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_f.x is not None and np.all(result_f.x >= -1e-10):
                h_new_f = np.maximum(result_f.x, 0.0)
                score_f = compute_score(h_new_f)
                if score_f < best_score:
                    best_score = score_f
                    best_h = h_new_f.copy()
        except Exception:
            pass
    
    # Strategy 4: Local coordinate descent polish
    if time_left() > 10:
        h_cd = best_h.copy()
        for iteration in range(200):
            if time_left() < 3:
                break
            improved = False
            for i in range(0, N, 3):  # Step by 3 for speed
                if time_left() < 2:
                    break
                # Try small perturbations
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    h_trial = h_cd.copy()
                    h_trial[i] = max(0.0, h_trial[i] + delta)
                    score_trial = compute_score(h_trial)
                    if score_trial < best_score:
                        h_cd = h_trial
                        best_score = score_trial
                        best_h = h_cd.copy()
                        improved = True
                        break
        best_h = h_cd.copy()
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
