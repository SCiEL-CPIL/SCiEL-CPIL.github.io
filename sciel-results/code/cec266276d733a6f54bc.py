import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=30.0):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=30.0):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    val = 2.0 * N * soft_max / (s * s)
    obj = abs(val)
    
    # d(soft_max)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    # So d(soft_max)/d(h[j]) = 2 * sum_{k=j}^{min(j+N-1, 2N-2)} weights[k] * h[k-j]
    
    d_soft_max = np.zeros(N)
    for j in range(N):
        k_start = j
        k_end = min(j + N - 1, 2 * N - 2)
        if k_start > k_end:
            continue
        d_soft_max[j] = 2.0 * np.dot(weights[k_start:k_end+1], h[0:k_end-k_start+1])
    
    # d(obj)/d(h[j]) = sign(val) * 2N * (d_soft_max[j]/s^2 - 2*soft_max/s^3)
    sign = np.sign(val)
    if sign == 0:
        sign = 1.0
    
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    return grad, obj


def _chirp_h(freq, chirp_rate, dc_bias, amplitude):
    x = np.arange(N)
    return dc_bias + amplitude * np.cos(2 * np.pi * (freq * x / N + chirp_rate * x * x / (N * N)))


def _diverse_init():
    """Generate diverse initial candidates."""
    x = np.arange(N)
    candidates = []
    
    # Uniform baseline
    candidates.append(np.ones(N) / N)
    
    # Single frequency cosine
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Chirp patterns
    for freq in [20, 30, 50, 75, 100, 150]:
        for chirp_rate in [-10, -5, -1, 0, 1, 5, 10]:
            for dc_bias in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for amp in [1.0, 2.0, 3.0]:
                    h = _chirp_h(freq, chirp_rate, dc_bias, amp)
                    s = np.sum(h)
                    if abs(s) > 1e-12:
                        candidates.append(h / s)
    
    # Two-frequency products
    for k1 in [1, 2, 3, 5, 7, 10]:
        for k2 in [11, 13, 17, 20, 25, 30, 40, 50]:
            for amp in [0.5, 1.0, 1.5, 2.0]:
                h = np.ones(N) / N + amp * np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N)
                s = np.sum(h)
                if abs(s) > 1e-12:
                    candidates.append(h / s)
    
    # Find best
    best_h = None
    best_val = 1e18
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective with L-BFGS-B."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Try different beta values
    for beta in [20.0, 30.0, 50.0, 80.0]:
        if time.time() - t0 > max_time * 0.7:
            break
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=beta),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=beta)[0],
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start_local(h_best, n_starts=8, max_time=40.0):
    """Multi-start local refinement with diverse perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.05 + 0.08 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=30.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=30.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _chirp_grid_search(max_time=25.0):
    """Grid search over chirp parameters."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    # Coarse grid
    for freq in range(10, 200, 5):
        if time.time() - t0 > max_time * 0.6:
            break
        for chirp_rate in np.linspace(-20, 20, 9):
            for dc_bias in np.linspace(-1.5, 1.5, 7):
                for amp in np.linspace(0.5, 4.0, 8):
                    h = _chirp_h(freq, chirp_rate, dc_bias, amp)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    # Fine grid around best
    if best_h is not None and time.time() - t0 < max_time * 0.8:
        # We don't know the exact params that gave best, so just do local refine
        pass
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init()
    
    # Phase 2: Chirp grid search
    h_chirp, val_chirp = _chirp_grid_search(max_time=min(20.0, time_left() - 30.0))
    
    # Pick best
    if val_chirp <= val_div:
        best_h = h_chirp.copy()
        best_val = val_chirp
    else:
        best_h = h_div.copy()
        best_val = val_div
    
    # Include seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 3: Local refinement
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(35.0, time_left() - 15.0))
    
    # Phase 4: Multi-start local refinement
    if time_left() > 25:
        best_h, best_val = _multi_start_local(best_h, n_starts=10, max_time=min(40.0, time_left() - 10.0))
    
    # Phase 5: Final high-precision polish
    if time_left() > 10:
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=80.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=80.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1200, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish
    if time_left() > 5:
        try:
            res = minimize(
                _value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
