"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses minimax epigraph formulation with SLSQP, coordinate descent polish,
and multi-start spectral initialization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 138  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _epigraph_obj_grad(x):
    """
    Minimax epigraph: minimize t subject to C_k(h) <= t for all k.
    x = [h_0, ..., h_{N-1}, t]
    Returns (objective_value, gradient).
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Objective: maximize the violation (we want to minimize t, so use t + max(0, cs - t))
    # Actually, for SLSQP we minimize t with constraints cs_k <= t.
    # Objective is just t.
    obj_val = t
    
    # Gradient: d(obj)/dx
    grad = np.zeros(N + 1)
    grad[N] = 1.0  # dt/dt = 1
    
    # For the constraint gradients, we need dC_k/dh_i
    # But SLSQP handles constraints separately.
    # The gradient of the objective is just [0,...,0, 1].
    
    return obj_val, grad


def _epigraph_constraint_grad(x):
    """
    Constraint: C_k(h) <= t for all k.
    Returns array of (constraint_value, constraint_gradient) for all k.
    constraint_value_k = C_k(h) - t (should be <= 0)
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    n = len(h)
    # Values: C_k - t for k=0..N-1
    vals = cs - t
    
    # We'll return the constraint values; SLSQP needs individual constraints
    # or we can use a vectorized inequality.
    # For efficiency, let's just compute the values and let SLSQP handle it.
    # Actually scipy SLSQP expects separate constraint dicts.
    # Instead, let's use a single inequality constraint with a large penalty approach,
    # or better: use the smooth max as a differentiable surrogate.
    
    return vals


def _smooth_max_val_grad(h, beta):
    """
    Gradient of the smooth maximum objective (log-sum-exp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max

    val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)

    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk

        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]

    return val, grad


def _minimax_slsqp(h0, maxiter=500, ftol=1e-12, gtol=1e-10):
    """
    Direct minimax optimization via SLSQP with epigraph variable t.
    Minimize t subject to C_k(h) <= t, 0 <= h_i <= 1, sum(h) = N/2.
    """
    n = len(h0)
    
    def objective(x):
        return x[N]  # t
    
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
    
    # Build constraints: C_k(h) - t <= 0 for each k
    # That's N inequality constraints + 1 equality (sum) + bounds
    
    # For SLSQP, we need to provide constraint functions.
    # With N=100, that's 100 inequality constraints which is heavy.
    # Alternative: use a single smooth-max constraint with high beta.
    
    beta = 200.0
    
    def smooth_max_constraint(x):
        h = x[:N]
        t = x[N]
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        smooth_max = m + np.log(sum_exp) / beta
        return smooth_max - t
    
    def smooth_max_constraint_grad(x):
        h = x[:N]
        t = x[N]
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        
        grad = np.zeros(N + 1)
        grad[N] = -1.0  # d/dt of (smooth_max - t) = -1
        
        for k in range(N):
            wk = w[k]
            if wk < 1e-15:
                continue
            factor = -(2.0 / N) * wk
            if k > 0:
                grad[k:] += factor * h[:N - k]
            grad[:N - k] += factor * h[k:]
        
        return grad
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N / 2.0,
         'jac': lambda x: np.concatenate([np.ones(N), [0.0]])},
        {'type': 'ineq', 'fun': smooth_max_constraint,
         'jac': smooth_max_constraint_grad}
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 2.0)]
    
    x0 = np.concatenate([h0, [np.max(_linear_overlap_values(h0)) + 0.1]])
    
    try:
        res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if res.success or res.fun < objective(x0):
            return res.x[:N]
    except Exception:
        pass
    
    return h0


def _coordinate_descent_polish(h0, max_iter=2000):
    """
    Coordinate descent: for each h_i, adjust to reduce the max overlap.
    This is a structural alternative to gradient methods.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        improved = False
        for idx in range(N):
            old_val = h[idx]
            # Try a small perturbation
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h[idx] = np.clip(old_val + delta, 0.0, 1.0)
                # Fix sum
                diff = N / 2.0 - np.sum(h)
                if abs(diff) > 1e-12:
                    h[idx] = np.clip(h[idx] - diff, 0.0, 1.0)
                
                s = _score(h)
                if s < best_score - 1e-14:
                    best_score = s
                    improved = True
                    break
            h[idx] = old_val  # Restore, will be set if improved
        
        if not improved:
            # Try larger steps
            for idx in range(N):
                old_val = h[idx]
                for delta in [0.05, -0.05, 0.1, -0.1]:
                    h[idx] = np.clip(old_val + delta, 0.0, 1.0)
                    diff = N / 2.0 - np.sum(h)
                    if abs(diff) > 1e-12:
                        h[idx] = np.clip(h[idx] - diff, 0.0, 1.0)
                    
                    s = _score(h)
                    if s < best_score - 1e-14:
                        best_score = s
                        improved = True
                        break
                h[idx] = old_val
            
            if not improved:
                break
    
    return h


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)

    # Second pass if clipping changed sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


def _spectral_init(frequencies, amplitudes, phases=None):
    """Generate a structured spectral initialization."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    for f, a in zip(frequencies, amplitudes):
        if phases is not None:
            phase = phases[frequencies.index(f)]
        else:
            phase = 0.0
        h += a * np.cos(2.0 * np.pi * f * i / N + phase)
    return _fix_sum(h)


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK] where h ~ 0.5 + sum (a_k cos + b_k sin)
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _projected_gradient_step(h, grad, lr):
    """
    One projected gradient step: move in -grad direction,
    then project onto {h: 0 <= h_i <= 1, sum(h) = N/2}.
    """
    h_new = h - lr * grad
    h_new = np.clip(h_new, 0.0, 1.0)
    h_new = _fix_sum(h_new)
    return h_new


def _nesterov_smooth_max(h0, beta, max_iter, lr_init=0.01):
    """
    Nesterov accelerated projected gradient on smooth max with box+sum constraints.
    Returns (h, score).
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init
    best_h = h.copy()
    best_score = _score(h)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = _projected_gradient_step(h, grad, lr)

        # Nesterov extrapolation
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate: decay slowly
        if it % 100 == 99:
            lr = max(0.002, lr * 0.9)

        # Track best
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    return best_h, best_score


def _optimize_spectral(h_init, K, beta, maxiter, lr_init=0.01):
    """
    Optimize in spectral coefficient space using L-BFGS-B with penalty for bounds.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2)
        val, _ = _smooth_max_val_grad(h, beta)
        return val + 1.0 * penalty

    def grad(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        _, grad_h = _smooth_max_val_grad(h, beta)

        # Gradient of penalty
        penalty_grad = np.zeros(N)
        penalty_grad[h < 0] += -2.0 * h[h < 0]
        penalty_grad[h > 1] += 2.0 * (h[h > 1] - 1.0)

        total_grad_h = grad_h + penalty_grad

        # Chain rule: d(total)/d(coeffs)
        n_modes = len(coeffs) // 2
        i = np.arange(N)
        grad_coeffs = np.zeros(len(coeffs))
        for k in range(1, n_modes + 1):
            cos_k = np.cos(2.0 * np.pi * k * i / N)
            sin_k = np.sin(2.0 * np.pi * k * i / N)
            grad_coeffs[2 * (k - 1)] = np.sum(total_grad_h * cos_k)
            grad_coeffs[2 * (k - 1) + 1] = np.sum(total_grad_h * sin_k)

        return grad_coeffs

    # Use L-BFGS-B for spectral optimization
    try:
        res = minimize(obj, coeffs0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(coeffs0):
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass

    # Fallback: Nesterov in spectral space
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = lr_init
    best_coeffs = coeffs.copy()
    best_val = obj(coeffs)

    for it in range(max(100, maxiter)):
        if time.time() > _DEADLINE - 10:
            break
        val = obj(coeffs)
        g = grad(coeffs)

        coeffs_new = coeffs - lr * g
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.001, lr * 0.95)

        v = obj(coeffs)
        if v < best_val:
            best_val = v
            best_coeffs = coeffs.copy()

    h_opt = _spectral_h_from_coeffs(best_coeffs)
    return h_opt


def _final_polish(h0):
    """
    Multi-stage final polish: minimax SLSQP + coordinate descent.
    """
    h = h0.copy()
    
    # Stage 1: Minimax SLSQP
    if time.time() < _DEADLINE - 30:
        h_mm = _minimax_slsqp(h, maxiter=400, ftol=1e-12, gtol=1e-10)
        h_mm = _fix_sum(h_mm)
        if _score(h_mm) < _score(h):
            h = h_mm
    
    # Stage 2: Coordinate descent polish
    if time.time() < _DEADLINE - 10:
        h_cd = _coordinate_descent_polish(h, max_iter=1000)
        h_cd = _fix_sum(h_cd)
        if _score(h_cd) < _score(h):
            h = h_cd
    
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # --- Phase 1: Diverse structured starts ---
    starts = []

    # Cosine fundamental
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_cos))

    # Sine fundamental
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(_fix_sum(h_sin))

    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))

    # Three-frequency
    h_3f = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N) + 0.15 * np.cos(6.0 * np.pi * i / N)
    starts.append(_fix_sum(h_3f))

    # Four-frequency
    h_4f = 0.5 + 0.2 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N) + 0.1 * np.cos(6.0 * np.pi * i / N) + 0.08 * np.cos(8.0 * np.pi * i / N)
    starts.append(_fix_sum(h_4f))

    # Ramp pattern
    h_ramp = np.linspace(0.2, 0.8, N)
    starts.append(_fix_sum(h_ramp))

    # Triangular wave
    h_tri = 0.5 + 0.3 * (2 * np.abs(2 * (i / N - 0.5) - 1) - 1)
    starts.append(_fix_sum(h_tri))

    # Alternating pattern
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    starts.append(_fix_sum(h_alt))

    # Block pattern
    h_block = np.zeros(N)
    h_block[:N//2] = 0.8
    h_block[N//2:] = 0.2
    starts.append(_fix_sum(h_block))

    # Random sines
    for _ in range(15):
        f = np.random.uniform(1, 8)
        phase = np.random.uniform(0, 2 * np.pi)
        h_s = 0.5 + 0.5 * np.sin(2 * np.pi * f * i / N + phase)
        starts.append(_fix_sum(h_s))

    # Random
    for _ in range(15):
        h_r = np.random.rand(N)
        starts.append(_fix_sum(h_r))

    for h0 in starts:
        _update_best(h0)

    # --- Phase 2: Minimax SLSQP from diverse starts ---
    # Run minimax on several of the best initial starts
    start_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # structured starts
    for idx in start_indices:
        if time.time() > _DEADLINE - 50:
            break
        h_mm = _minimax_slsqp(starts[idx], maxiter=300, ftol=1e-11, gtol=1e-9)
        _update_best(h_mm)

    # --- Phase 3: Spectral optimization from best start ---
    for K in [4, 8, 16]:
        if time.time() - t0 > _DEADLINE - 45:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=150)
        _update_best(h_spec)

    # --- Phase 4: Nesterov accelerated gradient refinement ---
    if best_h is not None and time.time() < _DEADLINE - 35:
        h_cand, s_cand = _nesterov_smooth_max(best_h, beta=100.0, max_iter=400, lr_init=0.008)
        _update_best(h_cand)

    # --- Phase 5: Coordinate descent polish on multiple candidates ---
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_cd = _coordinate_descent_polish(best_h, max_iter=1500)
        _update_best(h_cd)

    # --- Phase 6: Final minimax + CD polish ---
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_final = _final_polish(best_h)
        _update_best(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
