"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combined strategy:
  - Nesterov accelerated gradient descent on a smooth-max surrogate (current program)
  - Epigraph formulation with auxiliary variable t (inspiration)
  - Spectral-space Nesterov (current program)
  - SciPy SLSQP refinement (current program)
  - Coordinate descent polish (current program)
  - High-beta projected-gradient polish (inspiration)
Deadline guard: 142 s wall-clock (safety margin under 150 s kill).
"""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 142


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)

    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp

    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)

    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Epigraph formulation: minimize t + (1/beta)*log(sum(exp(beta*(C_k - t))))
    Returns (val, grad_h, grad_t).
    """
    cs = _linear_overlap_values(h)
    diff = cs - t
    diff_max = np.max(diff)
    exp_diff = np.exp(beta * (diff - diff_max))
    sum_exp = np.sum(exp_diff)
    w = exp_diff / sum_exp

    val = t + (1.0 / beta) * (np.log(sum_exp) + beta * diff_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad_h = -(2.0 / N) * (A + B)
    grad_t = 1.0 - np.sum(w)

    return val, grad_h, grad_t


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with adaptive learning rate and backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t = t_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _nesterov_epigraph(h0, beta, n_iter=500, lr=0.03):
    """
    Nesterov accelerated gradient descent using epigraph formulation.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    t_val = _score(h)

    h_prev = h.copy()
    t_prev = t_val
    t_mom = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_mom_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_mom_new

        y_h = h + momentum * (h - h_prev)
        y_t = t_val + momentum * (t_val - t_prev)

        val_y, grad_h, grad_t = _epigraph_obj_grad(y_h, y_t, beta)

        step_lr = lr_current
        h_new = None
        t_new_val = None
        for bt in range(max_backtracks):
            h_candidate = y_h - step_lr * grad_h
            h_candidate = _project(h_candidate, N / 2.0)
            t_candidate = y_t - step_lr * grad_t
            t_candidate = max(0.0, t_candidate)

            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                t_new_val = t_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y_h - lr_current * 0.5 * grad_h
            h_new = _project(h_new, N / 2.0)
            t_new_val = max(0.0, y_t - lr_current * 0.5 * grad_t)
            lr_current *= 0.5

        h_prev = h
        t_prev = t_val
        h = h_new
        t_val = t_new_val
        t_mom = t_mom_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _spectral_to_h(coeffs):
    """Convert cosine coefficients (excluding DC) to h in [0,1] with sum N/2."""
    n = N
    h = np.zeros(n)
    for k in range(1, n):
        h += 2.0 * coeffs[k-1] * np.cos(2 * np.pi * k * np.arange(n) / n)
    # DC component is fixed to ensure sum = N/2
    h += 0.5
    return h


def _h_to_spectral(h):
    """Extract cosine coefficients (excluding DC) from h."""
    n = N
    h_centered = h - 0.5
    coeffs = np.zeros(n-1)
    for k in range(1, n):
        coeffs[k-1] = (2.0 / n) * np.sum(h_centered * np.cos(2 * np.pi * k * np.arange(n) / n))
    return coeffs


def _spectral_score_and_grad(coeffs, beta):
    """Score and gradient in spectral space."""
    h = _spectral_to_h(coeffs)
    h = _project(h, N / 2.0)
    val, grad_h = _smooth_max_obj_grad(h, beta)

    # Gradient in spectral space
    grad_coeffs = np.zeros(N-1)
    for k in range(1, N):
        grad_coeffs[k-1] = (2.0 / N) * np.sum(grad_h * np.cos(2 * np.pi * k * np.arange(N) / N))

    return val, grad_coeffs


def _nesterov_spectral(coeffs0, beta, n_iter=500, lr=0.03):
    """Nesterov in spectral space."""
    coeffs = np.array(coeffs0, dtype=np.float64)

    h_init = _spectral_to_h(coeffs)
    h_init = _project(h_init, N / 2.0)
    best_h = h_init.copy()
    best_score = _score(h_init)

    coeffs_prev = coeffs.copy()
    t = 1.0

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

        y = coeffs + momentum * (coeffs - coeffs_prev)

        val_y, grad = _spectral_score_and_grad(y, beta)

        step_lr = lr
        coeffs_new = None
        for bt in range(20):
            c_candidate = y - step_lr * grad
            h_candidate = _spectral_to_h(c_candidate)
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                coeffs_new = c_candidate
                break
            step_lr *= 0.5

        if coeffs_new is None:
            coeffs_new = y - lr * grad

        coeffs_prev = coeffs
        coeffs = coeffs_new
        t = t_new

        h_current = _spectral_to_h(coeffs)
        h_current = _project(h_current, N / 2.0)
        current_score = _score(h_current)
        if current_score < best_score:
            best_score = current_score
            best_h = h_current.copy()

    return best_h, best_score


def _coordinate_descent_polish(h0, max_outer=50, max_inner=N):
    """
    Coordinate descent polish: adjust one element at a time with
    local search around current value.
    """
    h = np.array(h0, dtype=np.float64).copy()
    best_score = _score(h)
    best_h = h.copy()

    best_step = np.full(N, 0.01)

    for outer in range(max_outer):
        if time.time() > _DEADLINE - 2:
            break
        improved = False
        for i in range(max_inner):
            if time.time() > _DEADLINE - 2:
                break
            old_val = h[i]
            step = best_step[i]

            deltas = [step, step/2, step/4, step/8, step/16, step/32]
            candidates = set()
            for d in deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            fine_deltas = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
            for d in fine_deltas:
                candidates.add(old_val - d)
                candidates.add(old_val + d)
            candidates.add(0.0)
            candidates.add(1.0)

            best_candidate_val = None
            best_candidate_score = best_score - 1e-12

            for new_val in sorted(candidates):
                if new_val < -1e-12 or new_val > 1 + 1e-12:
                    continue
                new_val = np.clip(new_val, 0.0, 1.0)
                h_test = h.copy()
                h_test[i] = new_val
                h_test = _project(h_test, N / 2.0)
                s_test = _score(h_test)
                if s_test < best_candidate_score:
                    best_candidate_score = s_test
                    best_candidate_val = new_val
                    move = abs(new_val - old_val)
                    if move > 0:
                        best_step[i] = max(0.00005, move * 0.5)

            if best_candidate_val is not None:
                h[i] = best_candidate_val
                h = _project(h, N / 2.0)
                best_score = best_candidate_score
                best_h = h.copy()
                improved = True

        if not improved:
            break

    return best_h, best_score


def _scipy_refine(h0, beta=20.0, max_iter=200):
    """Use scipy minimize with SLSQP for refinement."""
    h = np.array(h0, dtype=np.float64)

    def objective(x):
        hx = _project(x, N / 2.0)
        val, _ = _smooth_max_obj_grad(hx, beta)
        return val

    def gradient(x):
        hx = _project(x, N / 2.0)
        _, grad = _smooth_max_obj_grad(hx, beta)
        return grad

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N

    try:
        result = scipy_minimize(
            objective, h, method='SLSQP', jac=gradient,
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        h_opt = _project(result.x, N / 2.0)
        s_opt = _score(h_opt)
        return h_opt, s_opt
    except:
        return h0, _score(h0)


def _high_beta_polish(h0, n_iter=200, beta=300.0, lr=0.003):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    Uses plain projected gradient descent for stability at high beta.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_obj_grad(h, beta)

        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
        h = h_new

    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')

    np.random.seed(42)

    # Initializations: diverse starting points
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.sin(4 * np.pi * np.arange(N) / N),
        0.5 + 0.25 * np.cos(2 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.random.randn(N),
        0.5 + 0.2 * np.sin(6 * np.pi * np.arange(N) / N),
        0.5 + 0.15 * np.cos(3 * np.pi * np.arange(N) / N),
        # Spectral-structured inits
        0.5 + 0.25 * np.cos(2 * np.pi * 2 * np.arange(N) / N) + 0.1 * np.cos(2 * np.pi * 5 * np.arange(N) / N),
        0.5 + 0.2 * np.sin(2 * np.pi * 3 * np.arange(N) / N) + 0.15 * np.sin(2 * np.pi * 7 * np.arange(N) / N),
    ]

    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Add random inits
    for _ in range(8):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)

    # Phase 1: Nesterov smooth-max with adaptive beta
    beta = 10.0
    n_restarts = 5

    for r in range(n_restarts):
        if time.time() > _DEADLINE - 50:
            break

        for h_init in inits:
            if time.time() > _DEADLINE - 40:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=350, lr=0.05)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

        beta *= 1.4

        # Lower learning rate for finer convergence
        for h_init in inits[:3]:
            if time.time() > _DEADLINE - 40:
                break
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=250, lr=0.02)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: Epigraph Nesterov (inspiration's structural contribution)
    if best_h is not None and time.time() < _DEADLINE - 35:
        beta_epi = 15.0
        for r in range(4):
            if time.time() > _DEADLINE - 30:
                break
            for h_init in inits[:5]:
                if time.time() > _DEADLINE - 30:
                    break
                h_opt, s_opt = _nesterov_epigraph(h_init, beta_epi, n_iter=300, lr=0.03)
                if s_opt < best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            beta_epi *= 1.3

    # Phase 3: Spectral-space Nesterov
    if best_h is not None and time.time() < _DEADLINE - 25:
        coeffs_init = _h_to_spectral(best_h)
        beta_spec = 12.0
        for r in range(3):
            if time.time() > _DEADLINE - 20:
                break
            h_opt, s_opt = _nesterov_spectral(coeffs_init, beta_spec, n_iter=250, lr=0.03)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                coeffs_init = _h_to_spectral(best_h)
            beta_spec *= 1.3

    # Phase 4: Scipy SLSQP refinement
    if best_h is not None and time.time() < _DEADLINE - 18:
        for beta_s in [15.0, 25.0, 40.0]:
            if time.time() > _DEADLINE - 14:
                break
            h_opt, s_opt = _scipy_refine(best_h, beta=beta_s, max_iter=120)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 5: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_pol, s_pol = _coordinate_descent_polish(best_h, max_outer=40, max_inner=N)
        if s_pol < best_score:
            best_score = s_pol
            best_h = h_pol.copy()

    # Phase 6: Final fine-grained coordinate descent
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_pol2, s_pol2 = _coordinate_descent_polish(best_h, max_outer=25, max_inner=N)
        if s_pol2 < best_score:
            best_score = s_pol2
            best_h = h_pol2.copy()

    # Phase 7: High-beta projected-gradient polish (inspiration's refinement)
    if best_h is not None and time.time() < _DEADLINE - 3:
        h_pol3, s_pol3 = _high_beta_polish(best_h, n_iter=150, beta=300.0, lr=0.003)
        if s_pol3 < best_score:
            best_score = s_pol3
            best_h = h_pol3.copy()

    # Phase 8: Final scipy polish
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_opt, s_opt = _scipy_refine(best_h, beta=50.0, max_iter=80)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
