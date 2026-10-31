"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_MARGIN = 15.0
TARGET_SUM = N / 2.0
KILL_TIME = 150.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-15:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        mask = (h > 1e-13) & (h < 1.0 - 1e-13)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Compute soft max of x and its gradient using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-14:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum == 0 or not np.isfinite(wsum):
        return max_x, np.zeros_like(x)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _overlap_and_grad(h, beta):
    """Compute smooth max of overlap and gradient w.r.t h."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1]
    else:
        grad_h[0] += grad_overlap[0]
    return val_smooth, grad_h, overlap


def _objective_and_grad(h, beta, penalty_weight=50.0):
    """Objective with sum constraint penalty."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    val_smooth, grad_h, _ = _overlap_and_grad(h, beta)
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    return total_val, total_grad


def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0, A3=0.0, phi3=0.0):
    """Generate step heights from sinusoidal model."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    if abs(A3) > 1e-12:
        h += A3 * np.sin(6.0 * np.pi * i / N + phi3)
    return h


def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal parameters via grid search + local refinement."""
    best_score = float('inf')
    best_h = None
    best_params = None

    # Phase 1: Coarse grid search
    A_grid = np.linspace(0.05, 0.45, 30)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 60)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        if time.time() > deadline - 30.0:
            break
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 6 params
    def obj_sin(params):
        A, phi, A2, phi2, A3, phi3 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        A3 = np.clip(A3, -0.2, 0.2)
        h = _sinusoidal_h(A, phi, A2, phi2, A3, phi3)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0, 0.0, 0.0])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Phase 3: Multiple starting points
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        if time.time() > deadline - 20.0:
            break
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0, 0.0, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_params = res.x
                except Exception:
                    pass

    if best_params is not None:
        h_sin = _sinusoidal_h(*best_params)
    else:
        h_sin = _sinusoidal_h(best_A, best_phi)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """Optimize h using L-BFGS-B with analytic gradients and adaptive beta."""
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)

    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]

    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break

        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val

        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad

        try:
            res = minimize(
                obj_func,
                h,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass

    return current_score, h


def _local_search_polish(h, max_iters=300, deadline=None):
    """Deterministic local search to polish the solution."""
    h = h.copy()
    current_score = _score(h)

    if deadline is None:
        deadline = time.time() + 5.0

    deltas = [0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02]

    for _ in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        for i in range(N):
            for delta in deltas:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-13:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try pairwise swaps
            for i in range(N - 1):
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-13:
                    h = h_trial
                    current_score = s
                    improved = True
                    break

        if not improved:
            break

    return current_score, h


def _differential_evolution_search(deadline):
    """Global search using differential evolution on a reduced parameterization."""
    # Use a low-dimensional parameterization: mean + Fourier coefficients
    # h_i = 0.5 + a1*sin(2pi*i/N + p1) + a2*sin(4pi*i/N + p2) + a3*sin(6pi*i/N + p3)
    # Params: [a1, p1, a2, p2, a3, p3]

    def de_objective(params):
        a1, p1, a2, p2, a3, p3 = params
        a1 = np.clip(a1, 0.0, 0.5)
        a2 = np.clip(a2, -0.3, 0.3)
        a3 = np.clip(a3, -0.2, 0.2)
        h = _sinusoidal_h(a1, p1, a2, p2, a3, p3)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    bounds = [(0.0, 0.5), (0.0, 2*np.pi), (-0.3, 0.3), (0.0, 2*np.pi), (-0.2, 0.2), (0.0, 2*np.pi)]

    try:
        res = differential_evolution(
            de_objective, bounds,
            maxiter=200, popsize=15,
            tol=1e-10, seed=42,
            polish=True,
            init='sobol'
        )
        if res.success and res.fun < float('inf'):
            h = _sinusoidal_h(*res.x)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            return s, h
    except Exception:
        pass

    return float('inf'), None


def construct():
    start_time = time.time()
    deadline = start_time + (KILL_TIME - DEADLINE_MARGIN)

    best_score = float('inf')
    best_h = None

    # Phase 1: Sinusoidal optimization
    if time.time() < deadline - 30.0:
        s_sin, h_sin = _optimize_sinusoidal(deadline - 30.0)
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # Phase 2: Differential evolution global search
    if time.time() < deadline - 40.0:
        s_de, h_de = _differential_evolution_search(deadline - 40.0)
        if s_de < best_score:
            best_score = s_de
            best_h = h_de.copy()

    # Phase 3: Direct optimization with analytic gradients
    if best_h is not None and time.time() < deadline - 25.0:
        s_direct, h_direct = _optimize_direct(best_h, deadline - 25.0)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 4: Local search polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=300, deadline=deadline - 10.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 5: Final high-precision optimization
    if best_h is not None and time.time() < deadline - 5.0:
        s_final, h_final = _optimize_direct(best_h, deadline - 5.0, beta_start=100.0, beta_end=1000.0, max_iter=300)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Phase 6: Final polish
    if best_h is not None and time.time() < deadline - 2.0:
        s_final2, h_final2 = _local_search_polish(best_h, max_iters=200, deadline=deadline - 2.0)
        if s_final2 < best_score:
            best_score = s_final2
            best_h = h_final2.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
