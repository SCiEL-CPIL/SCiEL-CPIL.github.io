"""Uncertainty bound minimization via sphere-constrained stochastic local search + analytic gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^{-x^2} dx for n=0..13 using recurrence."""
    I = np.zeros(14)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 14):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I_TABLE = _precompute_I()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    I = _I_TABLE

    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            if pj == 0.0:
                continue
            pp = pi * pj
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]

    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qj = q[j]
            if qj == 0.0:
                continue
            qq = qi * qj
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

    base_dp = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])  # shape (7, 3)

    a12_coeffs = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])  # dp_k/da12

    grad = np.zeros(3)

    # Precompute sums for efficiency
    s_p = np.zeros((3, 7))
    s_q = np.zeros((3, 7))
    for i in range(7):
        for k in range(3):
            sp = 0.0
            sq = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    sp += p[j] * I[idx]
                    sq += q[j] * I[idx]
            s_p[k, i] = sp
            s_q[k, i] = sq

    for c in range(3):
        dp = base_dp[:, c] + a12_coeffs * da12[c]
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * np.dot(dp, s_p[k])

        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 2.0 * 2.0 * np.pi * np.dot(dq, s_q[k])

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _time_remaining():
    return _DEADLINE - time.time()


def _sphere_search(initial_params, rng, max_iters=20000):
    """
    Stochastic local search on the unit sphere (after reparameterization).
    We work with c = params / ||params|| to normalize, but actually the problem
    is scale-invariant in a specific way. Let's think:
    
    If we scale all a_i by t, then P(x) scales by t, so |g|^2 scales by t^2.
    m_k scales by t^2, so vf = m2/m0 is invariant. Similarly for vF.
    So the bound is scale-invariant! This means we can fix ||a|| = 1.
    
    But wait - the constraint g(0)=0 involves a12 which depends linearly on a0,a4,a8.
    If we scale a by t, a12 also scales by t, so g(0) = t * (original g(0)) = 0.
    So the constraint is homogeneous. Good.
    
    So we can parameterize by direction on the sphere and any scale.
    Since the objective is scale-invariant, we just need to find the best direction.
    """
    # Start from initial_params, normalize to unit sphere
    a = initial_params.copy()
    norm = np.linalg.norm(a)
    if norm < 1e-15:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = a / norm
    
    val, _ = _bound_and_grad(a)
    if not np.isfinite(val) or val >= 1e17:
        # Try a different starting point
        a = np.array([-1.0, 0.05, 0.0003])
        norm = np.linalg.norm(a)
        a = a / norm
        val, _ = _bound_and_grad(a)
    
    best_val = val
    best_a = a.copy()
    
    step_size = 0.05
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iters):
        if _time_remaining() < 30:
            break
        
        # Generate random direction on sphere
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        
        # Try a step in direction d
        trial = a + step_size * d
        trial = trial / np.linalg.norm(trial)
        
        trial_val, _ = _bound_and_grad(trial)
        
        if np.isfinite(trial_val) and trial_val < val:
            a = trial
            val = trial_val
            no_improve = 0
            if val < best_val:
                best_val = val
                best_a = a.copy()
            # Occasionally increase step size
            if no_improve == 0 and it % 100 == 0:
                step_size = min(step_size * 1.1, 0.3)
        else:
            no_improve += 1
            # Decrease step size when stuck
            step_size = max(step_size * 0.95, 0.001)
        
        if no_improve > max_no_improve:
            # Reset to best and try a larger jump
            a = best_a.copy()
            val = best_val
            step_size = 0.2
            no_improve = 0
            
            # Make a big random jump
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            a = a + 0.5 * d
            a = a / np.linalg.norm(a)
            val, _ = _bound_and_grad(a)
            if not np.isfinite(val) or val >= 1e17:
                a = best_a.copy()
                val = best_val
    
    return best_a, best_val


def construct():
    rng = np.random.default_rng(42)
    
    # Initial diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.2, 0.05, 0.0006]),
    ]
    
    # Add random starts
    for i in range(10):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Stage 1: Sphere-based stochastic local search from multiple starts
    for start in starts:
        if _time_remaining() < 40:
            break
        try:
            sphere_a, sphere_val = _sphere_search(start, rng, max_iters=5000)
            if sphere_val < best_val and np.isfinite(sphere_val):
                best_val = sphere_val
                best_params = sphere_a.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Stage 2: L-BFGS-B refinement from the best sphere-search result
    if _time_remaining() > 30:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: More sphere search from the refined point
    if _time_remaining() > 20:
        try:
            sphere_a, sphere_val = _sphere_search(best_params, rng, max_iters=10000)
            if sphere_val < best_val and np.isfinite(sphere_val):
                best_val = sphere_val
                best_params = sphere_a.copy()
        except Exception:
            pass
    
    # Stage 4: Final L-BFGS-B polish
    if _time_remaining() > 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 5: Nelder-Mead final polish
    if _time_remaining() > 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-17,
                                   'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 6: Ultra-precise final L-BFGS-B
    if _time_remaining() > 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
