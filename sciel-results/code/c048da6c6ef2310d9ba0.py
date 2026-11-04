"""Combined program for the third autocorrelation inequality task.
Uses a parameterized smooth family (Fourier cosine) with Nelder-Mead optimization,
plus a final L-BFGS-B polish on the full vector.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _value(h):
    """Compute the evaluator's value: abs(2*N*max(convolve(h,h))/sum(h)^2)."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0

    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        v = _value(h_candidate)
        if v < best_val:
            best_val = v
            best_h = h_candidate.copy()

    # Precompute Fourier basis for the parameterized family
    t = np.arange(N)
    num_harmonics = 8
    cos_basis = []
    for k in range(1, num_harmonics + 1):
        cos_basis.append(np.cos(2 * np.pi * k * t / N))
    cos_basis = np.array(cos_basis)  # shape (num_harmonics, N)

    # Parameterized family: h = 1 + sum(a_k * cos_k)
    # This keeps h smooth and close to 1, which is good for the ratio
    def make_h(params):
        # params[0] is the offset, params[1:] are cosine coefficients
        offset = params[0]
        coeffs = params[1:]
        h = np.ones(N) + offset
        for k in range(num_harmonics):
            h += coeffs[k] * cos_basis[k]
        return h

    def objective_param(params):
        h = make_h(params)
        return _value(h)

    # Try multiple initializations in the parameter space
    np.random.seed(42)
    inits_param = []

    # Small perturbations around zero
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for _ in range(3):
            p = np.zeros(num_harmonics + 1)
            p[1:] = np.random.randn(num_harmonics) * scale
            inits_param.append(p)

    # Structured initializations in parameter space
    # Only first harmonic
    p1 = np.zeros(num_harmonics + 1)
    p1[1] = 0.1
    inits_param.append(p1)
    p1[1] = -0.1
    inits_param.append(p1)

    # First two harmonics
    p2 = np.zeros(num_harmonics + 1)
    p2[1] = 0.1
    p2[2] = 0.05
    inits_param.append(p2)
    p2[1] = -0.1
    p2[2] = 0.05
    inits_param.append(p2)

    # Random with larger scale
    for _ in range(5):
        p = np.zeros(num_harmonics + 1)
        p[1:] = np.random.randn(num_harmonics) * 0.3
        inits_param.append(p)

    # Optimize each initialization with Nelder-Mead
    num_inits = len(inits_param)
    for i, p_init in enumerate(inits_param):
        if time.time() > deadline - 20.0:
            break

        # Update best with the initial point itself
        h_init = make_h(p_init)
        try_update(h_init)

        # Run Nelder-Mead
        try:
            max_iter = max(50, int((deadline - time.time() - 10.0) / max(1, num_inits - i) * 100))
            res = minimize(objective_param, p_init, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            if time.time() > deadline - 15.0:
                break
            h_res = make_h(res.x)
            try_update(h_res)

        except Exception:
            pass

    # Final polish with L-BFGS-B on the full vector, starting from best_h
    if time.time() < deadline - 10.0:
        def objective_full(x):
            return _value(x)

        try:
            res = minimize(objective_full, best_h, method='L-BFGS-B',
                           options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
            try_update(res.x)
        except Exception:
            pass

        # Another Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                res = minimize(objective_full, best_h, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
                try_update(res.x)
            except Exception:
                pass

    # Final safety check
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
