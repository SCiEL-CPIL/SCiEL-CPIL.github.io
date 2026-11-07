"""Structural sum-of-squares parameterization for the Uncertainty Inequality task."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, minimize_scalar

def _compute_value(c0, c1, c2):
    """Compute the objective value for given coefficients."""
    try:
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1344.0
        H12_0 = 209088.0

        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

        # Build q(x) = g(x) / x^2 as a degree-10 polynomial (even, so only even powers)
        # q(x) = c3*x^10 - 84*c3*x^8 + (1680*c3 + c2)*x^6 - (16800*c3 + 28*c2)*x^4
        #        + (60480*c3 + 280*c2 + c1)*x^2 - (30240*c3 + 840*c2 + 6*c1)
        
        x0_coeff = -30240*c3 - 840*c2 - 6*c1
        if abs(x0_coeff) > 1e-10:
            return float('inf')

        q_coeffs = np.array([
            c3,
            0,
            -84*c3,
            0,
            1680*c3 + c2,
            0,
            -16800*c3 - 28*c2,
            0,
            60480*c3 + 280*c2 + c1,
            0,
            -30240*c3 - 840*c2 - 6*c1
        ])

        # Find roots of q(x)
        roots = np.roots(q_coeffs)

        # Find real roots
        real_roots = roots[np.abs(roots.imag) < 1e-10].real

        # For each real root, check if q changes sign (odd multiplicity)
        best_r = None
        for r in real_roots:
            eps = 1e-8
            q_left = np.polyval(q_coeffs, r - eps)
            q_right = np.polyval(q_coeffs, r + eps)
            if q_left * q_right < 0:
                if best_r is None or r > best_r:
                    best_r = r

        if best_r is None:
            return float('inf')

        value = best_r**2 / (2 * np.pi)
        return value

    except Exception:
        return float('inf')

def _objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return _compute_value(c0, c1, c2)

def construct():
    """Construct three coefficients that minimize the objective value."""
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit

    # Seed values
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_params = seed.copy()
    best_value = _compute_value(*seed)

    # Strategy 1: Differential Evolution (global optimization)
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            result = differential_evolution(
                _objective, bounds, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
                seed=42, polish=True
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 2: Nelder-Mead local refinement
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 3: BFGS local refinement
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective, best_params, method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-12}
            )
            if result.fun < best_value:
                best_params = result.x
                best_value = result.fun
        except Exception:
            pass

    # Strategy 4: Fine local search around best solution
    if time.time() < deadline:
        base_value = best_value
        for scale in [1e-4, 1e-5, 1e-6, 1e-7]:
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                perturbation = np.random.normal(0, scale, size=3)
                candidate = best_params + perturbation
                val = _compute_value(*candidate)
                if val < best_value:
                    best_params = candidate
                    best_value = val

    return np.array(best_params)
