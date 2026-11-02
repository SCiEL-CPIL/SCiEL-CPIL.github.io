"""Optimized program: numerical minimization of the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hermite_prob(n, x):
    """Probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_c4_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(g) * A(ĝ).
    
    Uses the analytical Fourier transform property:
    For the unitary Fourier transform F[f](ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx,
    F[Hn(x) e^{-x^2/2}](ξ) = (-i)^n Hn(ξ) e^{-ξ^2/2}
    
    Since n ∈ {0, 4, 8, 12} are all even, (-i)^n = 1 for all of them.
    Thus ĝ(ξ) = g(ξ), so A(ĝ) = A(g), and C4 bound = A(g)^2.
    
    However, we should verify this numerically to be safe.
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Compute g(x) on a fine grid
    x = np.linspace(-20, 20, 40001)
    h0 = _hermite_prob(0, x)
    h4 = _hermite_prob(4, x)
    h8 = _hermite_prob(8, x)
    h12 = _hermite_prob(12, x)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    g = poly * np.exp(-x ** 2 / 2.0)
    
    A_g = np.max(np.abs(g))
    
    # Analytical result: ĝ(ξ) = g(ξ) for even-degree Hermite combinations
    # This is because (-i)^n = 1 for n = 0, 4, 8, 12
    # So A_hat_g = A_g
    A_hat_g = A_g
    
    return A_g * A_hat_g


def construct():
    """
    Numerically optimize the three free Hermite coefficients (a0, a4, a8)
    to minimize the upper bound on C4 = A(g) * A(ĝ), with the constraint
    g(0) = 0 enforced by solving for a12.
    """
    def objective(coeffs):
        a0, a4, a8 = coeffs
        try:
            val = _compute_c4_bound(a0, a4, a8)
            if not np.isfinite(val):
                return 1e10
            return val
        except Exception:
            return 1e10
    
    # Multiple starting points for robustness
    start_points = [
        np.array([-1.13, 0.04, 0.000309]),  # near known good
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-0.5, 0.1, 0.001]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.5, 0.02, 0.0005]),
        np.array([-0.8, 0.08, 0.0002]),
        np.array([-1.2, 0.03, 0.0004]),
    ]
    
    best_val = np.inf
    best_x = start_points[0]
    
    # First pass: Nelder-Mead from multiple starts
    for sp in start_points:
        try:
            res = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second pass: refine with L-BFGS-B
    try:
        res = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Third pass: further refinement with Nelder-Mead from best
    try:
        res = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14,
                     'adaptive': True}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
