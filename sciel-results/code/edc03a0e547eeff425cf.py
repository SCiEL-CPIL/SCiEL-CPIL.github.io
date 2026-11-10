"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Gauss-Legendre 8-point nodes and weights on [-1, 1]
    t_gl = np.array([
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346425004484,
         0.1834346425004484,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    ])

    # Map to [0, 1]
    nodes_gl = 0.5 * t_gl + 0.5
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)

    # Best solution so far
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_obj = np.inf

    # CPL cosmology models: (w0, wa) pairs covering typical range
    cpl_models = [
        (0, 0),       # Lambda CDM
        (-1, 0),      # Cosmological constant
        (-1, 1),      # Quintessence-like
        (-0.5, 0),    # Mild dark energy
        (-1, -1),     # Phantom-ish
        (0.5, 0),     # Matter-dominated-ish
        (-1, 0.5),    # Evolving DE
        (-0.8, 0.2),  # Intermediate
        (-1, -0.5),   # Slightly phantom
        (-0.7, -0.3), # Mild evolution
        (-1, 1.5),    # Strong evolution
        (0, 1),       # Matter + evolving
        (-1.2, 0.3),  # Phantom with evolution
        (-0.9, -0.4), # Near -1 with negative evolution
    ]

    # Redshifts to test at
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    def _E(a, w0, wa):
        """H(a)/H0 for CPL model. a = 1/(1+z)"""
        Om_m = 0.3
        Om_La = 1.0 - Om_m
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        term_m = Om_m * a**(-3.0)
        term_de = Om_La * a**(exponent_de)
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))

    def _true_integral(w0, wa, z):
        """Compute true integral using dense quadrature"""
        u = np.linspace(0, 1, 10001)
        a = 1.0 / (1.0 + z * u)
        E_vals = _E(a, w0, wa)
        integrand = 1.0 / E_vals
        true_int = np.trapz(integrand, u)
        return (1.0 + z) * z * true_int

    def _quadrature_value(nodes, weights, w0, wa, z):
        """Compute quadrature approximation"""
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    def _objective(x):
        """Objective: minimize worst-case relative error"""
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total

        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)

        # Check constraints
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10

        # Compute worst-case relative error
        max_rel_err = 0.0
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_integral(w0, wa, z)
                quad_val = _quadrature_value(nodes, weights, w0, wa, z)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        return max_rel_err

    def _encode(nodes, weights):
        """Encode nodes and weights into parameter vector"""
        spacings = np.diff(nodes)
        log_spacings = np.log(spacings)
        log_weights = np.log(weights)
        return np.concatenate([log_spacings, log_weights])

    x0 = _encode(best_nodes, best_weights)

    # Try L-BFGS-B from GL initial point
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                # Decode result
                log_spacings = res.x[:8]
                spacings = np.exp(log_spacings)
                total = np.sum(spacings)
                best_nodes = np.cumsum(spacings) / total
                log_weights = res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                best_weights = exp_w / np.sum(exp_w)
        except Exception:
            pass

    # Final checks and normalization
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)

    # Ensure strict constraints
    if not np.all(np.diff(best_nodes) > 0) or not np.all(best_weights > 0) or abs(np.sum(best_weights) - 1.0) > 1e-12:
        # Fallback to midpoint if something went wrong
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0 / 8.0)

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
