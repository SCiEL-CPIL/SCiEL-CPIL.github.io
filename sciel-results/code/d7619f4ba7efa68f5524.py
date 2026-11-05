"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    # 8-point Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)

    # Map to [0, 1]
    nodes = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0

    # Ensure nodes are strictly increasing
    nodes = np.sort(nodes)
    # Re-sort weights accordingly
    order = np.argsort(gl_nodes)
    weights = gl_weights[order] / 2.0

    # Normalize weights to sum to exactly 1
    weights = weights / weights.sum()

    # Define cosmological models for optimization
    # E(a) = H(a)/H0 for different dark energy models
    def E_LCDM(a):
        """Lambda-CDM: Omega_m=0.3, Omega_L=0.7"""
        return np.sqrt(0.3 / a**3 + 0.7)

    def E_w0w3(a, w0, w3):
        """CPL parameterization with constant Omega_m=0.3"""
        Omega_m = 0.3
        Omega_L = 1.0 - Omega_m
        w = w0 + w3 * (1.0 - a)
        return np.sqrt(Omega_m / a**3 + Omega_L * a**(-3 * (1.0 + w)))

    def E_power(a, alpha):
        """Power-law dark energy: Omega_m=0.3, w=-1+alpha/3"""
        Omega_m = 0.3
        Omega_L = 1.0 - Omega_m
        w = -1.0 + alpha / 3.0
        return np.sqrt(Omega_m / a**3 + Omega_L * a**(-3 * (1.0 + w)))

    # Integrand for luminosity distance: f(z) = (1+z)*z / E(a) where a = 1/(1+z)
    # We're integrating over x in [0,1] where z = x / (1-x) is NOT what we want
    # Actually, the formula is: dL/(c/H0) = (1+z) * z * integral_0^z dz'/E(a')
    # But the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So the integral is over some variable x in [0,1] and z maps to x somehow.
    # Reading more carefully: the quadrature approximates integral_0^1 f(x) dx
    # where f(x) = 1/E(a(x)) and a(x) = 1/(1+z*x)
    # So we're computing (1+z)*z * integral_0^1 dx / E(1/(1+z*x))

    def reference_integral(z, E_func):
        """Compute reference integral using adaptive quadrature"""
        from scipy.integrate import quad
        def integrand(x):
            a = 1.0 / (1.0 + z * x)
            return 1.0 / E_func(a)
        result, _ = quad(integrand, 0.0, 1.0, limit=200)
        return result

    def quadrature_error(z, E_func):
        """Compute relative error of quadrature vs reference"""
        a_vals = 1.0 / (1.0 + z * nodes)
        quad_val = np.sum(weights / E_func(a_vals))
        ref_val = reference_integral(z, E_func)
        if ref_val == 0:
            return 0.0
        return abs(quad_val - ref_val) / abs(ref_val)

    # Define test cosmologies and redshifts
    test_models = [
        ("LCDM", E_LCDM),
        ("w0w3_(-0.5,0.5)", lambda a: E_w0w3(a, -0.5, 0.5)),
        ("w0w3_(-1.0,0.0)", lambda a: E_w0w3(a, -1.0, 0.0)),
        ("w0w3_(-0.8,0.3)", lambda a: E_w0w3(a, -0.8, 0.3)),
        ("power_0.2", lambda a: E_power(a, 0.2)),
        ("power_0.5", lambda a: E_power(a, 0.5)),
        ("w0w3_(-1.2,0.8)", lambda a: E_w0w3(a, -1.2, 0.8)),
        ("w0w3_(-0.3,0.2)", lambda a: E_w0w3(a, -0.3, 0.2)),
    ]

    test_zs = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Compute current worst-case error
    def compute_worst_error(nodes_arr, weights_arr):
        worst = 0.0
        for name, E_func in test_models:
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = reference_integral(z, E_func)
                if ref_val != 0:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    if err > worst:
                        worst = err
        return worst

    # Try to optimize using a smooth surrogate (soft-max) of the worst-case error
    # We'll use a log-sum-exp based soft-max to smooth the min-max objective

    def soft_max_error(nodes_arr, weights_arr, beta=50.0):
        """Smooth approximation of worst-case relative error"""
        errors = []
        for name, E_func in test_models:
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = reference_integral(z, E_func)
                if ref_val != 0:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    errors.append(err)
        errors = np.array(errors)
        # Soft-max using log-sum-exp
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        shifted = errors / max_err - 1.0
        # Use a smooth max approximation
        return max_err * np.log(np.sum(np.exp(beta * (errors / max_err - 1.0)))) / beta

    # Parameterize nodes to ensure they're in (0,1) and strictly increasing
    # Use a transformation: nodes_i = sigmoid(cumsum(params)) or similar
    # Simpler: optimize the 8 nodes directly with bounds, and weights with positivity constraint

    # Start from Gauss-Legendre as initial guess
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_score = compute_worst_error(best_nodes, best_weights)

    # Try Nelder-Mead optimization on the soft-max error
    def objective(params):
        # params: 8 node positions + 8 weight positions (before normalization)
        n_nodes = params[:8]
        n_weights = params[8:16]

        # Ensure nodes are in (0,1) and increasing
        # Use a cumulative sum transformation
        # Actually, let's just use the raw values with bounds
        if np.any(n_nodes <= 0) or np.any(n_nodes >= 1):
            return 1e10
        if not np.all(np.diff(n_nodes) > 1e-6):
            return 1e10

        # Normalize weights to be positive and sum to 1
        if np.any(n_weights <= 0):
            return 1e10
        w_norm = n_weights / np.sum(n_weights)

        # Compute soft-max error
        return soft_max_error(n_nodes, w_norm, beta=50.0)

    # Initial parameters
    x0 = np.concatenate([best_nodes, best_weights])

    # Run Nelder-Mead
    try:
        result = optimize.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < best_score and time.time() < deadline:
            opt_nodes = result.x[:8]
            opt_weights = result.x[8:]
            if np.all(opt_nodes > 0) and np.all(opt_nodes < 1) and np.all(np.diff(opt_nodes) > 1e-6):
                if np.all(opt_weights > 0):
                    opt_weights = opt_weights / np.sum(opt_weights)
                    best_nodes = opt_nodes
                    best_weights = opt_weights
                    best_score = compute_worst_error(best_nodes, best_weights)
    except Exception:
        pass

    # Final polish: try a few random perturbations around the best solution
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            pert_nodes = best_nodes + np.random.normal(0, 0.01, size=8)
            pert_weights = best_weights + np.random.normal(0, 0.01, size=8)

            # Clip and normalize
            pert_nodes = np.clip(pert_nodes, 1e-4, 1 - 1e-4)
            pert_nodes = np.sort(pert_nodes)
            if not np.all(np.diff(pert_nodes) > 1e-6):
                continue

            pert_weights = np.clip(pert_weights, 1e-6, None)
            pert_weights = pert_weights / np.sum(pert_weights)

            score = compute_worst_error(pert_nodes, pert_weights)
            if score < best_score:
                best_nodes = pert_nodes
                best_weights = pert_weights
                best_score = score

    # Ensure final validity
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
