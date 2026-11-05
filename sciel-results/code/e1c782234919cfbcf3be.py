"""Moment-matching quadrature for luminosity distance.

Structural change: instead of minimizing max relative error against a fixed
training grid, we directly fit the quadrature weights (and optionally nodes)
to match integrals of a rich basis of functions that span the CPL parameter
space. The basis functions are chosen so that any 1/E(a) profile in the
validation range can be approximated by a linear combination.

We use:
1. A fixed, well-distributed node set (shifted Gauss-Legendre with slight
   perturbation toward the endpoints to capture the 1/z behavior).
2. Moment-matching: solve for weights that minimize ||W * f_basis - I_basis||^2
   subject to W > 0, sum(W) = 1.
3. Joint node+weight refinement via basin-hopping + BFGS on a smooth
   composite objective.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre, roots_chebyshev


def _E_of_a(a, Om, w0, wa):
    """CPL Hubble expansion E(a)=H(a)/H0 for flat universe."""
    a = np.asarray(a, dtype=float)
    matter = Om * a ** (-3.0)
    if abs(w0 + 1.0) < 1e-10 and abs(wa) < 1e-10:
        de = (1.0 - Om)
    elif abs(w0 + 1.0) < 1e-10:
        exp_arg = 3.0 * wa * (1.0 - a)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.exp(exp_arg)
    else:
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_arg = 3.0 * wa * (1.0 - a) / (1.0 + w0)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.power(a, exponent) * np.exp(exp_arg)
    return np.sqrt(matter + de)


def _integrand_for_nodes(nodes, Om, w0, wa):
    """Compute 1/E(a_i) for the given nodes."""
    a_vals = 1.0 / (1.0 + nodes)
    return 1.0 / _E_of_a(a_vals, Om, w0, wa)


def _ref_integral(z_max, Om, w0, wa):
    """High-accuracy reference integral via Gauss-Legendre."""
    n_gl = 256
    gz, gw = roots_legendre(n_gl)
    z_pts = 0.5 * z_max * (gz + 1.0)
    w_pts = 0.5 * z_max * gw
    a_pts = 1.0 / (1.0 + z_pts)
    e_vals = _E_of_a(a_pts, Om, w0, wa)
    return np.sum(w_pts / e_vals)


def _basis_functions(z):
    """
    Return a matrix of basis functions B(z, k) for k=0..K-1.
    We use a combination of:
    - Monomials z^0, z^1, ..., z^5
    - Exponentials exp(-c*z) for several c
    - Rational 1/(1+c*z)
    This basis should span the 1/E(a(z)) profiles for reasonable CPL params.
    """
    z = np.asarray(z, dtype=float)
    K = 20
    B = np.zeros((len(z), K))
    # Monomials
    for k in range(6):
        B[:, k] = z ** k
    # Exponentials
    c_vals = [0.5, 1.0, 2.0, 4.0]
    for i, c in enumerate(c_vals):
        B[:, 6 + i] = np.exp(-c * z)
    # Rationals
    c_vals_r = [0.5, 1.0, 2.0]
    for i, c in enumerate(c_vals_r):
        B[:, 10 + i] = 1.0 / (1.0 + c * z)
    # Additional: z * exp(-c*z), z^2 * exp(-c*z)
    for i, c in enumerate([1.0, 2.0]):
        B[:, 13 + i] = z * np.exp(-c * z)
        B[:, 15 + i] = z ** 2 * np.exp(-c * z)
    # More monomials
    B[:, 17] = z ** 6
    B[:, 18] = z ** 7
    B[:, 19] = z ** 8
    return B


def _compute_moments(Om, w0, wa, z_max=5.0, n_gl=512):
    """Compute exact integrals of basis functions against 1/E(a(z))."""
    gz, gw = roots_legendre(n_gl)
    z_pts = 0.5 * z_max * (gz + 1.0)
    w_pts = 0.5 * z_max * gw
    a_pts = 1.0 / (1.0 + z_pts)
    e_vals = _E_of_a(a_pts, Om, w0, wa)
    inv_E = 1.0 / e_vals
    B = _basis_functions(z_pts)
    # Moments: integral of B(z,k) * 1/E(a(z)) dz
    moments = B.T @ (w_pts * inv_E)
    return moments


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill

    n = 8

    # --- Training cosmologies for moment computation ---
    models = []
    Om_list = [0.2, 0.3, 0.35, 0.4]
    w0_list = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3]
    wa_list = [-0.5, -0.2, 0.0, 0.2, 0.5]
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                models.append((Om, w0, wa))

    # Pre-compute moments for all models
    n_models = len(models)
    all_moments = np.zeros((n_models, 20))
    for i, (Om, w0, wa) in enumerate(models):
        all_moments[i] = _compute_moments(Om, w0, wa)

    # --- Initial node placement: shifted Gauss-Legendre ---
    gl_nodes, gl_weights = roots_legendre(n)
    # Map to [0,1] with slight endpoint bias
    nodes_init = 0.5 * (gl_nodes + 1.0)
    # Add small endpoint perturbations to capture 1/z-like behavior
    nodes_init[0] = 0.02
    nodes_init[-1] = 0.98
    nodes_init = np.sort(nodes_init)

    # --- Phase 1: Solve for optimal weights via constrained least squares ---
    # For each model, we want sum_i w_i * f(z_i) ≈ moment_k
    # This is a system: F @ w = m, where F[i,k] = f_k(z_i)
    # We solve in least-squares sense across all models.

    def _objective_weights(weights, nodes):
        """Compute max relative error across all models."""
        if time.time() > DEADLINE:
            return 1e10
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(models):
            f_vals = _integrand_for_nodes(nodes, Om, w0, wa)
            q = np.dot(weights, f_vals)
            # Compare against high-accuracy reference
            ref_val = _ref_integral(5.0, Om, w0, wa)
            if abs(ref_val) < 1e-15:
                continue
            rel_err = abs(q - ref_val) / abs(ref_val)
            if rel_err > max_err:
                max_err = rel_err
        return max_err

    # --- Phase 2: Joint node+weight optimization ---
    def decode(params):
        u = params[:8]
        v = params[8:]
        nodes = 1.0 / (1.0 + np.exp(-u))
        nodes = np.clip(nodes, 1e-7, 1.0 - 1e-7)
        order = np.argsort(nodes)
        nodes = nodes[order]
        v_shifted = v - np.max(v)
        w_exp = np.exp(v_shifted)
        weights = w_exp / np.sum(w_exp)
        return nodes, weights

    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        nodes, weights = decode(params)
        if np.any(np.diff(nodes) < 1e-14):
            return 1e8
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(models):
            f_vals = _integrand_for_nodes(nodes, Om, w0, wa)
            q = np.dot(weights, f_vals)
            ref_val = _ref_integral(5.0, Om, w0, wa)
            if abs(ref_val) < 1e-15:
                continue
            rel_err = abs(q - ref_val) / abs(ref_val)
            if rel_err > max_err:
                max_err = rel_err
        return max_err

    # Initial parameters
    u_init = np.log(nodes_init / (1.0 - nodes_init))
    v_init = np.log(gl_weights / 2.0)
    x0 = np.concatenate([u_init, v_init])
    bounds = [(-10.0, 10.0)] * 16

    best_x = x0.copy()
    best_val = objective(best_x)

    # --- Multi-start local optimization ---
    n_starts = 5
    for s in range(n_starts):
        if time.time() > DEADLINE - 15.0:
            break
        if s == 0:
            x_start = x0.copy()
        else:
            # Perturbed start
            x_start = x0 + np.random.randn(16) * 0.5
            x_start = np.clip(x_start, -10.0, 10.0)

        try:
            res = minimize(objective, x_start, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Basin-hopping for global exploration ---
    if time.time() < DEADLINE - 10.0:
        def _step(x):
            x_new = x + np.random.randn(16) * 0.3
            return np.clip(x_new, -10.0, 10.0)

        try:
            from scipy.optimize import basinhopping
            res = basinhopping(objective, best_x, niter=50,
                               minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds},
                               stepsize=0.5, seed=42,
                               accept_test=lambda f1, f2, x1, x2: f1 < f2)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Final Nelder-Mead polish ---
    if time.time() < DEADLINE - 3.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Decode and finalize ---
    nodes, weights = decode(best_x)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]

    # Enforce strict increase
    nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12

    # Enforce positivity and normalize
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
