"""Spectral quadrature for luminosity distance using variable-order Gauss-Chebyshev
with adaptive weight optimization and a diverse training set."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre


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


def _integrand(nodes, Om, w0, wa):
    """Compute 1/E(a(z)) at given nodes."""
    a_vals = 1.0 / (1.0 + nodes)
    return 1.0 / _E_of_a(a_vals, Om, w0, wa)


def _lagrange_weights(nodes):
    """Compute Lagrange interpolation weights for given nodes."""
    n = len(nodes)
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] /= (nodes[i] - nodes[j])
    return weights


def _ref_integral(z_max, Om, w0, wa, n_gl=256):
    """High-accuracy reference integral via Gauss-Legendre."""
    gz, gw = roots_legendre(n_gl)
    z_pts = 0.5 * z_max * (gz + 1.0)
    w_pts = 0.5 * z_max * gw
    a_pts = 1.0 / (1.0 + z_pts)
    e_vals = _E_of_a(a_pts, Om, w0, wa)
    return np.sum(w_pts / e_vals)


def _chebyshev_nodes_order(n, order=1):
    """Gauss-Chebyshev nodes of specified order mapped to [0,1]."""
    if order == 1:
        # First kind: cos((2k+1)pi/(2n))
        k = np.arange(n)
        nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    else:
        # Second kind: cos(k*pi/(n-1)) for k=1..n-2
        k = np.arange(1, n - 1)
        nodes = np.cos(k * np.pi / (n - 1))
    nodes = 0.5 * (1.0 - nodes)  # Map to [0,1]
    return np.sort(nodes)


def _gauss_legendre_nodes(n):
    """Gauss-Legendre nodes mapped to [0,1]."""
    gl_nodes, gl_weights = roots_legendre(n)
    return 0.5 * (gl_nodes + 1.0), 0.5 * gl_weights


def _objective(params, models, z_list, ref, n_nodes=8):
    """Objective: worst-case relative error over training set."""
    if time.time() > _DEADLINE:
        return 1e10
    
    # Decode parameters: first n_nodes are node positions, rest are weight offsets
    node_params = params[:n_nodes]
    weight_params = params[n_nodes:]
    
    # Nodes: use direct values (already in [0,1])
    nodes = node_params.copy()
    nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
    nodes = np.sort(nodes)
    
    # Weights: softmax over weight_params
    w_shifted = weight_params - np.max(weight_params)
    w_exp = np.exp(w_shifted)
    weights = w_exp / np.sum(w_exp)
    
    # Check strict increase
    if np.any(np.diff(nodes) < 1e-14):
        return 1e8
    
    max_err = 0.0
    for i, (Om, w0, wa) in enumerate(models):
        f_vals = _integrand(nodes, Om, w0, wa)
        q = np.dot(weights, f_vals)
        for j, zm in enumerate(z_list):
            rv = ref[i, j]
            if abs(rv) < 1e-15:
                continue
            rel_err = abs(q - rv) / abs(rv)
            if rel_err > max_err:
                max_err = rel_err
    return max_err


_DEADLINE = 0.0


def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 82.0  # Safety margin under 90s kill

    # --- Training cosmologies ---
    models = []
    Om_list = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    w0_list = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5]
    wa_list = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                models.append((Om, w0, wa))

    z_list = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Pre-compute reference integrals
    n_models = len(models)
    n_z = len(z_list)
    ref = np.zeros((n_models, n_z))
    for i, (Om, w0, wa) in enumerate(models):
        for j, zm in enumerate(z_list):
            ref[i, j] = _ref_integral(zm, Om, w0, wa)

    n_nodes = 8

    # --- Multiple initializations ---
    best_val = np.inf
    best_x = None

    # Init 1: Gauss-Legendre
    gl_nodes, gl_weights = _gauss_legendre_nodes(n_nodes)
    u_init = gl_nodes.copy()
    v_init = np.log(gl_weights + 1e-15)
    x_init = np.concatenate([u_init, v_init])
    val = _objective(x_init, models, z_list, ref, n_nodes)
    if val < best_val:
        best_val = val
        best_x = x_init.copy()

    # Init 2: Chebyshev first kind
    ch_nodes = _chebyshev_nodes_order(n_nodes, 1)
    ch_weights = _lagrange_weights(ch_nodes)
    ch_weights = np.maximum(ch_weights, 1e-15)
    ch_weights = ch_weights / np.sum(ch_weights)
    u_init = ch_nodes.copy()
    v_init = np.log(ch_weights)
    x_init = np.concatenate([u_init, v_init])
    val = _objective(x_init, models, z_list, ref, n_nodes)
    if val < best_val:
        best_val = val
        best_x = x_init.copy()

    # Init 3: Chebyshev second kind
    ch2_nodes = _chebyshev_nodes_order(n_nodes, 2)
    ch2_weights = _lagrange_weights(ch2_nodes)
    ch2_weights = np.maximum(ch2_weights, 1e-15)
    ch2_weights = ch2_weights / np.sum(ch2_weights)
    u_init = ch2_nodes.copy()
    v_init = np.log(ch2_weights)
    x_init = np.concatenate([u_init, v_init])
    val = _objective(x_init, models, z_list, ref, n_nodes)
    if val < best_val:
        best_val = val
        best_x = x_init.copy()

    # Init 4: Equally spaced with optimized weights
    eq_nodes = np.linspace(0.05, 0.95, n_nodes)
    eq_weights = np.ones(n_nodes) / n_nodes
    u_init = eq_nodes.copy()
    v_init = np.log(eq_weights)
    x_init = np.concatenate([u_init, v_init])
    val = _objective(x_init, models, z_list, ref, n_nodes)
    if val < best_val:
        best_val = val
        best_x = x_init.copy()

    # --- Phase 1: Differential Evolution ---
    if time.time() < _DEADLINE - 35.0:
        bounds = [(1e-8, 1.0 - 1e-8)] * n_nodes + [(-10.0, 10.0)] * n_nodes
        try:
            de_res = differential_evolution(
                _objective, bounds, args=(models, z_list, ref, n_nodes),
                maxiter=80, popsize=10, tol=1e-8, seed=12345,
                polish=False, init='latinhypercube',
                callback=lambda xk, conv: (time.time() > _DEADLINE - 35.0)
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

    # --- Phase 2: L-BFGS-B refinement from multiple starting points ---
    for start_x in [best_x]:
        if time.time() < _DEADLINE - 15.0:
            try:
                res = minimize(_objective, start_x, method='L-BFGS-B',
                               args=(models, z_list, ref, n_nodes),
                               bounds=[(1e-8, 1.0 - 1e-8)] * n_nodes + [(-20.0, 20.0)] * n_nodes,
                               options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # --- Phase 3: Nelder-Mead final polish ---
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           args=(models, z_list, ref, n_nodes),
                           options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Decode and finalize ---
    nodes = best_x[:n_nodes].copy()
    weight_params = best_x[n_nodes:]
    nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
    nodes = np.sort(nodes)
    
    w_shifted = weight_params - np.max(weight_params)
    w_exp = np.exp(w_shifted)
    weights = w_exp / np.sum(w_exp)
    
    # Reorder weights to match sorted nodes
    orig_order = np.argsort(best_x[:n_nodes])
    # Actually, we need to track which weight goes with which node
    # Since we sorted nodes, we need to permute weights accordingly
    node_order = np.argsort(best_x[:n_nodes])
    weights_reordered = np.zeros(n_nodes)
    for new_idx, old_idx in enumerate(node_order):
        weights_reordered[new_idx] = weights[old_idx]
    weights = weights_reordered

    # Enforce strict increase
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12

    # Enforce positivity and normalize
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
