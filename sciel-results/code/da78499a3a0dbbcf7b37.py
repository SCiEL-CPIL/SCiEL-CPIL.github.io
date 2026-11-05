import numpy as np
from scipy.optimize import differential_evolution, minimize
from numpy.polynomial.legendre import leggauss
import time

def construct():
    t0 = time.time()
    deadline = 82.0  # safety margin before 90s kill

    # Diverse CPL models for training
    models = []
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.3, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for w0 in w0_vals:
        for wa in wa_vals:
            models.append((w0, wa))
    # Add extreme cases
    models.append((-2.0, 2.0))
    models.append((-2.0, -2.0))
    models.append((1.0, -1.0))
    models.append((1.0, 1.0))
    models.append((-1.0, -1.0))
    models.append((-1.0, 1.0))
    models.append((0.0, 0.0))

    Om_m = 0.3
    Om_de = 1.0 - Om_m

    def E_of_a(a, w0, wa):
        z = 1.0 / a - 1.0
        w = w0 + wa * z
        E2 = (1.0 + z)**3 + Om_de * (1.0 + z)**(3.0 * (1.0 + w))
        return np.sqrt(np.maximum(E2, 1e-30))

    def reference_integral(w0, wa, n_points=20000):
        z = np.linspace(0, 1, n_points)
        a = 1.0 / (1.0 + z)
        E_vals = E_of_a(a, w0, wa)
        f = (1.0 + z) * z / E_vals
        if (n_points - 1) % 2 != 0:
            n_points -= 1
            z = np.linspace(0, 1, n_points)
            a = 1.0 / (1.0 + z)
            E_vals = E_of_a(a, w0, wa)
            f = (1.0 + z) * z / E_vals
        h = 1.0 / (n_points - 1)
        result = h/3.0 * (f[0] + f[-1] + 4.0*np.sum(f[1:-1:2]) + 2.0*np.sum(f[2:-1:2]))
        return result

    # Precompute reference integrals
    refs = np.array([reference_integral(w0, wa) for (w0, wa) in models])
    n_models = len(models)

    def compute_errors(nodes, weights):
        """Compute relative errors for all models given nodes and weights."""
        a = 1.0 / (1.0 + nodes)
        errors = np.zeros(n_models)
        for i, (w0, wa) in enumerate(models):
            E_vals = E_of_a(a, w0, wa)
            q_val = np.sum(weights * (1.0 + nodes) * nodes / E_vals)
            ref_val = refs[i]
            if ref_val > 1e-10:
                errors[i] = abs(q_val - ref_val) / abs(ref_val)
        return errors

    def objective16(params):
        """Minimize max relative error. params: 8 node params + 8 weight params."""
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: use base grid + tanh perturbation, then enforce ordering
        base_nodes = np.linspace(0.1, 0.9, 8)
        nodes = base_nodes + 0.15 * np.tanh(node_params)
        nodes = np.clip(nodes, 0.001, 0.999)
        # Enforce strict ordering
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.001
        nodes = np.clip(nodes, 0.001, 0.999)
        
        # Weights: softmax to ensure positive, then normalize
        w_raw = np.exp(weight_params - np.max(weight_params))
        weights = w_raw / np.sum(w_raw)
        
        errors = compute_errors(nodes, weights)
        return np.max(errors)

    # Bounds for DE
    lb_nodes = np.full(8, -5.0)
    ub_nodes = np.full(8, 5.0)
    lb_weights = np.full(8, -5.0)
    ub_weights = np.full(8, 5.0)
    bounds = list(zip(np.concatenate([lb_nodes, lb_weights]),
                      np.concatenate([ub_nodes, ub_weights])))

    # Initial population: Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, gl_weights = leggauss(8)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    
    base_nodes = np.linspace(0.1, 0.9, 8)
    # Invert the transformation approximately for GL nodes
    x0_nodes = np.arctanh(np.clip((gl_nodes - base_nodes) / 0.15, -0.999, 0.999))
    x0_weights = np.log(gl_weights) - np.mean(np.log(gl_weights))
    x0 = np.concatenate([x0_nodes, x0_weights])

    best_err = objective16(x0)
    best_params = x0.copy()

    # Run differential evolution
    if time.time() - t0 < deadline - 25:
        try:
            de_result = differential_evolution(
                objective16, bounds, seed=42, maxiter=80, popsize=25,
                tol=1e-10, polish=True, init='sobol', updating='immediate'
            )
            if de_result.fun < best_err:
                best_err = de_result.fun
                best_params = de_result.x
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() - t0 < deadline - 5:
        try:
            nm_result = minimize(
                objective16, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_result.fun < best_err:
                best_err = nm_result.fun
                best_params = nm_result.x
        except Exception:
            pass

    # Second pass: try Powell from best point
    if time.time() - t0 < deadline - 3:
        try:
            pw_result = minimize(
                objective16, best_params, method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if pw_result.fun < best_err:
                best_err = pw_result.fun
                best_params = pw_result.x
        except Exception:
            pass

    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    base_nodes = np.linspace(0.1, 0.9, 8)
    nodes = base_nodes + 0.15 * np.tanh(node_params)
    nodes = np.clip(nodes, 0.001, 0.999)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.001
    nodes = np.clip(nodes, 0.001, 0.999)
    
    w_raw = np.exp(weight_params - np.max(weight_params))
    weights = w_raw / np.sum(w_raw)
    
    # Final normalization
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
