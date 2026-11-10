import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    """
    Construct an 8-node positive quadrature rule for luminosity distance.
    Uses adaptive optimization against a family of CPL dark-energy models.
    """
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Define a diverse set of CPL cosmologies (w0, wa, Om, Ol)
    # These represent the "validation" models the rule must work on
    cosmologies = [
        # (w0, wa, Om, Or)  -- Omega_L = 1 - Om - Or
        (-0.5, 0.0, 0.3, 1e-5),    # Lambda-like
        (-1.0, 0.0, 0.3, 1e-5),    # Cosmological constant
        (-0.5, 0.5, 0.3, 1e-5),    # CPL positive wa
        (-0.5, -0.5, 0.3, 1e-5),   # CPL negative wa
        (-1.5, 0.0, 0.3, 1e-5),    # Phantom-like
        (-1.0, 1.0, 0.3, 1e-5),    # Strong CPL
        (-1.0, -1.0, 0.3, 1e-5),   # Strong negative CPL
        (-0.5, 0.3, 0.25, 1e-5),   # Different Om
        (-1.2, 0.2, 0.35, 1e-5),   # Another combo
        (-0.8, 0.4, 0.3, 1e-5),    # Another combo
    ]
    
    def E_of_z(z, w0, wa, Om, Or):
        """H(z)/H0 for CPL model"""
        Ol = 1.0 - Om - Or
        # CPL: w(z) = w0 + wa * z / (1 + z)
        # E(z)^2 = Om*(1+z)^3 + Or*(1+z)^4 + Ol * (1+z)^(3*(1+w0+wa)) ... 
        # Actually the exact CPL integral form:
        # E(z) = [Om*(1+z)^3 + Or*(1+z)^4 + Ol * (1+z)^(3*(1+w0+wa))]^(1/2)
        # But the true CPL has a more complex form. Let's use the standard approximation:
        # For CPL, the dark energy term is:
        # rho_DE(z) / rho_DE(0) = (1+z)^(3*(1+w0+wa)) * (1+z)^(-3*wa/(1+z))
        # = (1+z)^(3*(1+w0)) * exp(-3*wa*log(1+z))
        # Simplified: use (1+z)^(3*(1+w0+wa)) as the power-law approximation
        # This is the standard "CPL power-law" approximation used in many codes
        
        # More accurate CPL:
        # E(z)^2 = Om*(1+z)^3 + Or*(1+z)^4 + Ol * (1+z)^(3*(1+w0+wa)) * (1+z)^(-3*wa/(1+z))
        # The last factor: (1+z)^(-3*wa/(1+z)) = exp(-3*wa*log(1+z)/(1+z))
        
        # Let's use the standard form:
        # E(z)^2 = Om*(1+z)^3 + Or*(1+z)^4 + Ol * (1+z)^(3*(1+w0+wa))
        # This is the "CPL" in the sense of Chevallier-Polarski-Linder but using the
        # power-law approximation for the DE density.
        
        # Actually, the exact CPL DE density evolution is:
        # rho_DE(z) = rho_DE(0) * (1+z)^(3*(1+w0+wa)) * exp(-3*wa * ln(1+z)/(1+z))
        # Let's use this more accurate form
        
        a = 1.0 / (1.0 + z)
        # DE term
        ln1pz = np.log(1.0 + z)
        de_factor = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * ln1pz / (1.0 + z))
        E2 = Om * (1.0 + z)**3 + Or * (1.0 + z)**4 + Ol * de_factor
        return np.sqrt(E2)
    
    def reference_integral(z, w0, wa, Om, Or):
        """Compute the integral I(z) = ∫_0^z du/E(u) using adaptive quadrature"""
        def integrand(u):
            return 1.0 / E_of_z(u, w0, wa, Om, Or)
        result, _ = quad(integrand, 0.0, z, limit=200, epsabs=1e-14, epsrel=1e-14)
        return result
    
    def quadrature_integral(z, nodes, weights, w0, wa, Om, Or):
        """Compute the quadrature approximation of I(z)"""
        # I(z) = z * sum_i weights[i] / E(z * nodes[i])
        # where nodes[i] in [0,1]
        a_vals = 1.0 / (1.0 + z * nodes)
        # E(a) = E(z) at redshift z*t for t=nodes[i]
        # Actually: E(z*t) where t = nodes[i]
        E_vals = np.array([E_of_z(z * t, w0, wa, Om, Or) for t in nodes])
        return z * np.sum(weights / E_vals)
    
    def worst_case_error(nodes, weights, z_max=2.0, n_z=20):
        """Compute worst-case relative error across all cosmologies and redshifts"""
        z_test = np.linspace(0.05, z_max, n_z)
        max_err = 0.0
        for (w0, wa, Om, Or) in cosmologies:
            for z in z_test:
                ref = reference_integral(z, w0, wa, Om, Or)
                if abs(ref) < 1e-15:
                    continue
                approx = quadrature_integral(z, nodes, weights, w0, wa, Om, Or)
                rel_err = abs(approx - ref) / abs(ref)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err
    
    # Parameterization to ensure constraints:
    # Nodes: use cumulative softplus to ensure strictly increasing in (0,1)
    # Weights: use softmax to ensure positive and sum to 1
    
    def params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights"""
        # params[0:8] -> nodes via cumulative softplus
        # params[8:16] -> weights via softmax
        
        # For nodes: start with small offset, add softplus of params
        base = 0.01
        offsets = np.maximum(0.05, np.exp(params[0:8]) - 1.0)  # positive offsets
        # Actually, let's use a cleaner parameterization
        # nodes[i] = sigmoid(cumsum of something)
        
        # Simpler: use the fact that we want 8 points in (0,1)
        # Let raw = params[0:8], then nodes = 1/(1+exp(-cumsum(softplus(raw))))
        # No, let's just use:
        # node_gaps = softplus(params[0:8])  # positive gaps
        # nodes = cumsum(node_gaps) / cumsum(node_gaps)[-1]  # normalize to [0,1]
        # But this doesn't guarantee strictly in (0,1) with margin
        
        # Better: 
        # Let x_i = params[0:8]
        # nodes = 0.05 + 0.9 * sigmoid(x) sorted? No, need to preserve order.
        
        # Use: gaps = 1 + softplus(params[0:8])  # gaps > 1
        # nodes_raw = cumsum(gaps)
        # nodes = nodes_raw / nodes_raw[-1]  # now in (0,1), last node = 1
        # That's bad, last node should be < 1
        
        # Let me use: 
        # t_i = sigmoid(params[0:8])  # in (0,1)
        # Sort them: t_sorted = np.sort(t)
        # nodes = t_sorted  # but this loses the "which param maps to which node"
        
        # Actually, the simplest robust approach:
        # Use 8 parameters, map to (epsilon, 1-epsilon) via sigmoid, then sort
        eps = 1e-6
        t = eps + (1 - 2*eps) / (1 + np.exp(-params[0:8]))
        nodes = np.sort(t)
        
        # Weights: softmax
        w_raw = params[8:16]
        w_max = np.max(w_raw)
        weights = np.exp(w_raw - w_max)
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    # Initial guess: midpoint rule
    n = 8
    nodes0 = (np.arange(n) + 0.5) / n
    weights0 = np.full(n, 1.0/n)
    
    # Invert the parameterization for initial guess
    # For nodes: x = log(t/(1-t)) for each t
    x0 = np.log(nodes0 / (1 - nodes0))
    # For weights: w_raw = log(weights) (up to constant, softmax is shift-invariant)
    w_raw0 = np.log(weights0)
    
    params0 = np.concatenate([x0, w_raw0])
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        nodes, weights = params_to_nodes_weights(params)
        # Check validity
        if not np.all(np.diff(nodes) > 1e-10):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10
        return worst_case_error(nodes, weights, z_max=1.5, n_z=12)
    
    # Optimize
    best_params = params0.copy()
    best_obj = objective(params0)
    
    # Try L-BFGS-B first (fast)
    try:
        if time.time() < deadline - 30:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead if time permits
    try:
        if time.time() < deadline - 20:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    try:
        if time.time() < deadline - 15:
            rng = np.random.default_rng(42)
            for _ in range(3):
                perturb = rng.normal(0, 0.5, size=16)
                p_try = best_params + perturb
                obj_try = objective(p_try)
                if obj_try < best_obj:
                    best_obj = obj_try
                    best_params = p_try
                    # Local refine
                    try:
                        result3 = minimize(objective, best_params, method='L-BFGS-B',
                                         options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-10})
                        if result3.fun < best_obj:
                            best_obj = result3.fun
                            best_params = result3.x
                    except Exception:
                        pass
    except Exception:
        pass
    
    nodes, weights = params_to_nodes_weights(best_params)
    
    # Final safety: ensure strict validity
    nodes = np.sort(nodes)
    # Ensure strictly in (0,1)
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    # Renormalize weights
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
