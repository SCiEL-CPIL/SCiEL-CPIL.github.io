"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: we have 90s wall-clock; use 85s for safety
_DEADLINE = time.time() + 85.0


def _time_left():
    return _DEADLINE - time.time()


def _E_a(a, w0, wa, w):
    """E(a) = H(a)/H0 for CPL dark energy in flat universe.
    a = 1/(1+z). 1+z = 1/a.
    H(z)^2/H0^2 = Om_m*(1+z)^3 + Om_r*(1+z)^4 + Om_k*(1+z)^0 + Om_de*(1+z)^{-3(1+w0+w*(1-1/(1+z)))}
    With Om_k = 1 - Om_m - Om_de.
    For simplicity, neglect radiation (Om_r=0) for z <~ 3000, which covers our integration range.
    """
    z = 1.0 / a - 1.0
    one_plus_z = 1.0 + z
    # CPL: w(z) = w0 + w * (1 - 1/(1+z)) = w0 + w * z/(1+z)
    # Integral of (1+w(z)) dz from 0 to z:
    # (1+w0+w)*z - w*ln(1+z)
    # So (1+z)^(3(1+w0+w) - 3w*ln(1+z))
    # Actually, the standard formula:
    # E^2 = Om_m*(1+z)^3 + Om_de * exp(3 * (1+w0+w) * z - 3*w * ln(1+z))
    # Wait, let me be precise.
    # H^2/H0^2 = Om_m (1+z)^3 + Om_de (1+z)^(3(1+w0+w)) * (1+z)^{-3w} ... no.
    # The standard CPL parametrization gives:
    # E(z)^2 = Om_m (1+z)^3 + Om_de (1+z)^{3(1+w0+w)} (1+z)^{-3w}
    # No, that's not right either. Let me recall.
    # w(z) = w0 + w * (1 - 1/(1+z))
    # The DE density evolves as: rho_de(z) = rho_de0 * exp(-3 * int_0^z (1+w(z')) dz')
    # int_0^z (1+w(z')) dz' = int_0^z (1+w0+w - w/(1+z')) dz' = (1+w0+w)*z - w*ln(1+z)
    # So rho_de(z)/rho_de0 = exp(-3*((1+w0+w)*z - w*ln(1+z))) = (1+z)^{3w} * exp(-3*(1+w0+w)*z)
    # Therefore E^2 = Om_m*(1+z)^3 + Om_de*(1+z)^3 * (rho_de(z)/rho_de0) ... wait.
    # E^2 = Om_m*(1+z)^3 + Om_r*(1+z)^4 + Om_k + Om_de * (1+z)^3 * (rho_de(z)/rho_de0)
    # Hmm, actually E^2 = H^2/H0^2 = Om_m*(1+z)^3 + Om_r*(1+z)^4 + Om_k + Om_de*(1+z)^3*(rho_de(z)/rho_de0)
    # No, that's not right. The Friedmann equation is:
    # H^2/H0^2 = sum_i Om_i * (a0/a)^{3(1+w_i)} for constant w_i
    # For CPL, it's:
    # H^2/H0^2 = Om_m*(1+z)^3 + Om_r*(1+z)^4 + Om_k + Om_de * (1+z)^{3(1+w0+w)} * (1+z)^{-3w}
    # Wait, I think the correct form is:
    # E^2(z) = Om_m (1+z)^3 + Om_de (1+z)^{3(1+w0+w)} (1+z)^{-3w}
    # Let me just use: E^2 = Om_m*(1+z)^3 + Om_de*(1+z)^3 * exp(-3*((1+w0+w)*z - w*log(1+z)))
    # which simplifies to Om_m*(1+z)^3 + Om_de*(1+z)^3*(1+z)^{3w}*exp(-3*(1+w0+w)*z)
    # = Om_m*(1+z)^3 + Om_de*(1+z)^{3(1+w0+w)}*exp(-3*(1+w0+w)*z) ... no.
    # Let me just compute it directly.
    integral = (1.0 + w0 + w) * z - w * np.log1p(z)
    de_factor = np.exp(-3.0 * integral)
    e2 = 0.315 * one_plus_z**3 + 0.685 * de_factor
    # Add a small Om_k if needed, but for flat universe Om_k=0
    # Actually, let's make it more general. Use Om_m=0.3, Om_de=0.7 for a representative model.
    # But we need the SAME quadrature to work across ALL models. So we should test on multiple models.
    return np.sqrt(e2)


def _lum_dist(z, nodes, weights, Om_m, Om_de, w0, wa):
    """Compute dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i) where a_i=1/(1+z*nodes[i])."""
    # a_i = 1/(1 + z * nodes[i])
    a_i = 1.0 / (1.0 + z * nodes)
    E_vals = _E_a(a_i, w0, wa, w0 + wa * (1 - a_i))
    # Wait, w(z) = w0 + wa * (1 - 1/(1+z)) = w0 + wa * z/(1+z)
    # At a_i, z_i = 1/a_i - 1 = z*nodes[i]
    # So w(z_i) = w0 + wa * (1 - a_i)
    # Let me recompute E properly.
    z_i = z * nodes
    one_plus_z_i = 1.0 + z_i
    integral_i = (1.0 + w0 + wa) * z_i - wa * np.log1p(z_i)
    de_factor_i = np.exp(-3.0 * integral_i)
    e2_i = Om_m * one_plus_z_i**3 + Om_de * de_factor_i
    E_vals = np.sqrt(e2_i)
    # The quadrature: sum_i weights[i] / E(a_i)
    quad = np.sum(weights / E_vals)
    return (1.0 + z) * z * quad


def _true_lum_dist(z, Om_m, Om_de, w0, wa, n_steps=10000):
    """High-accuracy reference using adaptive Simpson or fine trapezoidal."""
    # Use fine trapezoidal rule
    z_arr = np.linspace(0, z, n_steps + 1)
    a_arr = 1.0 / (1.0 + z_arr)
    z_arr_inner = z_arr  # for the integrand
    one_plus_z_arr = 1.0 + z_arr
    integral_arr = (1.0 + w0 + wa) * z_arr - wa * np.log1p(z_arr)
    de_factor_arr = np.exp(-3.0 * integral_arr)
    e2_arr = Om_m * one_plus_z_arr**3 + Om_de * de_factor_arr
    E_arr = np.sqrt(e2_arr)
    integrand = 1.0 / E_arr
    # Integrate from 0 to z: int_0^z dz'/E(z')
    # Use trapezoidal
    integral_val = np.trapz(integrand, z_arr)
    return (1.0 + z) * z * integral_val


def _objective(params):
    """Max relative error over a set of test cosmologies and redshifts."""
    # params: log-spaced nodes and log-spaced weights
    # nodes: 8 values in (0,1), strictly increasing
    # weights: 8 positive values summing to 1
    
    # Use softplus or exponential to ensure positivity and ordering
    # For nodes: use sigmoid of sorted parameters
    # For weights: use softmax
    
    node_params = params[:8]
    weight_params = params[8:16]
    
    # Nodes: use sigmoid to map to (0,1), then sort
    nodes_raw = 1.0 / (1.0 + np.exp(-node_params))
    nodes = np.sort(nodes_raw)
    
    # Weights: softmax
    weights = np.exp(weight_params - np.max(weight_params))
    weights = weights / np.sum(weights)
    
    # Test cosmologies
    test_cosmos = [
        (0.3, 0.7, 0.0, 0.0),    # LCDM
        (0.3, 0.7, -1.0, 0.0),   # Cosmological constant
        (0.3, 0.7, -1.0, 0.5),   # CPL with w0=-1, wa=0.5
        (0.3, 0.7, -0.5, -0.5),  # CPL with w0=-0.5, wa=-0.5
        (0.3, 0.7, -1.5, 1.0),   # CPL with w0=-1.5, wa=1.0
        (0.3, 0.7, -0.8, 0.3),   # CPL with w0=-0.8, wa=0.3
        (0.3, 0.7, -1.2, -0.3),  # CPL with w0=-1.2, wa=-0.3
        (0.3, 0.7, -0.9, 0.2),   # CPL with w0=-0.9, wa=0.2
    ]
    
    test_zs = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    max_rel_err = 0.0
    for Om_m, Om_de, w0, wa in test_cosmos:
        for z in test_zs:
            dl_approx = _lum_dist(z, nodes, weights, Om_m, Om_de, w0, wa)
            dl_true = _true_lum_dist(z, Om_m, Om_de, w0, wa, n_steps=5000)
            if dl_true > 1e-10:
                rel_err = abs(dl_approx - dl_true) / dl_true
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
    
    # Use log-sum-exp for smoothness
    return np.log(1.0 + max_rel_err * 100.0)  # Scale to make it more sensitive


def construct():
    """Construct an 8-node positive quadrature rule for luminosity distance."""
    start_time = time.time()
    
    # Initial guess: midpoint rule
    n = 8
    nodes_init = (np.arange(n) + 0.5) / n
    weights_init = np.full(n, 1.0 / n)
    
    # Convert to parameter space
    node_params_init = np.log(nodes_init / (1 - nodes_init))  # inverse sigmoid
    weight_params_init = np.log(weights_init)
    
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    if time.time() > _DEADLINE - 10:
        # Time's up, return best
        nodes = 1.0 / (1.0 + np.exp(-best_params[:8]))
        nodes = np.sort(nodes)
        weights = np.exp(best_params[8:16] - np.max(best_params[8:16]))
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}
    
    # Try multiple restarts
    np.random.seed(42)
    for restart in range(5):
        if time.time() > _DEADLINE - 15:
            break
        
        # Perturb initial guess
        noise = np.random.normal(0, 0.5, size=16)
        params_try = best_params + noise
        
        try:
            result = minimize(
                _objective,
                params_try,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 3000}
            )
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    nodes = 1.0 / (1.0 + np.exp(-best_params[:8]))
    nodes = np.sort(nodes)
    weights = np.exp(best_params[8:16] - np.max(best_params[8:16]))
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly in (0,1) and increasing
    # Add small epsilon to avoid boundary issues
    eps = 1e-8
    nodes = np.clip(nodes, eps, 1 - eps)
    # Ensure strict increase
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + eps
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
