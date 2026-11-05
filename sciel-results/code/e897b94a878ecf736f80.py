"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _reference_value_gl(z, Om, w0, wa, n_gl=100):
    """
    High-accuracy reference using Gauss-Legendre quadrature.
    The integral is I = int_0^z z'/(1+z') * 1/E(a(z')) dz'
    Wait, the formula in the task is:
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i]).
    This corresponds to the integral:
    I = int_0^1 (z * t) / E(1/(1+z*t)) dt
    Let u = z*t, dt = du/z. Limits 0 to z.
    I = int_0^z (u / E(1/(1+u))) (du/z) * z ? No.
    
    Let's re-derive the mapping.
    Task: dL = (c/H0) * (1+z) * int_0^z dz' / E(z')
    Let z' = z * t, where t in [0,1].
    dz' = z dt.
    dL = (c/H0) * (1+z) * int_0^1 (z dt) / E(z*t)
    dL = (c/H0) * (1+z) * z * int_0^1 1/E(z*t) dt
    
    The task formula is: (1+z)*z*sum_i weights[i]/E(a_i) with a_i = 1/(1+z*nodes[i]).
    E(a) is H(a)/H0.
    The integrand in the t-domain is f(t) = 1/E(z*t).
    The quadrature approximates int_0^1 f(t) dt.
    
    So we need to compute the exact value of int_0^1 1/E(z*t) dt.
    """
    if z < 1e-10:
        return 0.0
        
    # Nodes and weights for GL on [0,1]
    # Standard leggauss is on [-1,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    t_nodes = (x_gl + 1.0) / 2.0
    t_weights = w_gl / 2.0
    
    # Calculate z * t_nodes
    z_t = z * t_nodes
    
    # Calculate E(a) where a = 1/(1+z*t)
    a_vals = 1.0 / (1.0 + z_t)
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
        
    # E^2 = Om/a^3 + Ode * a^(-3w0) * exp(-3wa(1-a))
    term1 = Om / (a_vals**3)
    term2 = Ode * (a_vals**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a_vals))
    E2 = term1 + term2
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    
    integrand_vals = 1.0 / E
    
    # Perform quadrature
    integral_val = np.sum(t_weights * integrand_vals)
    
    # The total distance factor is (1+z)*z * integral_val
    return (1.0 + z) * z * integral_val

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    if z < 1e-10:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a_vals**3)
    term2 = Ode * (a_vals**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a_vals))
    E2 = term1 + term2
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.sum(weights / E)

def _decode_params(params):
    """
    Decode unconstrained parameters to valid nodes and weights.
    params: array of length 16. First 8 are log-spacings for nodes, next 8 are log-weights.
    """
    u = params[:8]
    v = params[8:16]
    
    # Nodes: cumulative sum of exponentials
    # e^u_i are positive.
    exp_u = np.exp(u)
    cumsum_u = np.cumsum(exp_u)
    nodes = cumsum_u / cumsum_u[-1]
    # nodes are strictly increasing in (0, 1)
    
    # Weights: softmax
    exp_v = np.exp(v)
    weights = exp_v / np.sum(exp_v)
    
    return nodes, weights

def _objective_unconstrained(params, portfolio, z_grid, t_start, deadline):
    if time.time() - t_start > deadline:
        return 1e10
        
    nodes, weights = _decode_params(params)
    
    # Calculate errors
    errors = []
    for Om, w0, wa in portfolio:
        for z in z_grid:
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = _reference_value_gl(z, Om, w0, wa, n_gl=100)
            
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                errors.append(rel_err)
                
    if not errors:
        return 1e10
        
    errors = np.array(errors)
    
    # Soft-max objective
    # We want to minimize the maximum error.
    # Using log-sum-exp for smoothness.
    beta = 50.0
    # Shift for numerical stability
    max_err = np.max(errors)
    log_sum_exp = np.log(np.sum(np.exp(beta * (errors - max_err)))) + beta * max_err
    return (1.0 / beta) * log_sum_exp

def construct():
    t_start = time.time()
    deadline = 85.0  # seconds, leave 5s margin
    
    # Portfolio of cosmologies
    portfolio = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.27, -1.0, 0.0),
        (0.27, -1.2, 0.5),
        (0.27, -0.8, -0.5),
        (0.31, -1.1, 0.2),
        (0.31, -0.9, -0.2),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
    ]
    
    # z grid for error evaluation
    z_grid = np.linspace(0.1, 2.0, 40)
    
    # Initial guess: Gauss-Legendre nodes/weights mapped to unconstrained space
    # We need to find u, v such that decode(u,v) approximates GL.
    # This is an inverse problem.
    # Alternatively, just start with u=0, v=0 (uniform nodes/weights) or random.
    # Uniform nodes: nodes = [1/8, 2/8, ..., 7/8].
    # This corresponds to exp(u) being constant, so u=0.
    # Uniform weights: weights = 1/8.
    # This corresponds to exp(v) being constant, so v=0.
    
    x0 = np.zeros(16)
    
    best_params = x0.copy()
    best_obj = _objective_unconstrained(x0, portfolio, z_grid, t_start, deadline)
    
    # Optimization loop with multiple restarts
    # We use L-BFGS-B as it is fast for unconstrained problems
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - t_start > deadline - 5.0:
            break
            
        # For the first run, use the initial guess.
        # For subsequent runs, use small random perturbations of the best found so far.
        if i == 0:
            x_curr = x0.copy()
        else:
            # Perturb the best solution found so far
            noise = np.random.normal(0, 0.5, 16)
            x_curr = best_params + noise
        
        try:
            res = minimize(_objective_unconstrained, x_curr, 
                           args=(portfolio, z_grid, t_start, deadline),
                           method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final polish with SLSQP on the constrained variables?
    # Or just use the best unconstrained parameters.
    # Let's do one final polish using the decoded nodes/weights as the starting point
    # for a constrained optimization to ensure we are at a local minimum in the
    # constrained space as well, although the mapping is bijective.
    
    nodes, weights = _decode_params(best_params)
    
    # Ensure strict constraints are met (though they should be)
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1-1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
