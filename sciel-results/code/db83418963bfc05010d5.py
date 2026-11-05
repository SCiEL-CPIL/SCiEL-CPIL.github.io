"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    """
    Constructs an 8-node positive quadrature rule by performing a local
    optimization to minimize the worst-case relative error on a small,
    representative set of cosmologies.
    """
    
    # Define representative cosmologies for training
    # E(a)^2 = Om*a^(-3) + (1-Om)*a^(-3*(1+w0+wa))*(1+a*wa)^(-3)
    # This is an approximation for CPL, sufficient for finding a robust rule.
    def make_E(w0, wa, Om=0.3):
        def E(a):
            term_matter = Om * a**(-3)
            term_de = (1 - Om) * a**(-3 * (1 + w0 + wa)) * (1 + a * wa)**(-3)
            # Handle potential numerical issues if terms are negative (shouldn't be for physical params)
            val = term_matter + term_de
            return np.sqrt(val) if val > 0 else 1e-10
        return E

    # Training set: diverse w0, wa and z
    # We want to cover the space of possible integrand shapes.
    train_cosmos = [
        make_E(-1.0, 0.0),   # Lambda-CDM
        make_E(-1.2, 0.5),   # Phantom/Evolving
        make_E(-0.8, -0.5),  # Quintessence-like
        make_E(-1.0, 1.0),   # Strongly evolving
    ]
    train_zs = [0.3, 0.7, 1.5, 3.0]
    
    n = 8
    
    # Pre-compute "exact" integrals for the training set
    # J(z) = ∫_0^1 dt / E(1/(1+zt))
    exact_integrals = []
    for E_func in train_cosmos:
        for z in train_zs:
            def integrand(t):
                a = 1.0 / (1.0 + z * t)
                return 1.0 / E_func(a)
            val, _ = quad(integrand, 0, 1, limit=50)
            exact_integrals.append(val)
            
    exact_integrals = np.array(exact_integrals)
    
    # Initial guess: Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes_init = 0.5 * (x_gl + 1)
    weights_init = 0.5 * w_gl
    
    # Objective function: Minimize sum of squared relative errors
    def objective(params):
        # First n params are nodes, next n are log-weights
        nodes = params[:n]
        log_weights = params[n:]
        
        # Ensure nodes are in (0,1) and strictly increasing
        # Apply sigmoid to map to (0,1)
        nodes = 1.0 / (1.0 + np.exp(-10 * (nodes - 0.5)))
        
        # Enforce strict increase by adding a small penalty or fixing
        # If not increasing, return large penalty
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                return 1e6
        
        # Weights from log-weights, normalized
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        # Calculate errors
        errors = 0.0
        idx = 0
        for E_func in train_cosmos:
            for z in train_zs:
                # Quadrature approximation
                approx = 0.0
                for i in range(n):
                    a_i = 1.0 / (1.0 + z * nodes[i])
                    approx += weights[i] / E_func(a_i)
                
                exact = exact_integrals[idx]
                idx += 1
                
                if exact == 0:
                    continue
                rel_err = (approx - exact) / exact
                errors += rel_err**2
                
        return errors

    # Initial parameters
    params_init = np.concatenate([nodes_init, np.log(weights_init)])
    
    # Bounds for L-BFGS-B
    # Nodes: We mapped via sigmoid, so raw params can be anything, but let's bound them to avoid extreme sigmoid values
    # Sigmoid input bounds: [-5, 5] maps to approx [0.006, 0.994]
    # Weights: log-weights, can be anything, but let's bound to avoid overflow
    bounds = [(-5, 5)] * n + [(-10, 10)] * n
    
    # Run optimization
    # Maxiter kept low to ensure < 60s
    res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 100, 'ftol': 1e-10})
    
    # Extract optimal nodes and weights
    if res.success:
        nodes_opt = res.x[:n]
        log_weights_opt = res.x[n:]
        
        # Apply sigmoid
        nodes_opt = 1.0 / (1.0 + np.exp(-10 * (nodes_opt - 0.5)))
        
        # Ensure strict increase (should be handled by objective, but double check)
        # If failed, fall back to init
        valid = True
        for i in range(1, n):
            if nodes_opt[i] <= nodes_opt[i-1]:
                valid = False
                break
        if not valid:
            nodes_opt = nodes_init
            log_weights_opt = np.log(weights_init)
            
        weights_opt = np.exp(log_weights_opt)
        weights_opt = weights_opt / np.sum(weights_opt)
        
        return {"nodes": nodes_opt, "weights": weights_opt}
    else:
        # Fallback to Gauss-Legendre if optimization fails
        return {"nodes": nodes_init, "weights": weights_init}
# EVOLVE-BLOCK-END
