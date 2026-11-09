"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    Uses a rational-function (Padé-like) approach to approximate 1/E(a) and derive
    optimal nodes/weights via convex optimization.
    """
    # We want to find nodes z_i in (0,1) and weights w_i > 0, sum w_i = 1
    # such that the quadrature approximates:
    #   dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i)
    # where a_i = 1/(1 + z*z_i)
    #
    # The approach: 
    # 1. Define a representative set of CPL cosmologies (w0, wa)
    # 2. For each, compute the exact integral via adaptive quadrature
    # 3. Optimize nodes and weights to minimize worst-case relative error
    #
    # To make this tractable, we parameterize nodes via a rational mapping
    # and solve a convex subproblem for weights given nodes.

    # Representative CPL cosmologies for training
    # E(a) = (a^-3 + (1+w0+wa)*a^(-3(1+w0)) * exp(-3*wa*(1-a)/(1+w0)))^0.5
    # for w0 != -1; for w0 = -1, E(a) = (a^-3 + (1+w0+wa)*a^(-3(1+w0)) * exp(-3*wa*(1-a)/(1+w0)))^0.5
    # Actually the standard CPL form is:
    # E(a) = (a^-3 + (1+w0+wa) * a^(-3(1+w0)) * exp(-3*wa*(1-a)/(1+w0)))^0.5
    # But this has issues at w0=-1. Let's use the more standard form:
    # E(a) = (a^-3 + (1+w0+wa) * a^(-3(1+w0)) * exp(-3*wa*(1-a)/(1+w0)))^0.5
    # For w0 close to -1, use the limit.

    def E_of_a(a, w0, wa):
        """Hubble parameter normalized by H0 for CPL model."""
        a = np.maximum(a, 1e-15)  # avoid division by zero
        if abs(w0 + 1.0) < 1e-10:
            # w0 = -1 limit: E(a) = (a^-3 + (1+wa) * exp(-3*wa*(1-a)))^0.5
            # Actually for w0=-1: E(a)^2 = a^-3 + (1+wa)*a^0 * exp(-3*wa*(1-a))
            # = a^-3 + (1+wa)*exp(-3*wa*(1-a))
            return np.sqrt(a**(-3) + (1.0 + wa) * np.exp(-3.0 * wa * (1.0 - a)))
        else:
            # General CPL
            exponent = -3.0 * (1.0 + w0)
            prefactor = (1.0 + w0 + wa)
            exp_term = np.exp(-3.0 * wa * (1.0 - a) / (1.0 + w0))
            return np.sqrt(a**(-3) + prefactor * a**exponent * exp_term)

    def exact_luminosity_distance(z, w0, wa):
        """Compute exact dL/(c/H0) via adaptive quadrature."""
        from scipy.integrate import quad
        
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return (1.0 + zp) * zp / E_of_a(a, w0, wa)
        
        result, _ = quad(integrand, 0.0, z, limit=200)
        return result

    def quadrature_luminosity_distance(z, nodes, weights, w0, wa):
        """Compute quadrature approximation of dL/(c/H0)."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)

    # Training cosmologies
    training_cosmos = [
        (0.0, 0.0),    # Lambda CDM
        (-0.5, 0.5),   # quintessence-like
        (-1.5, 0.5),   # phantom-like
        (-1.0, 0.5),   # w0 = -1
        (-1.0, -0.5),  # w0 = -1, negative wa
        (-0.5, -0.5),  # another
        (-1.5, -0.5),  # another
        (0.0, 0.5),    # another
    ]

    # Test redshifts
    test_zs = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

    # Precompute exact values
    exact_values = {}
    for (w0, wa) in training_cosmos:
        exact_values[(w0, wa)] = np.array([exact_luminosity_distance(z, w0, wa) for z in test_zs])

    # We'll optimize nodes and weights.
    # Parameterize nodes via sigmoid of linear parameters to ensure they're in (0,1) and increasing.
    # Let t_i be parameters, nodes = sigmoid(t_i), and we enforce t_1 < t_2 < ... < t_8.
    
    n = 8
    
    def objective(params):
        # params: [t_1, ..., t_8, log_w_1, ..., log_w_8]
        t = params[:n]
        log_w = params[n:]
        
        nodes = 1.0 / (1.0 + np.exp(-t))  # sigmoid
        w = np.exp(log_w)
        
        # Enforce increasing nodes: penalty if not
        diffs = np.diff(nodes)
        if np.any(diffs <= 0):
            return 1e10 + np.sum(np.minimum(diffs, 0)**2)
        
        # Normalize weights
        w = w / np.sum(w)
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for (w0, wa) in training_cosmos:
            exact = exact_values[(w0, wa)]
            for i_z, z in enumerate(test_zs):
                approx = quadrature_luminosity_distance(z, nodes, w, w0, wa)
                rel_err = abs(approx - exact[i_z]) / max(exact[i_z], 1e-15)
                max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err

    # Initial guess: uniform nodes and weights
    t_init = np.log(np.arange(1, n+1) / (n + 1 - np.arange(1, n+1)))  # logit of uniform nodes
    log_w_init = np.zeros(n)
    
    x0 = np.concatenate([t_init, log_w_init])
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    t_opt = result.x[:n]
    log_w_opt = result.x[n:]
    
    nodes = 1.0 / (1.0 + np.exp(-t_opt))
    weights = np.exp(log_w_opt)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    # Clip nodes to (1e-6, 1-1e-6)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    # Ensure strict increase
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    
    # Final normalization
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
