"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    The same nodes and weights must work across all models. We minimize the worst relative
    error against independent adaptive integration on disjoint public validation cosmologies.
    
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i]) and E(a)=H(a)/H0
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Define a set of representative dark-energy models for optimization
    # We'll use a variety of CPL (Chevallier-Polarski-Linder) parameterizations
    # and some other common models to ensure universality
    
    # Model 1: LCDM (w=-1)
    # Model 2: CPL with w0=-1, wa=0.5
    # Model 3: CPL with w0=-0.5, wa=0.3
    # Model 4: CPL with w0=-1.5, wa=0.2
    # Model 5: CPL with w0=-0.8, wa=0.4
    # Model 6: CPL with w0=-1.2, wa=0.6
    # Model 7: CPL with w0=-0.3, wa=0.1
    # Model 8: CPL with w0=-1.0, wa=0.8
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model in flat universe."""
        # E^2 = (1-a)^(-3(1+w0+wa)) * a^(-3(1+w0)) * (1 + wa*(1-a))
        # More precisely: E^2 = (1+z)^3 * (1 + w0 + wa*z)^(3*wa/(1+w0+wa)) ... 
        # Actually, the standard CPL parametrization:
        # w(z) = w0 + wa*z
        # E(z) = (1+z)^(3/2) * (1 + w0 + wa*z)^(3*wa/(1+w0+wa)) * [1 + (3/2)*(w0+1)*(1-(1+z)^(-1))]^... 
        # Let me use the correct formula:
        # For flat universe: E(z)^2 = (1+z)^3 * (1 + w0 + wa*z)^(3*wa) / (1 + wa*z)^(3*(w0+1+wa))
        # Wait, let me be more careful. The Hubble parameter for CPL:
        # H(z) = H0 * (1+z)^(3(1+w0+wa)/2) * (1 + w0 + wa*z)^(3*wa/(2*(1+w0+wa))) * 
        #        [1 + (3/2)*(w0+1)*(1 - (1+z)^(-1))]^... 
        # Actually, the standard form is:
        # E(z)^2 = (1+z)^3 * (1 + w0 + wa*z)^(3*wa) / (1 + wa*z)^(3*(w0+1))
        # No, that's not right either. Let me use the integral form:
        # E(z)^2 = (1+z)^(3(1+w0+wa)) * (1 + w0 + wa*z)^(3*wa/(1+w0+wa))
        # Hmm, I need to be careful. The correct formula for CPL in flat universe:
        # H(z)/H0 = (1+z)^(3/2) * (1 + w0 + wa*z)^(3*wa/(2*(1+w0+wa))) * 
        #           [1 + (3/2)*(w0+1)*(1 - (1+z)^(-1))]^... 
        # Actually, the simplest correct form:
        # E(z)^2 = (1+z)^3 * (1 + w0 + wa*z)^(3*wa) / (1 + wa*z)^(3*(w0+1))
        # Let me verify: if wa=0, this gives E(z)^2 = (1+z)^3 * 1 / 1 = (1+z)^3, which is correct for w=-1 (LCDM).
        # If w0=-0.5, wa=0: E(z)^2 = (1+z)^3 * 1 / (1)^{3*0.5} = (1+z)^3. But for w=-0.5, E(z)^2 should be (1+z)^{3*(1+0.5)} = (1+z)^{4.5}.
        # So that formula is wrong.
        
        # Correct formula: For CPL w(z) = w0 + wa*z, in flat universe:
        # E(z)^2 = (1+z)^3 * (1 + w0 + wa*z)^(3*wa) / (1 + wa*z)^(3*(w0+1+wa))
        # No, that's still not right.
        
        # The correct expression is:
        # E(z) = (1+z)^(3(1+w0+wa)/2) * (1 + w0 + wa*z)^(3*wa/(2*(1+w0+wa))) * 
        #        [1 + (3/2)*(w0+1)*(1 - (1+z)^(-1))]^... 
        # Actually, I recall now. The standard CPL formula in terms of a = 1/(1+z):
        # E(a)^2 = a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(3*wa/(1+w0+wa))
        # Let me verify: if wa=0, E(a)^2 = a^(-3(1+w0)) = (1+z)^(3(1+w0)), which is correct for constant w.
        # For LCDM (w0=-1, wa=0): E(a)^2 = a^(-3*0) = 1... that's wrong.
        # Wait, for LCDM, w=-1, so 1+w0 = 0, so a^(-3*0) = 1. But E(a)^2 should be a^(-3) for LCDM.
        # So that formula is also wrong.
        
        # Let me start over. The Friedmann equation for flat universe:
        # H^2(a) = H0^2 * [Omega_m * a^(-3) + Omega_DE * a^(-3(1+w(a)))]
        # For CPL: w(a) = w0 + wa*(1-a)  [note: some use w(z) = w0 + wa*z, which is w(a) = w0 + wa*(1-a)/a]
        # Let's use w(z) = w0 + wa*z, so w(a) = w0 + wa*(1-a)/a
        
        # The integral for E(a)^2 involves:
        # E(a)^2 = a^(-3) * [1 + (Omega_DE/Omega_m) * a^3 * exp(3*integral_a^1 w(a')/a' da')]
        # This is getting complicated. Let me just use the numerical form that's standard.
        
        # Actually, for the purpose of this quadrature task, the evaluator computes the integrands itself.
        # The task says: "The evaluator computes integrands itself."
        # So I need to optimize nodes and weights such that the quadrature approximates the integral well
        # for a variety of E(a) functions.
        
        # The integral is: dL/(c/H0) = integral_0^z (1+z')/E(z') dz'
        # In terms of a: z = 1/a - 1, dz = -da/a^2
        # dL/(c/H0) = integral_1^{1/(1+z)} (1 + 1/a - 1)/E(1/a - 1) * da/a^2
        #           = integral_1^{1/(1+z)} (1/a)/E(1/a - 1) * da/a^2
        #           = integral_1^{1/(1+z)} da/(a^3 * E(1/a - 1))
        
        # Wait, let me redo this. The standard luminosity distance formula:
        # dL = c/H0 * (1+z) * integral_0^z dz'/E(z')
        # In terms of a = 1/(1+z): when z=0, a=1; when z=z, a=1/(1+z)
        # dz = -da/a^2
        # integral_0^z dz'/E(z') = integral_1^{1/(1+z)} (-da/a^2)/E(1/a - 1)
        #                         = integral_{1/(1+z)}^1 da/(a^2 * E(1/a - 1))
        # So dL/(c/H0) = (1+z) * integral_{1/(1+z)}^1 da/(a^2 * E(1/a - 1))
        
        # But the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
        # This suggests the quadrature is applied to integral_0^z f(z') dz' where f(z') = 1/E(z')
        # and the nodes are in [0,1] mapped to z' = z * nodes[i]
        # So the integral is approximated as: z * sum_i weights[i] * f(z * nodes[i])
        # = z * sum_i weights[i] / E(z * nodes[i])
        # And then multiplied by (1+z)
        
        # So the quadrature rule is for approximating integral_0^1 g(t) dt where g(t) = f(z*t) = 1/E(z*t)
        # and the result is z * sum_i weights[i] * g(nodes[i])
        
        # For the quadrature to be universal, the nodes and weights should work well for various shapes of g(t)
        # which depend on the cosmology model and the redshift z.
        
        # Since the evaluator computes integrands itself and measures relative error against adaptive integration,
        # I need to find nodes and weights that minimize worst-case relative error across different cosmologies.
        
        # Strategy: Use a set of representative E(a) functions, compute the "true" integral via adaptive quadrature,
        # then optimize nodes and weights to minimize the maximum relative error.
        
        pass
    
    # Let me define specific models and use scipy to optimize
    # I'll use several CPL models and maybe some others
    
    # Define E(z) for different models
    def E_lcdm(z):
        """LCDM: Omega_m=0.3, Omega_L=0.7, w=-1"""
        Omega_m = 0.3
        Omega_L = 0.7
        return np.sqrt(Omega_m * (1+z)**3 + Omega_L)
    
    def E_cpl(z, w0, wa, Omega_m=0.3):
        """CPL model with given w0, wa, and Omega_m"""
        Omega_DE = 1.0 - Omega_m
        # E(z)^2 = Omega_m*(1+z)^3 + Omega_DE * exp(3*integral_0^z w(z')/(1+z') dz')
        # For CPL: w(z') = w0 + wa*z'
        # integral_0^z (w0 + wa*z')/(1+z') dz' = integral_0^z [w0 + wa - wa/(1+z')] dz'
        #                                       = (w0+wa)*z - wa*ln(1+z)
        # So E(z)^2 = Omega_m*(1+z)^3 + Omega_DE * exp(3*((w0+wa)*z - wa*ln(1+z)))
        #           = Omega_m*(1+z)^3 + Omega_DE * (1+z)^(-3*wa) * exp(3*(w0+wa)*z)
        
        # This can be numerically unstable for large z. Let me use a more stable form.
        # Actually, let me just compute it directly.
        exp_term = np.exp(3.0 * ((w0 + wa) * z - wa * np.log1p(z)))
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * (1.0 + z)**(-3.0*wa) * exp_term)
    
    # Define a set of models to test against
    models = [
        ('lcdm', E_lcdm),
        ('cpl1', lambda z: E_cpl(z, -1.0, 0.5)),
        ('cpl2', lambda z: E_cpl(z, -0.5, 0.3)),
        ('cpl3', lambda z: E_cpl(z, -1.5, 0.2)),
        ('cpl4', lambda z: E_cpl(z, -0.8, 0.4)),
        ('cpl5', lambda z: E_cpl(z, -1.2, 0.6)),
        ('cpl6', lambda z: E_cpl(z, -0.3, 0.1)),
        ('cpl7', lambda z: E_cpl(z, -1.0, 0.8)),
        ('cpl8', lambda z: E_cpl(z, 0.0, 0.0)),  # Einstein-de Sitter like (but with Omega_m=0.3)
        ('cpl9', lambda z: E_cpl(z, -2.0, 0.1)),
    ]
    
    # Test redshifts
    z_values = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
    
    # Function to compute the true integral using adaptive quadrature
    from scipy.integrate import quad
    
    def true_integral(E_func, z):
        """Compute (1+z) * integral_0^z dz'/E(z') using adaptive quadrature."""
        integrand = lambda zp: 1.0 / E_func(zp)
        result, _ = quad(integrand, 0, z, limit=200)
        return (1.0 + z) * result
    
    # Function to compute the quadrature approximation
    def quadrature_approx(nodes, weights, E_func, z):
        """Compute (1+z)*z*sum_i weights[i]/E(z*nodes[i])"""
        z_nodes = z * nodes
        E_vals = E_func(z_nodes)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    # Objective function: maximize worst-case relative error (we minimize this)
    def objective(params):
        # params: 16 values - 8 nodes (unconstrained) and 8 weights (unconstrained)
        # We need to ensure nodes are in (0,1) and strictly increasing, weights are positive and sum to 1
        
        # Transform to ensure constraints
        # For nodes: use softmax or cumulative sum of exponentials
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: transform to (0,1) using cumulative sum of softmax
        exp_nodes = np.exp(node_params)
        nodes = np.cumsum(exp_nodes)
        nodes = nodes / nodes[-1]  # normalize to [0,1]
        # Ensure strictly increasing: add small epsilon
        nodes = np.maximum.accumulate(nodes)
        nodes = nodes + np.linspace(0, 1e-10, 8)  # ensure strictly increasing
        nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
        
        # Weights: transform to positive and sum to 1 using softmax
        exp_weights = np.exp(weight_params)
        weights = exp_weights / np.sum(exp_weights)
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for model_name, E_func in models:
            for z in z_values:
                true_val = true_integral(E_func, z)
                approx_val = quadrature_approx(nodes, weights, E_func, z)
                if true_val > 1e-10:
                    rel_err = abs(approx_val - true_val) / true_val
                    max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err
    
    # Initial guess: midpoint rule
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    
    # Convert to unconstrained parameters
    # For nodes: use log-spaced or something
    # For weights: use log
    init_params = np.concatenate([np.log(init_nodes), np.log(init_weights)])
    
    # Optimize using Nelder-Mead
    best_params = init_params.copy()
    best_obj = objective(init_params)
    
    # Multiple restarts with different initial points
    restarts = [
        init_params,
        np.log(np.linspace(0.05, 0.95, 8)) + np.log(init_weights),
        np.log(1.0 - np.linspace(0.05, 0.95, 8)) + np.log(init_weights),
        init_params + np.random.randn(16) * 0.1,
    ]
    
    for i, init_p in enumerate(restarts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            result = minimize(objective, init_p, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    exp_nodes = np.exp(node_params)
    nodes = np.cumsum(exp_nodes)
    nodes = nodes / nodes[-1]
    nodes = np.maximum.accumulate(nodes)
    nodes = nodes + np.linspace(0, 1e-10, 8)
    nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
    
    exp_weights = np.exp(weight_params)
    weights = exp_weights / np.sum(exp_weights)
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    # Final check: ensure nodes are strictly increasing in [0,1]
    if not np.all(np.diff(nodes) > 0) or not np.all(nodes > 0) or not np.all(nodes < 1):
        # Fall back to midpoint rule
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
