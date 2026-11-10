"""
Strategy: Adaptive Minimax Optimization of Quadrature Nodes and Weights.
Rationale: The integrand 1/E(z) in CPL cosmologies is smooth but not a polynomial.
Gauss-Legendre quadrature is optimal for polynomials but suboptimal for this specific
class of functions, particularly at high redshifts where the shape of 1/E(z) deviates
from polynomial behavior. By treating the nodes and weights as free parameters and
optimizing them to minimize the worst-case relative error (minimax criterion) against
a high-resolution reference integral over a diverse set of CPL parameters (w0, wa)
and redshifts (z), we can find a rule that is specifically tailored to the geometry
of cosmological distance integrals. This approach trades the theoretical guarantees
of GL for empirical robustness across the validation space.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leaving 5s margin for safety

def E_z(z, w0, wa):
    """
    Compute H(z)/H0 for a CPL (Chevalier-Polarski-Linder) cosmology.
    E(z) = sqrt((1+z)^3 * (1 + w0 - w0/(1+z)) * (1 + wa * z/(1+z))^(-3*wa))
    Note: Standard CPL form is E^2 = (1+z)^3 * Omega_m + Omega_k(1+z)^2 + Omega_L (1+wa z/(1+z))^{3(1+w0+wa)}
    For spatially flat (Omega_k=0) and assuming matter domination is handled or 
    the task implies a specific simplified form. 
    Looking at the prompt: "spatially flat, time-varying dark-energy models".
    Usually dL = c/H0 * (1+z) * integral_0^z dz'/E(z').
    The prompt formula: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i]).
    This implies the integral is over x in [0,1] of 1/E(z*x) dx.
    E(a) = H(a)/H0.
    Standard CPL E(a) = a^-3 * [Omega_m + Omega_L (a/(1+wa))^{3(1+w0+wa)}]^{1/2}
    Since it's spatially flat, Omega_k=0.
    However, the prompt doesn't specify Omega_m. 
    Wait, "universal luminosity-distance quadrature".
    If Omega_m is not fixed, the rule can't be universal unless the integrand 
    structure allows it or the error metric is normalized.
    Re-reading: "Minimize the worst relative error against independent adaptive integration on disjoint public validation cosmologies."
    This implies the validation cosmologies have specific parameters.
    The "public task contract" says "reconstructed from published distance and CPL equations".
    Usually, in these synthetic tasks, if parameters aren't given, we assume a standard 
    Lambda-CDM like background or the optimization must be robust to a range of parameters.
    Given the "universal" requirement, the nodes/weights must work for ALL models.
    This suggests the integrand 1/E(z) has a common feature.
    Let's assume the standard flat universe with matter and dark energy.
    E(a) = sqrt( a^-3 * Omega_m + Omega_L * (a/(1+wa))^{3(1+w0+wa)} )
    For the optimization to work, we need a representative set of (w0, wa, Omega_m).
    Common values: w0=-1, wa=0 (LambdaCDM).
    Validation likely includes variations like w0=-0.9, wa=0.2, etc.
    We will optimize against a grid of such parameters.
    """
    if z < 0:
        return 1.0
    # Assume standard flat cosmology parameters for the "shape" of the integrand.
    # Since the rule must be universal, we optimize against a range of parameters.
    # For the local optimization, we need a specific E(z) function.
    # We will use a representative set.
    pass

def compute_E(z, w0, wa, omega_m):
    """
    Compute E(z) = H(z)/H0 for flat CPL cosmology.
    E^2 = (1+z)^3 * [ omega_m + (1-omega_m) * (1+wa*z/(1+z))^{3(1+w0+wa)} ]
    Note: Omega_L = 1 - omega_m.
    """
    if z < 0:
        return 1.0
    base = 1 + wa * z / (1 + z)
    if base < 1e-12:
        base = 1e-12
    de_term = base ** (3 * (1 + w0 + wa))
    e2 = (1 + z) ** 3 * (omega_m + (1 - omega_m) * de_term)
    return np.sqrt(e2)

def true_integral_ref(z, w0, wa, omega_m):
    """
    High-resolution reference integral for int_0^1 dx / E(z*x).
    """
    if z == 0:
        return 1.0
    x_arr = np.linspace(0, 1, 2000)
    z_arr = z * x_arr
    E_arr = np.array([compute_E(zv, w0, wa, omega_m) for zv in z_arr])
    integrand = 1.0 / E_arr
    # Use Simpson's rule or high-res trapz
    return np.trapz(integrand, x_arr)

def quadrature_approx(nodes, weights, z, w0, wa, omega_m):
    """
    Approximate the integral using the given nodes and weights.
    """
    z_arr = z * nodes
    E_arr = np.array([compute_E(zv, w0, wa, omega_m) for zv in z_arr])
    return np.sum(weights / E_arr)

def objective(params):
    """
    Minimize the worst-case relative error across a set of test cases.
    params: 16 values (8 nodes, 8 weights).
    """
    nodes = params[:8]
    weights = params[8:]
    
    # Enforce constraints:
    # 1. Nodes strictly increasing in [0, 1]
    # 2. Weights positive and sum to 1
    
    # Penalize violations heavily
    penalty = 0.0
    
    # Check nodes
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        penalty += 1000.0
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            penalty += 1000.0
    
    # Check weights
    if np.any(weights <= 0):
        penalty += 1000.0
    w_sum = np.sum(weights)
    if abs(w_sum - 1.0) > 1e-9:
        penalty += 1000.0 * abs(w_sum - 1.0)
        
    if penalty > 1e-6:
        return penalty

    # Normalize weights to ensure sum=1 for evaluation (if slightly off due to optimizer noise)
    weights_norm = weights / np.sum(weights)
    
    # Test cases: Representative CPL parameters and redshifts
    # w0, wa, omega_m, z
    test_cases = [
        (-1.0, 0.0, 0.3, 1.0),   # LambdaCDM, z=1
        (-1.0, 0.0, 0.3, 5.0),   # LambdaCDM, z=5
        (-0.9, 0.1, 0.3, 1.0),   # Phantom-ish, z=1
        (-0.9, 0.1, 0.3, 5.0),   # Phantom-ish, z=5
        (-1.1, -0.1, 0.3, 1.0),  # Quintessence-ish, z=1
        (-1.1, -0.1, 0.3, 5.0),  # Quintessence-ish, z=5
        (-1.0, 0.0, 0.2, 10.0),  # High z, low matter
        (-1.0, 0.0, 0.4, 10.0),  # High z, high matter
        (-0.5, 0.5, 0.3, 2.0),   # Strong wa variation
        (-1.5, 0.5, 0.3, 2.0),   # Extreme w0
    ]
    
    max_rel_err = 0.0
    for w0, wa, om, z in test_cases:
        try:
            true_val = true_integral_ref(z, w0, wa, om)
            approx_val = quadrature_approx(nodes, weights_norm, z, w0, wa, om)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
        except Exception:
            return 1000.0
            
    return max_rel_err

def construct():
    """
    Construct the 8-node positive quadrature rule.
    """
    # Initial guess: Gauss-Legendre nodes and weights mapped to [0,1]
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (nodes_gl + 1.0)
    weights_init = 0.5 * weights_gl
    
    x0 = np.concatenate([nodes_init, weights_init])
    
    # Optimization
    best_x = x0
    best_err = np.inf
    
    try:
        # Use Nelder-Mead for derivative-free optimization, robust to non-smooth minimax objective
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < best_err:
            best_err = result.fun
            best_x = result.x
            
        # Try a few more local optimizations from perturbed starts if time permits
        if time.time() < DEADLINE - 10:
            for i in range(3):
                # Perturb initial guess
                x_pert = x0 + 0.1 * np.random.randn(16)
                # Ensure basic validity for start
                x_pert[:8] = np.sort(x_pert[:8])
                x_pert[:8] = np.clip(x_pert[:8], 1e-6, 1-1e-6)
                x_pert[8:] = np.abs(x_pert[8:])
                x_pert[8:] /= np.sum(x_pert[8:])
                
                res_pert = minimize(objective, x_pert, method='Nelder-Mead', 
                                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res_pert.fun < best_err:
                    best_err = res_pert.fun
                    best_x = res_pert.x
                    if time.time() > DEADLINE - 15:
                        break
                        
    except Exception as e:
        # If optimization fails, fall back to GL
        best_x = x0

    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Post-processing to ensure strict validity
    # 1. Sort nodes
    nodes = np.sort(nodes)
    
    # 2. Clip nodes to (0, 1)
    eps = 1e-12
    nodes = np.clip(nodes, eps, 1.0 - eps)
    
    # 3. Enforce strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
            
    # 4. Ensure weights are positive and sum to 1
    weights = np.abs(weights)
    if np.sum(weights) < 1e-12:
        weights = np.full(8, 1.0/8.0)
    else:
        weights = weights / np.sum(weights)
        
    # Final check: if any weight is too small, redistribute?
    # The task requires positive weights.
    if np.any(weights <= 0):
        weights = np.full(8, 1.0/8.0)
        
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
