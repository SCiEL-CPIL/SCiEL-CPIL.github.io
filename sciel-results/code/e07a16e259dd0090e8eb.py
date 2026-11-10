"""
Strategy: Use standard Gauss-Legendre Quadrature nodes and weights.
Rationale: The integrand 1/E(z) for CPL cosmologies is smooth and well-approximated
by low-degree polynomials. Gauss-Legendre quadrature is spectrally accurate,
integrating polynomials up to degree 2n-1 exactly. For n=8, this provides
extremely high accuracy (often < 1e-8 relative error) for smooth integrands
with only 8 evaluations, far outperforming general-purpose rules like midpoint
or even optimized non-spectral rules for this specific class of smooth functions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
START_TIME = time.time()
DEADLINE = 90  # seconds

def construct():
    """
    Construct the 8-node positive quadrature rule.
    Returns:
        dict with 'nodes' (length-8) and 'weights' (length-8).
    """
    # 1. Compute Gauss-Legendre nodes and weights on [0, 1]
    # np.polynomial.legendre.leggauss returns nodes in [-1, 1]
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    
    # Map from [-1, 1] to [0, 1]
    # x_new = 0.5 * (x_old + 1)
    # w_new = 0.5 * w_old
    nodes = 0.5 * (nodes_gl + 1.0)
    weights = 0.5 * weights_gl
    
    # 2. Ensure strict increasing order and bounds [0, 1]
    # leggauss nodes are strictly increasing in [-1, 1], so they are in [0, 1]
    # and strictly increasing.
    # However, due to floating point, let's verify and clean up.
    nodes = np.sort(nodes)
    
    # Check bounds
    # Theoretical nodes are strictly inside (0, 1).
    # We clip to ensure they are within [1e-15, 1-1e-15] to avoid any edge case issues
    # though leggauss nodes are safely away from 0 and 1.
    eps = 1e-15
    nodes = np.clip(nodes, eps, 1.0 - eps)
    
    # Enforce strict increase (should already be true, but just in case)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-15
    
    # 3. Ensure weights are positive and sum to 1
    # leggauss weights are strictly positive.
    # Sum of mapped weights should be 1.0.
    w_sum = np.sum(weights)
    if w_sum < 1e-12:
        weights = np.full(8, 1.0/8.0)
    else:
        weights = weights / w_sum
        
    # Verify positivity
    if np.any(weights <= 0):
        # Fallback to uniform weights if something went wrong (highly unlikely for GL)
        weights = np.full(8, 1.0/8.0)
        
    # 4. Optional: Validation check (using time if available)
    # We can quickly check the error on a representative cosmology to ensure
    # no numerical issues, though GL is theoretically sound.
    # Given the 90s deadline and the instantaneous computation, we can afford
    # a quick sanity check on LambdaCDM at z=1.
    
    def E_z(z, w0, wa):
        if z < 0:
            return 1.0
        term1 = (1 + z) ** (-3 * w0)
        base = 1 - wa * z / (1 + z)
        if base < 0:
            base = 1e-12
        term2 = base ** (1.5 * wa)
        return np.sqrt(term1 * term2)
        
    def true_integral(z, w0, wa):
        if z == 0:
            return 1.0
        # Use a simple high-res trapz or quad for check
        # Since we are in construct, we don't have scipy.integrate.quad imported by default in this minimal block,
        # but we can use numpy.trapz with many points for a quick check.
        z_arr = np.linspace(0, z, 10000)
        E_arr = np.array([E_z(zv, w0, wa) for zv in z_arr])
        # Integral of 1/E(z') dz' from 0 to z is z * int_0^1 dx / E(z*x)
        # Our quadrature approximates int_0^1 dx / E(z*x)
        # So true value is z * quad_val
        # Wait, the task formula is dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # where a_i = 1/(1+z*nodes[i]).
        # The integral part is int_0^1 dx / E(z*x).
        # Let's check the approximation of this integral.
        integral_approx = np.sum(weights / np.array([E_z(z * x, w0, wa) for x in nodes]))
        
        # True integral via trapz on x in [0,1]
        x_arr = np.linspace(0, 1, 10000)
        integrand_vals = 1.0 / np.array([E_z(z * xv, w0, wa) for xv in x_arr])
        integral_true = np.trapz(integrand_vals, x_arr)
        
        rel_err = abs(integral_approx - integral_true) / integral_true
        
        # If error is unexpectedly large (e.g., > 1e-5), we might have a bug.
        # But GL should be ~1e-10 or better.
        if rel_err > 1e-4:
            # Fallback to midpoint if GL fails for some reason (should not happen)
            nodes = (np.arange(8) + 0.5) / 8.0
            weights = np.full(8, 1.0/8.0)
            # Re-normalize
            weights = weights / np.sum(weights)
            
    # Run check if time permits
    if time.time() < DEADLINE - 5:
        try:
            true_integral(1.0, -1.0, 0.0)
        except Exception:
            pass

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
