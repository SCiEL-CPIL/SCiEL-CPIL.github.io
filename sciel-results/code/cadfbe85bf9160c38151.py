"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to stay well under 90s

    n = 8

    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes0 = 0.5 * (x_gl + 1.0)
    nodes0.sort()
    weights0 = w_gl / 2.0

    # Define the integrand for a given CPL model (w0, wa)
    # E(a) = (a^(-3(1+w0+wa)) + a^(-3))^(1/2) for flat CDMDM
    # Actually E^2 = Omega_m a^-3 + Omega_de a^(-3(1+w0+wa))
    # Omega_m = 0.3, Omega_de = 0.7
    Omega_m = 0.3
    Omega_de = 0.7

    def E_sq(a, w0, wa):
        return Omega_m * a**(-3) + Omega_de * a**(-3 * (1 + w0 + wa))

    def integrand(z, nodes, w0, wa):
        a = 1.0 / (1.0 + z * nodes)
        return (1.0 + z) * z / np.sqrt(E_sq(a, w0, wa))

    # Reference integration: use a high-order Gauss-Legendre rule (e.g., 64 points)
    n_ref = 64
    x_ref, w_ref = np.polynomial.legendre.leggauss(n_ref)
    z_ref = 0.5 * (x_ref + 1.0)
    w_ref = w_ref / 2.0

    def reference_integral(w0, wa):
        vals = integrand(z_ref, np.ones(n_ref), w0, wa)  # dummy nodes, we compute directly
        # Wait, the integrand depends on nodes. Let me re-read the formula.
        # dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
        # This is a quadrature approximation. The "true" value is the integral:
        # dL/(c/H0) = integral_0^1 (1+z) z / E(a(z)) dz
        # So the reference should be the integral.
        a_ref = 1.0 / (1.0 + z_ref)
        E_ref = np.sqrt(E_sq(a_ref, w0, wa))
        f_ref = (1.0 + z_ref) * z_ref / E_ref
        return np.sum(w_ref * f_ref)

    # Sample some CPL models to define the objective
    # We want to minimize the worst-case relative error over a set of models
    # Let's sample a grid of (w0, wa)
    w0_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    wa_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    models = [(w0, wa) for w0 in w0_vals for wa in wa_vals]

    def objective(params):
        if time.time() > deadline:
            return 1e10  # penalty for timeout

        nodes = params[:n]
        weights = params[n:]

        # Enforce constraints: nodes in [0,1] strictly increasing, weights > 0, sum(weights)=1
        # For optimization, we can use log-weights or just check
        # Let's use a penalty approach or transform variables
        # Transform: nodes = sigmoid(x) * (1-eps) + eps, weights = softmax(y)
        # But the current params are raw. Let's assume params are already transformed.
        # Actually, let's transform inside the objective or use a different parameterization.
        
        # Let's use the raw params but penalize violations
        # Nodes: 0 < nodes[0] < ... < nodes[n-1] < 1
        # Weights: weights > 0, sum = 1
        
        # Check nodes
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
            
        # Check weights
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10

        # Compute relative error for each model
        max_err = 0.0
        for w0, wa in models:
            ref_val = reference_integral(w0, wa)
            a_i = 1.0 / (1.0 + z_ref * nodes)  # Wait, z_ref is the reference grid, not the quadrature nodes
            # The quadrature is: sum_i weights[i] * (1+z) * z / E(a_i) where z is the variable?
            # No, the formula is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
            # This looks like the integrand is f(z) = (1+z)*z/E(a(z)) and we approximate integral f(z) dz
            # But the formula has (1+z)*z outside the sum. That's strange.
            # Let me re-read: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
            # This implies the integrand is g(z) = 1/E(a(z)) and the measure is (1+z)z dz?
            # Or is it that the integrand is (1+z)z/E(a(z)) and the quadrature is sum w_i f(z_i)?
            # Standard quadrature: integral f(z) dz ≈ sum w_i f(z_i).
            # If f(z) = (1+z)z/E(a(z)), then integral ≈ sum w_i (1+z_i)z_i/E(a_i).
            # The formula says (1+z)*z * sum w_i/E(a_i). This would mean f(z) = (1+z)z and the quadrature is for 1/E(a)?
            # No, that doesn't make sense because (1+z)z depends on z.
            # Let's assume the standard interpretation: the integrand is f(z) = (1+z)z/E(a(z)) and we use a quadrature rule sum w_i f(z_i).
            # But the formula explicitly writes (1+z)*z outside. Maybe it's a typo in my understanding or the task?
            # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
            # If this is the formula, then for a given z, the value is (1+z)z * S(z) where S(z) = sum w_i/E(a_i).
            # But a_i = 1/(1+z*nodes[i]). So S(z) depends on z.
            # This is not a standard quadrature for the integral.
            # Let's look at the standard luminosity distance formula:
            # d_L = (1+z) * c/H0 * integral_0^z dz'/E(z')
            # Let u = z'/z, then dz' = z du.
            # d_L = (1+z) * c/H0 * z * integral_0^1 du / E(z*u)
            # So the integral is I = integral_0^1 du / E(z*u).
            # The formula given is: (1+z)*z * sum_i w_i / E(a_i) where a_i = 1/(1+z*nodes[i]).
            # So E(a_i) = E(z*nodes[i])? No, E(a) is defined as H(a)/H0.
            # In cosmology, E(z) = H(z)/H0. E(a) = H(a)/H0.
            # a = 1/(1+z). So a_i = 1/(1+z*nodes[i]).
            # This means the variable in the integrand is a, not z.
            # Let's change variable from z to a.
            # z = 1/a - 1. dz = -1/a^2 da.
            # Integral from z=0 to z=z_max becomes a=1 to a=1/(1+z_max).
            # But the formula has nodes in [0,1].
            # If the nodes are in z-space, a_i = 1/(1+z_i).
            # If the nodes are in a-space, a_i = nodes[i].
            # The task says: nodes in [0,1], a_i = 1/(1+z*nodes[i]).
            # This implies the nodes are in a transformed space, let's call it u.
            # u in [0,1]. a = 1/(1+z*u).
            # When u=0, a=1 (z=0). When u=1, a=1/(1+z) (z=z_max).
            # So the integral is over u from 0 to 1.
            # The integrand in u is: 1/E(a(u)).
            # And the prefactor is (1+z)z.
            # So the quadrature approximates integral_0^1 du / E(a(u)).
            # The quadrature is sum_i w_i / E(a_i).
            # So we are approximating J(z) = integral_0^1 du / E(1/(1+z*u)).
            # The "true" value is this integral.
            # The reference should compute this integral accurately.

            # Let's re-calculate the reference for this interpretation.
            # J(z) = integral_0^1 du / E(1/(1+z*u)).
            # We need to evaluate this for a given z. But the task says "Minimize the worst relative error against independent adaptive integration on disjoint public validation cosmologies."
            # And "exactly eight integrand samples per distance."
            # This implies we compute a distance for a given z (or a given cosmology model and a set of redshifts?).
            # "The evaluator computes integrands itself."
            # "Minimize worst relative validation error in percent"
            # It likely means: for a set of test cases (cosmology models + redshifts), compute the distance using our rule and compare to a high-precision reference.
            
            # Let's assume the test cases are a grid of (w0, wa, z_max).
            # For each case, compute d_L_approx = (1+z_max)*z_max * sum_i w_i / E(a_i)
            # and d_L_ref = (1+z_max)*z_max * integral_0^1 du / E(1/(1+z_max*u))
            
            # Let's define the reference integral function.
            def ref_J(z_max, w0, wa):
                # Integrate u from 0 to 1 of 1/E(1/(1+z_max*u)) du
                # Use Gauss-Legendre with 64 points
                u_vals = z_ref  # 0.5*(x+1) is in [0,1]
                a_vals = 1.0 / (1.0 + z_max * u_vals)
                E_vals = np.sqrt(E_sq(a_vals, w0, wa))
                f_vals = 1.0 / E_vals
                return np.sum(w_ref * f_vals)

            # Now, for a given z_max, compute the approximate J
            a_i = 1.0 / (1.0 + z_max * nodes)
            E_i = np.sqrt(E_sq(a_i, w0, wa))
            J_approx = np.sum(weights / E_i)
            
            J_ref = ref_J(z_max, w0, wa)
            
            rel_err = abs(J_approx - J_ref) / J_ref
            if rel_err > max_err:
                max_err = rel_err

        return max_err

    # We need to optimize nodes and weights.
    # To ensure positivity and sum=1, let's parameterize weights as exp(y) / sum(exp(y)).
    # And nodes as sigmoid(x) * (1-2*eps) + eps to keep them in (eps, 1-eps).
    # But the objective currently takes raw nodes and weights.
    # Let's change the approach: optimize over x and y, and transform inside.

    def objective_transformed(params):
        if time.time() > deadline:
            return 1e10

        x = params[:n]
        y = params[n:]

        eps = 1e-6
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = eps + nodes * (1.0 - 2.0 * eps)
        
        # Ensure strictly increasing? Sigmoid is monotonic, so if x is sorted, nodes are sorted.
        # But x is not necessarily sorted. We can sort x, but that breaks differentiability.
        # Instead, we can penalize if nodes are not sorted.
        if np.any(np.diff(nodes) <= 0):
            return 1e10

        weights = np.exp(y)
        weights = weights / np.sum(weights)

        # Now compute the objective as before
        max_err = 0.0
        # Sample some z_max values
        z_max_vals = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        
        for w0, wa in models:
            for z_max in z_max_vals:
                a_i = 1.0 / (1.0 + z_max * nodes)
                E_i = np.sqrt(E_sq(a_i, w0, wa))
                J_approx = np.sum(weights / E_i)
                
                # Reference
                u_vals = z_ref
                a_ref = 1.0 / (1.0 + z_max * u_vals)
                E_ref = np.sqrt(E_sq(a_ref, w0, wa))
                f_ref = 1.0 / E_ref
                J_ref = np.sum(w_ref * f_ref)
                
                rel_err = abs(J_approx - J_ref) / J_ref
                if rel_err > max_err:
                    max_err = rel_err

        return max_err

    # Initial guess for x and y
    x0 = np.log(nodes0 / (1 - nodes0))  # Inverse sigmoid
    y0 = np.log(weights0)

    result = minimize(objective_transformed, np.concatenate([x0, y0]), method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})

    if time.time() > deadline:
        # Return the initial guess if we timed out
        return {"nodes": nodes0, "weights": weights0}

    x_opt = result.x[:n]
    y_opt = result.x[n:]

    eps = 1e-6
    nodes_opt = 1.0 / (1.0 + np.exp(-x_opt))
    nodes_opt = eps + nodes_opt * (1.0 - 2.0 * eps)
    nodes_opt.sort()

    weights_opt = np.exp(y_opt)
    weights_opt = weights_opt / np.sum(weights_opt)

    # Ensure weights sum to 1 within 1e-12
    weights_opt = weights_opt / np.sum(weights_opt)

    return {"nodes": nodes_opt, "weights": weights_opt}
# EVOLVE-BLOCK-END
