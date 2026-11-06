import numpy as np
import scipy.integrate
import scipy.optimize
import time

# EVOLVE-BLOCK-START

def E_of_a(a, Om, w0, wa):
    """
    Compute E(a) = H(a)/H0 for flat CPL model.
    Om: Omega_m
    w0, wa: CPL parameters
    a: scale factor
    """
    if np.any(a <= 0):
        # Avoid log(0)
        a = np.maximum(a, 1e-10)
    
    # Integral of (1+w(a))/a da from 0 to a
    # w(a) = w0 + wa(1-a)
    # 1+w(a) = 1+w0 + wa - wa*a
    # (1+w0+wa)/a - wa
    # Int: (1+w0+wa)ln(a) - wa*a
    # Evaluated from epsilon to a
    # Limit as eps->0: (1+w0+wa)ln(eps) - wa*eps -> -inf if 1+w0+wa > 0
    # This form is tricky for numerical integration at 0.
    # Standard formula:
    # E(a)^2 = Om a^-3 + O_L * exp(3 * Int)
    # Int = (1+w0)ln(a) + wa*(a - ln(a))
    # This assumes w(a) = w0 + wa(1-a)
    
    O_L = 1.0 - Om
    
    # To avoid issues at a=0, we can use the explicit power law form if possible,
    # but CPL is not a simple power law.
    # However, for a > 0, the formula is:
    # I(a) = (1+w0)*np.log(a) + wa*(a - np.log(a))
    # Note: at a=1, I(1) = 0.
    # E(1) = sqrt(Om + O_L) = 1. Correct.
    
    # Handle vector a
    I = (1.0 + w0) * np.log(a) + wa * (a - np.log(a))
    # Wait, if a < 1, log(a) is negative.
    # If a is small, I is large negative (if 1+w0 > 0).
    # exp(3*I) will be small.
    
    term_m = Om / (a**3)
    term_l = O_L * np.exp(3.0 * I)
    
    return np.sqrt(term_m + term_l)

def exact_dL_integral(z, Om, w0, wa):
    """
    Compute integral I = int_0^z dz'/E(z')
    using change of variable a = 1/(1+z')
    I = int_{1/(1+z)}^1 da / (a^2 E(a))
    """
    if z == 0:
        return 0.0
    
    a_min = 1.0 / (1.0 + z)
    a_max = 1.0
    
    def integrand(a):
        Ea = E_of_a(a, Om, w0, wa)
        return 1.0 / (a**2 * Ea)
    
    # Use quad for high accuracy
    res, err = scipy.integrate.quad(integrand, a_min, a_max, limit=100)
    return res

def approx_dL_integral(z, nodes, weights, Om, w0, wa):
    """
    Compute approximate integral using quadrature rule.
    nodes: x_i in [0,1]
    weights: w_i
    """
    # Map x to z' = z * x
    # a_i = 1 / (1 + z * x_i)
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = E_of_a(a_vals, Om, w0, wa)
    # Integrand in x space: 1/E(z x) * z ?
    # Wait.
    # Original: dL = (c/H0) (1+z) int_0^z dz'/E(z')
    # Change var z' = z x, dz' = z dx
    # int_0^z dz'/E(z') = z int_0^1 dx/E(z x)
    # So we approximate int_0^1 dx/E(z x)
    # Sum w_i / E(z x_i)
    # The factor z is outside the sum in the prompt formula:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # So we return sum_i w_i / E(a_i)
    # The caller (evaluator) multiplies by (1+z)*z.
    # But for error calculation, we compare the integral part.
    # Exact integral part: I_exact = int_0^z dz'/E(z')
    # Approx integral part: I_approx = z * sum_i w_i / E(a_i)
    # We want I_approx to match I_exact.
    
    val = np.sum(weights / E_vals)
    return z * val

def construct():
    start_time = time.time()
    deadline = 80.0 # Safety margin
    
    # Training Set
    # Om, w0, wa, z
    Om_vals = [0.25, 0.3, 0.35]
    w0_vals = [-1.0, -0.8]
    wa_vals = [0.0, 0.5]
    z_vals = [0.1, 0.5, 1.0, 3.0, 10.0]
    
    # Pre-compute exact integrals
    train_points = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for z in z_vals:
                    try:
                        I_exact = exact_dL_integral(z, Om, w0, wa)
                        if I_exact > 1e-6: # Avoid division by zero for very small z
                            train_points.append((z, Om, w0, wa, I_exact))
                    except Exception:
                        pass
                        
    if not train_points:
        # Fallback to midpoint
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    n_nodes = 8
    
    def unpack_params(params):
        # params: 16 floats
        # First 8: v for nodes (sigmoid)
        # Next 8: u for weights (softmax)
        v = params[:n_nodes]
        u = params[n_nodes:]
        
        # Nodes
        x = 1.0 / (1.0 + np.exp(-v))
        # Weights
        # Shift u to prevent overflow
        u_max = np.max(u)
        w = np.exp(u - u_max)
        w = w / np.sum(w)
        
        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        w = w[sort_idx]
        
        return x, w

    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop optimization
        
        x, w = unpack_params(params)
        
        max_err = 0.0
        for z, Om, w0, wa, I_exact in train_points:
            I_approx = approx_dL_integral(z, x, w, Om, w0, wa)
            if I_exact == 0:
                err = 0.0
            else:
                err = abs(I_approx - I_exact) / I_exact
            if err > max_err:
                max_err = err
                
        return max_err

    # Initial guess
    # Midpoint nodes: 0.0625, 0.1875, ...
    x_init = (np.arange(n_nodes) + 0.5) / n_nodes
    # Invert sigmoid: v = log(x/(1-x))
    x_init_clipped = np.clip(x_init, 1e-6, 1-1e-6)
    v_init = np.log(x_init_clipped / (1 - x_init_clipped))
    
    # Weights equal
    w_init = np.ones(n_nodes) / n_nodes
    # Invert softmax: u = log(w) + c
    u_init = np.log(w_init)
    
    params_init = np.concatenate([v_init, u_init])
    
    # Optimize
    best_params = params_init
    best_score = objective(params_init)
    
    # Use Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            objective, 
            params_init, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 5:
        # Return best found so far
        x_final, w_final = unpack_params(best_params)
        return {"nodes": x_final, "weights": w_final}

    # Refine with L-BFGS-B if time permits? 
    # Nelder-Mead might be stuck. Let's try a second pass with different init or method.
    # But 80s limit is tight.
    
    x_final, w_final = unpack_params(best_params)
    
    # Ensure strict increasing and positive weights
    # x from sigmoid is strictly increasing if v are distinct. 
    # If v are close, x might be very close. 
    # Sorting ensures increasing order.
    # Weights are positive by construction.
    # Sum to 1 by construction.
    
    return {"nodes": x_final, "weights": w_final}

# EVOLVE-BLOCK-END
