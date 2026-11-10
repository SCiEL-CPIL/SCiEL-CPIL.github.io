import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Start with Gauss-Legendre nodes mapped to [0,1]
    t_gl = np.array([
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425004484,
         0.1834346425004484,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763
    ])
    
    nodes_init = 0.5 * t_gl + 0.5
    weights_init = 0.5 * w_gl
    weights_init = weights_init / np.sum(weights_init)

    # Define test cosmologies and z-values for optimization
    # These cover a range of CPL parameters and redshifts
    cosmologies = [
        (0.3, 0.0, 0.0),    # LambdaCDM
        (0.3, -0.5, 0.0),   # w0=-0.5
        (0.3, -1.0, 1.0),   # w0=-1, wa=1
        (0.3, -1.0, -1.0),  # w0=-1, wa=-1
        (0.2, -1.0, 0.5),   # Om_m=0.2
        (0.4, -0.8, 0.3),   # Om_m=0.4
        (0.3, -1.2, 0.8),   # More phantom
        (0.3, -0.5, 0.5),   # Moderate
    ]
    z_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    def _E(z, w0, wa, Om_m):
        one_plus_z = 1.0 + z
        matter = Om_m * one_plus_z**3
        de_base = (1.0 - Om_m) * one_plus_z**(3.0 * (1.0 + w0))
        de_arg = 1.0 - wa * (1.0 - 1.0 / one_plus_z)
        de_arg = np.maximum(de_arg, 1e-30)
        de = de_base * de_arg**(1.5 * wa)
        return np.sqrt(matter + de)

    def _integrand(z, w0, wa, Om_m):
        if z < 1e-15:
            return 0.0
        return z / _E(z, w0, wa, Om_m)

    def _true_integral(zmax, w0, wa, Om_m):
        if zmax < 1e-15:
            return 0.0
        from scipy.integrate import quad
        result, _ = quad(_integrand, 0, zmax, args=(w0, wa, Om_m), limit=200)
        return result

    def _quadrature(zmax, nodes, weights, w0, wa, Om_m):
        # dL/(c/H0) = (1+z)*z * sum_i weights[i] / E(a_i)
        # where a_i = 1/(1 + z * nodes[i])
        # The integral is ∫_0^1 1/E(a) da with a = 1/(1+z(1-t)) ? No.
        # Contract says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # a_i = 1/(1 + z * nodes[i])
        # This implies the integral is over t in [0,1] where a = 1/(1+z(1-t)) or similar?
        # Let's check: dL = c/H0 * (1+z) * ∫_0^z dz'/E(z')
        # Change variable: a = 1/(1+z'). z' = 1/a - 1. dz' = -1/a^2 da.
        # Limits: z'=0 -> a=1, z'=z -> a=1/(1+z).
        # ∫_0^z dz'/E(z') = ∫_1^{1/(1+z)} (-1/a^2) / E(a) da = ∫_{1/(1+z)}^1 da / (a^2 E(a))
        # So dL/(c/H0) = (1+z) ∫_{1/(1+z)}^1 da / (a^2 E(a))
        # The contract says: (1+z)*z * sum w_i / E(a_i) with a_i = 1/(1+z*nodes[i]).
        # This looks like a specific quadrature rule for the integral.
        # Let's assume the formula in the contract is the definition of the quadrature.
        # We need to minimize the error between this quadrature and the true dL.
        
        a_i = 1.0 / (1.0 + zmax * nodes)
        E_ai = _E(zmax * nodes, w0, wa, Om_m) # E(z') where z' = z*nodes? No, E(a) is H(a)/H0.
        # E(a) = H(a)/H0. In terms of z, E(z) = H(z)/H0.
        # a = 1/(1+z'). So z' = 1/a - 1.
        # The integrand in the contract is 1/E(a_i).
        # a_i = 1/(1 + z*nodes[i]). So z'_i = z*nodes[i].
        # So E(a_i) = E(z'_i) = H(z'_i)/H0.
        # The sum is sum w_i / E(z'_i).
        # This approximates ∫_0^1 1/E(z*t) dt?
        # Let's just compute the quadrature value as defined.
        
        E_vals = _E(zmax * nodes, w0, wa, Om_m)
        quad_val = (1.0 + zmax) * zmax * np.sum(weights / E_vals)
        return quad_val

    def _objective(params):
        # params: 16 values, 8 nodes, 8 weights
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints softly or check
        if not np.all(np.diff(nodes) > 1e-8):
            return 1e10
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e10
        if not np.all(weights > 1e-8):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
            
        max_err = 0.0
        for w0, wa, Om_m in cosmologies:
            for z in z_values:
                true_val = _true_integral(z, w0, wa, Om_m)
                quad_val = _quadrature(z, nodes, weights, w0, wa, Om_m)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    max_err = max(max_err, rel_err)
        return max_err

    # Optimization
    from scipy.optimize import minimize
    
    # Initial guess
    x0 = np.concatenate([nodes_init, weights_init])
    
    # Try Nelder-Mead first as it doesn't require gradients and handles non-smooth max
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Run optimization
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception:
        pass

    # Refine with Powell if time permits
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_objective, best_x, method='Powell',
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_obj:
                best_x = res2.x
                best_obj = res2.fun
        except Exception:
            pass

    # Extract final nodes and weights
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Final constraint check
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
        # Fallback to initial Gauss-Legendre
        nodes = nodes_init
        weights = weights_init
        weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
