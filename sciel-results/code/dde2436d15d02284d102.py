import numpy as np
from scipy.optimize import minimize

def construct():
    # 1. Define training set
    training_cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.0),
        (0.3, -1.0, 1.0),
        (0.3, -1.0, -1.0),
        (0.25, -0.95, 0.2),
    ]
    
    z_grid = np.logspace(-2, 1, 50)
    
    # 2. Pre-compute "true" integrals using high-precision quadrature (100 pts GL)
    # We need E(z) for arbitrary z.
    def E_of_z(z, om, w0, w1):
        a = 1.0 / (1.0 + z)
        pow_m = (1.0 + z) ** 3
        exp_de = 3.0 * (1.0 + w0 + w1 * (1.0 - a))
        pow_de = (1.0 + z) ** exp_de
        od = 1.0 - om
        return np.sqrt(om * pow_m + od * pow_de)
        
    true_integrals = np.zeros((len(training_cosmologies), len(z_grid)))
    # Use 100-point GL for reference
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(100)
    # Map to [0,1] for u variable
    nodes_ref_u = (nodes_ref + 1) / 2
    weights_ref_u = weights_ref * 0.5
    
    for i, (om, w0, w1) in enumerate(training_cosmologies):
        for j, z in enumerate(z_grid):
            if z < 1e-6:
                true_integrals[i, j] = 0.0
                continue
            # Integrand in u: z / E(z*u)
            # u = nodes_ref_u
            # z_prime = z * u
            z_prime = z * nodes_ref_u
            E_vals = E_of_z(z_prime, om, w0, w1)
            integrand = z / E_vals
            true_integrals[i, j] = np.sum(weights_ref_u * integrand)
            
    # 3. Objective function using pre-computed true integrals
    def objective(params):
        # Decode params
        x = np.exp(params[:9])
        x_sum = np.sum(x)
        x_norm = x / x_sum
        nodes = np.cumsum(x_norm)[:8]
        
        y = np.exp(params[9:17])
        w_sum = np.sum(y)
        weights = y / w_sum
        
        max_err = 0.0
        # Vectorized calculation over all cosmologies and z
        # nodes shape (8,), weights shape (8,)
        # true_integrals shape (N_cosmo, N_z)
        
        # For each cosmology i and z j:
        # approx = z * sum(weights / E(z * nodes))
        
        # We can vectorize the E calculation.
        # z_grid shape (N_z,)
        # nodes shape (8,)
        # z_prime shape (N_z, 8)
        z_prime = np.outer(z_grid, nodes) # (N_z, 8)
        
        # E_vals shape (N_cosmo, N_z, 8)
        # We need to loop over cosmologies because parameters differ
        for i, (om, w0, w1) in enumerate(training_cosmologies):
            E_vals = E_of_z(z_prime, om, w0, w1) # (N_z, 8)
            # approx shape (N_z,)
            approx = z_grid * np.sum(weights / E_vals, axis=1)
            
            true_vals = true_integrals[i]
            
            # Avoid division by zero
            mask = true_vals > 1e-10
            if np.any(mask):
                rel_errs = np.abs((approx[mask] - true_vals[mask]) / true_vals[mask])
                curr_max = np.max(rel_errs)
                if curr_max > max_err:
                    max_err = curr_max
                    
        return max_err

    # 4. Initial guess
    params_init = np.zeros(17)
    
    # 5. Optimize
    res = minimize(objective, params_init, method='Powell', 
                   options={'maxiter': 500, 'ftol': 1e-10})
    
    # Decode final result
    x = np.exp(res.x[:9])
    x_sum = np.sum(x)
    x_norm = x / x_sum
    nodes_opt = np.cumsum(x_norm)[:8]
    
    y = np.exp(res.x[9:17])
    w_sum = np.sum(y)
    weights_opt = y / w_sum
    
    # Final sanity checks
    # Ensure nodes are strictly increasing and in (0,1)
    # The parameterization guarantees this.
    
    return {"nodes": nodes_opt, "weights": weights_opt}
