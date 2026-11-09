import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl_exact(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for exact CPL model.
    a is a numpy array or scalar.
    w0, wa are scalars or arrays broadcastable to a.
    """
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    
    D = 1.0 + w0 + wa
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    power_term = a**(-3.0 * D)
    term_de = Om_de * power_term * exp_term
    
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_vectorized(Z, w0, wa, num_points=256):
    """
    Compute true dL for vector of Z, w0, wa.
    Returns array of true dL values.
    """
    # Z, w0, wa are 1D arrays of length N
    N = len(Z)
    if N == 0:
        return np.array([])
    
    # Create a grid for integration
    # We need to integrate from 0 to Z_k for each k.
    # This is tricky with a single grid if Z varies.
    # We can use a normalized grid u in [0,1] and integrate \int_0^1 f(u * Z_k) * Z_k du
    # Or just use a dense grid in z and truncate?
    # Better: use adaptive or high-res grid for each Z?
    # Vectorized approach:
    # For each k, we need \int_0^{Z_k} dz/E(z).
    # Let z = u * Z_k. dz = Z_k du.
    # Integral = Z_k * \int_0^1 du / E(u * Z_k).
    
    u_grid = np.linspace(0, 1, num_points)
    
    # Z is (N,), u_grid is (M,)
    # Z_k * u_grid is (N, M)
    Z_vals = Z[:, np.newaxis]
    u_vals = u_grid[np.newaxis, :]
    z_vals = Z_vals * u_vals
    
    a_vals = 1.0 / (1.0 + z_vals)
    
    # w0, wa are (N,)
    w0_vals = w0[:, np.newaxis]
    wa_vals = wa[:, np.newaxis]
    
    E_vals = _E_cpl_exact(a_vals, w0_vals, wa_vals)
    integrand = 1.0 / E_vals
    
    # Integrate over u (from 0 to 1)
    # np.trapz(integrand, u_grid, axis=1) gives \int_0^1 ... du
    integral_u = np.trapz(integrand, u_grid, axis=1)
    
    # Multiply by Z_k
    true_vals = Z_vals.flatten() * integral_u
    # Multiply by (1+Z_k)
    true_vals = (1.0 + Z_vals.flatten()) * true_vals
    
    return true_vals

def _quad_dL_vectorized(Z, w0, wa, nodes, weights):
    """
    Compute quadrature dL for vector of Z, w0, wa.
    """
    N = len(Z)
    if N == 0:
        return np.array([])
        
    # nodes is (8,), weights is (8,)
    # Z is (N,)
    
    # a_i = 1 / (1 + z * x_i)
    # z is (N,), nodes is (8,)
    # We want (N, 8)
    Z_vals = Z[:, np.newaxis]
    nodes_arr = nodes[np.newaxis, :]
    z_nodes = Z_vals * nodes_arr
    
    a_vals = 1.0 / (1.0 + z_nodes)
    
    # w0, wa are (N,)
    w0_vals = w0[:, np.newaxis]
    wa_vals = wa[:, np.newaxis]
    
    E_vals = _E_cpl_exact(a_vals, w0_vals, wa_vals)
    
    # weights is (8,)
    weights_arr = weights[np.newaxis, :]
    
    # sum(weights / E_vals, axis=1)
    sum_term = np.sum(weights_arr / E_vals, axis=1)
    
    quad_vals = (1.0 + Z_vals.flatten()) * Z_vals.flatten() * sum_term
    
    return quad_vals

def _decode_params(params):
    """
    Decode 17 parameters into 8 nodes and 8 weights.
    params[0:9] -> gaps t_0..t_8
    params[9:17] -> weights v_1..v_8
    """
    t = params[:9]
    t_max = np.max(t)
    exp_t = np.exp(t - t_max)
    gaps = exp_t / np.sum(exp_t)
    
    nodes = np.cumsum(gaps)
    nodes = nodes[:8] # Ensure length 8
    
    v = params[9:17]
    v_max = np.max(v)
    exp_v = np.exp(v - v_max)
    weights = exp_v / np.sum(exp_v)
    
    return nodes, weights

# Test set
w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
z_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Create meshgrid
W0_grid, WA_grid, Z_grid = np.meshgrid(w0_vals, wa_vals, z_vals, indexing='ij')
W0_flat = W0_grid.flatten()
WA_flat = WA_grid.flatten()
Z_flat = Z_grid.flatten()

N_TEST = len(Z_flat)

# Pre-compute true values? 
# The true values depend only on w0, wa, z.
# So we can compute them once!
TRUE_VALS = _true_dL_vectorized(Z_flat, W0_flat, WA_flat, num_points=512)

def _objective(params):
    if _time_left() < 1.0:
        return 1e10
        
    nodes, weights = _decode_params(params)
    
    quad_vals = _quad_dL_vectorized(Z_flat, W0_flat, WA_flat, nodes, weights)
    
    # Avoid division by zero
    rel_errs = np.abs(quad_vals - TRUE_VALS) / np.maximum(TRUE_VALS, 1e-10)
    
    return np.max(rel_errs)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess: Uniform gaps and uniform weights
    x0 = np.zeros(17)
    
    bounds = [(-10, 10)] * 17
    
    best_result = None
    best_score = 1e10
    
    # Phase 1: Differential Evolution
    # Since objective is vectorized and fast, we can run longer
    try:
        if _time_left() > 20:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=10, 
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        pass
    
    # Phase 2: Local Refinement
    if best_result is not None and _time_left() > 5:
        try:
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
            pass
            
    # Fallback to Gauss-Legendre if optimization fails
    if best_result is None:
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (x_gl + 1.0)
        weights = 0.5 * w_gl
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}

    nodes, weights = _decode_params(best_result)
    
    # Final check
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
