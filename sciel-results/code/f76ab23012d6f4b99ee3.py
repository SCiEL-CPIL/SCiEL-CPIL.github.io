import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = _T0 + 80.0  # Safety margin

# Pre-compute true values for a grid of validation points
# We assume the validation set covers a reasonable range of CPL parameters and redshifts.
# z in [0.1, 3.0], w0 in [-1.5, 0.5], wa in [-1.0, 1.0]
_Z_GRID = np.linspace(0.1, 3.0, 10)
_W0_GRID = np.linspace(-1.5, 0.5, 7)
_WA_GRID = np.linspace(-1.0, 1.0, 7)

# Flatten the grid
_z_flat = []
_w0_flat = []
_wa_flat = []
_true_vals = []

# High accuracy reference integration
_N_REF = 100
_x_leg, _w_leg = np.polynomial.legendre.leggauss(_N_REF)

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat universe with matter + CPL dark energy."""
    Om = 0.3
    Ode = 1.0 - Om
    term_m = Om / (a ** 3)
    # CPL: w(a) = w0 + wa * (1-a)
    # rho_de(a) = rho_de0 * a^{-3(1+w0+wa)} * exp(3*wa*(1-a))
    exponent = 3.0 * wa * (a - 1.0) + 3.0 * (w0 + wa) * (1.0 - a)
    term_de = Ode * np.exp(exponent)
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))

def _compute_true_values():
    """Compute true integral values for the grid."""
    global _z_flat, _w0_flat, _wa_flat, _true_vals
    _z_flat = []
    _w0_flat = []
    _wa_flat = []
    _true_vals = []
    
    for z in _Z_GRID:
        for w0 in _W0_GRID:
            for wa in _WA_GRID:
                _z_flat.append(z)
                _w0_flat.append(w0)
                _wa_flat.append(wa)
                
                # Integral: (1+z) * integral_0^z dz' / E(z')
                # Change variable u = z'/z, z' = u*z, dz' = z*du
                # a(u) = 1 / (1 + u*z)
                # Integral = (1+z) * z * integral_0^1 du / E(a(u))
                
                a_pts = 1.0 / (1.0 + _x_leg * z) # _x_leg is in [-1, 1], so u in [0,1] needs mapping
                # leggauss nodes are in [-1, 1]. Map to [0, 1]: u = 0.5*(x+1)
                u_pts = 0.5 * (_x_leg + 1.0)
                a_pts = 1.0 / (1.0 + u_pts * z)
                
                E_vals = _E_of_a(a_pts, w0, wa)
                integrand_vals = 1.0 / E_vals
                
                # Gauss-Legendre weights need scaling for [0,1]
                w_scaled = 0.5 * _w_leg
                integral_u = np.sum(w_scaled * integrand_vals)
                
                true_val = (1.0 + z) * z * integral_u
                _true_vals.append(true_val)
                
_z_flat = np.array(_z_flat)
_w0_flat = np.array(_w0_flat)
_wa_flat = np.array(_wa_flat)
_true_vals = np.array(_true_vals)

def _objective(params):
    """
    Objective function for differential evolution.
    params: [u1, ..., u8, ln_w1, ..., ln_w8]
    """
    if time.time() > _DEADLINE - 2.0:
        return 1e10
        
    u = params[:8]
    ln_w = params[8:]
    
    # Check node ordering and bounds
    if np.any(u <= 0) or np.any(u >= 1):
        return 1e10
    if np.any(np.diff(u) <= 1e-12):
        return 1e10
        
    w = np.exp(ln_w)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return 1e10
    w = w / w_sum
    
    # Compute quadrature values for all grid points
    # Vectorized computation
    # u is (8,), z is (N,), w0 is (N,), wa is (N,)
    # We need to compute for each grid point i:
    # I_quad_i = (1+z_i) * z_i * sum_j w_j / E(a_{ij})
    # where a_{ij} = 1 / (1 + u_j * z_i)
    
    # a shape: (N, 8)
    # z_i[:, None] shape (N, 1)
    # u[None, :] shape (1, 8)
    a_grid = 1.0 / (1.0 + _z_flat[:, None] * u[None, :])
    
    # E_grid shape (N, 8)
    # _E_of_a needs to handle arrays.
    Om = 0.3
    Ode = 1.0 - Om
    term_m = Om / (a_grid ** 3)
    exponent = 3.0 * _wa_flat[:, None] * (a_grid - 1.0) + 3.0 * (_w0_flat[:, None] + _wa_flat[:, None]) * (1.0 - a_grid)
    term_de = Ode * np.exp(exponent)
    E_sq = term_m + term_de
    E_grid = np.sqrt(np.maximum(E_sq, 1e-30))
    
    inv_E = 1.0 / E_grid
    # sum over nodes (axis=1)
    # w shape (8,) -> broadcast to (N, 8)
    quad_integrals = np.sum(inv_E * w[None, :], axis=1)
    I_quad = (1.0 + _z_flat) * _z_flat * quad_integrals
    
    # Relative error
    rel_err = np.abs(I_quad - _true_vals) / _true_vals
    max_err = np.max(rel_err)
    
    return max_err

def construct():
    t_start = time.time()
    
    # Initial bounds
    # Nodes in (0, 1)
    u_bounds = [(1e-4, 1 - 1e-4)] * 8
    # Log weights in (-10, 10)
    w_bounds = [(-10, 10)] * 8
    bounds = u_bounds + w_bounds
    
    # Use differential evolution
    # popsize=15, maxiter=50, tol=1e-6
    # This might be too slow. Let's try a smaller popsize and fewer iterations, then refine.
    
    try:
        # DE is stochastic. Set seed for reproducibility? No, we want best result.
        # However, DE can be slow. 
        # Let's try a hybrid: start with a few good initial guesses and use local optimization.
        
        # Initial guess 1: Gauss-Legendre
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u_init = 0.5 * (gl_nodes + 1.0)
        w_init = 0.5 * gl_weights
        ln_w_init = np.log(w_init)
        params_init = np.concatenate([u_init, ln_w_init])
        
        # Initial guess 2: Equally spaced
        u_eq = (np.arange(1, 9) - 0.5) / 8.0
        w_eq = np.ones(8) / 8.0
        ln_w_eq = np.log(w_eq)
        params_eq = np.concatenate([u_eq, ln_w_eq])
        
        # Evaluate initial guesses
        err_init = _objective(params_init)
        err_eq = _objective(params_eq)
        
        best_params = params_init if err_init < err_eq else params_eq
        best_err = min(err_init, err_eq)
        
        # Run Differential Evolution
        # To save time, use a smaller population and fewer iterations, but with a good start?
        # DE doesn't use a single start, it uses a population.
        # Let's run DE for a limited number of iterations.
        
        # Time check
        if time.time() < _DEADLINE - 30.0:
            try:
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    seed=42, 
                    maxiter=20, 
                    popsize=10, 
                    tol=1e-4,
                    init='sobol'
                )
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        # Refine with local optimization (Nelder-Mead or Powell)
        # Powell is good for non-smooth functions (like the max error).
        if time.time() < _DEADLINE - 5.0:
            try:
                res2 = minimize(
                    _objective, 
                    best_params, 
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res2.fun < best_err:
                    best_err = res2.fun
                    best_params = res2.x
            except Exception:
                pass
        
        # Extract nodes and weights
        u_final = best_params[:8]
        ln_w_final = best_params[8:]
        w_final = np.exp(ln_w_final)
        w_final = w_final / np.sum(w_final)
        
        # Ensure constraints
        u_final = np.sort(u_final)
        # Check strictly increasing
        if not np.all(np.diff(u_final) > 0):
            # Fallback to initial guess
            u_final = u_init
            w_final = w_init
            
        # Final check
        if np.any(u_final <= 0) or np.any(u_final >= 1) or np.any(w_final <= 0):
            u_final = u_init
            w_final = w_init

        return {"nodes": u_final, "weights": w_final}
        
    except Exception:
        # Fallback
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u_final = 0.5 * (gl_nodes + 1.0)
        w_final = 0.5 * gl_weights
        return {"nodes": u_final, "weights": w_final}

# Run pre-computation at module load
_compute_true_values()
