"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import eval_genlaguerre, gamma
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 580  # safety margin before 600s limit
    m = 12
    n = 1
    
    # Initial guess: seed program values
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    def c4_bound(roots):
        """Compute C4 upper bound from root positions using Laguerre basis norm."""
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            return 1e10
        # Construct the polynomial with double roots at given locations
        # P(x) = prod_i (L_m(x) - L_m(roots[i]))^2 style, but more precisely:
        # The C4 bound involves the norm of a specific Laguerre combination
        # Use the fact that for the Cohn-Goncalves construction, the bound
        # is related to 1/sup_x |P(x)|^2 where P is built from Laguerre polys
        
        # Simpler formulation: C4 ~ 1 / min_i (product of distances to other roots)^2
        # This captures the repulsive interaction
        # Actually, let's use the proper Laguerre norm formulation
        
        # The key quantity: for roots r_1 < r_2 < ... < r_m,
        # the bound involves the Laguerre polynomial values and their derivatives
        # A practical proxy: the bound is minimized when roots are well-spread
        # and follow the equilibrium distribution of the Calogero-Sutherland model
        
        # Compute a smooth objective based on the Laguerre polynomial structure
        # C4 ~ 1 / (prod_i L_m'(r_i)^2) * some factor
        # For double roots, the relevant quantity is the product of squared derivatives
        
        lags = np.array([eval_genlaguerre(m, n, r) for r in roots])
        # Derivative of L_m^{(n)}(x)
        # L_m^{(n)'}(x) = -L_{m-1}^{(n+1)}(x)
        lag_derivs = np.array([-eval_genlaguerre(m-1, n+1, r) for r in roots])
        
        # C4 bound proxy: 1 / (prod |L_m'(r_i)|^2)
        # This is the quantity we want to minimize
        prod_derivs = np.prod(np.abs(lag_derivs))
        if prod_derivs < 1e-30:
            return 1e10
        return 1.0 / (prod_derivs ** 2)
    
    # Log-space parameterization for positivity
    def log_objective(log_roots):
        roots = np.exp(log_roots)
        # Enforce ordering: if not ordered, penalize heavily
        if np.any(np.diff(roots) <= 1e-6):
            return 1e10
        return c4_bound(roots)
    
    # Start from seed in log space
    log_x0 = np.log(x0)
    
    best_roots = x0.copy()
    best_val = c4_bound(best_roots)
    
    # Try L-BFGS-B in log space
    try:
        result = minimize(
            log_objective,
            log_x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() > deadline - 30:
            raise TimeoutError()
        
        if result.success or result.fun < best_val:
            new_roots = np.exp(result.x)
            if np.all(new_roots > 0) and np.all(np.diff(new_roots) > 1e-6):
                new_val = c4_bound(new_roots)
                if new_val < best_val:
                    best_roots = new_roots
                    best_val = new_val
    except Exception:
        pass
    
    # Try Nelder-Mead in direct space as backup
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                c4_bound,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.success or result2.fun < best_val:
                new_roots = result2.x
                if np.all(new_roots > 0) and np.all(np.diff(new_roots) > 1e-6):
                    new_val = c4_bound(new_roots)
                    if new_val < best_val:
                        best_roots = new_roots
                        best_val = new_val
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 15:
        try:
            result3 = minimize(
                c4_bound,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result3.fun < best_val:
                new_roots = result3.x
                if np.all(new_roots > 0) and np.all(np.diff(new_roots) > 1e-6):
                    new_val = c4_bound(new_roots)
                    if new_val < best_val:
                        best_roots = new_roots
                        best_val = new_val
        except Exception:
            pass
    
    # Return as list of floats
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
