"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.integrate

# Start time for deadline tracking
_START_TIME = time.time()
_DEADLINE = 170.0  # 170s to be safe within 180s

def _hermite_polynomials(x):
    """
    Compute H0, H4, H8, H12 for input array x.
    Physicist's Hermite polynomials.
    """
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 4608*x**6 + 23040*x**4 - 34560*x**2 + 1344
    H12 = 4096*x**12 - 184320*x**10 + 3225600*x**8 - 22809600*x**6 + 7077880*x**4 - 84672000*x**2 + 467712
    return H0, H4, H8, H12

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound J = (||g||_1)^2 / ||g||_2^2
    for g(x) = exp(-x^2/2) * (c0*H0 + c4*H4 + c8*H8 + c12*H12)
    with c12 determined by g(0)=0.
    """
    # Calculate c12 such that g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=467712
    c12 = -(c0 + 12*c4 + 1344*c8) / 467712.0
    
    def integrand_abs(x):
        H0, H4, H8, H12 = _hermite_polynomials(x)
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = np.exp(-x**2 / 2.0) * P
        return np.abs(g)
        
    def integrand_sq(x):
        H0, H4, H8, H12 = _hermite_polynomials(x)
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = np.exp(-x**2 / 2.0) * P
        return g**2

    # Integrate over a sufficiently large range
    # Gaussian decays fast, +/- 10 is plenty for double precision
    L1, _ = scipy.integrate.quad(integrand_abs, -10, 10, limit=200)
    L2_sq, _ = scipy.integrate.quad(integrand_sq, -10, 10, limit=200)
    
    if L2_sq < 1e-15:
        return 1e10 # Avoid division by zero
        
    return (L1 * L1) / L2_sq

def construct():
    # Seed values from the current program
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_coeffs = seed_coeffs
    best_score = _compute_bound(*seed_coeffs)
    
    # Optimization using L-BFGS-B
    # We optimize c0, c4, c8
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    # Try to improve from the seed
    # Set a time limit for the optimizer
    start_opt = time.time()
    max_opt_time = 120.0 # 120 seconds for optimization
    
    try:
        # Use L-BFGS-B, which handles bounds well and is efficient for smooth problems
        # No explicit bounds needed, but we can add loose ones if necessary
        res = scipy.optimize.minimize(
            objective, 
            seed_coeffs, 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
            
    except Exception:
        pass
        
    # If we still have time, try a few random restarts
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 20:
        np.random.seed(42)
        for _ in range(5):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            # Random perturbation around seed or uniform
            # Uniform in a reasonable range
            random_coeffs = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-1, 1),
                np.random.uniform(-0.1, 0.1)
            ])
            try:
                res_r = scipy.optimize.minimize(
                    objective, 
                    random_coeffs, 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if res_r.fun < best_score:
                    best_score = res_r.fun
                    best_coeffs = res_r.x
            except Exception:
                pass

    # Return the best coefficients found
    return best_coeffs
# EVOLVE-BLOCK-END
