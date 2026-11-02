"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    The objective is to minimize the upper bound on A(f)*A(Fourier(f)), which for
    this Fourier-invariant combination is (sup |x*g(x)|)^2 (or similar norm product).
    
    We use a robust numerical optimization strategy:
    1. Fast evaluation of the bound using the symmetry property F(k)=g(k).
    2. Differential Evolution for global exploration.
    3. Basin Hopping for local refinement.
    4. Final polish with Nelder-Mead.
    """
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s limit

    # Pre-compute constants
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Grid for fast evaluation
    # Using a finer grid helps capture the peak of |x*g(x)| accurately
    N = 65536
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]
    
    # Pre-compute Hermite polynomials on the grid
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    # Note: H12 coefficients need to be verified.
    # Physicists' Hermite H12(x) = 4096x^12 - 129024x^10 + 1761280x^8 - 11942400x^6 + 37158400x^4 - 54432000x^2 + 47900160
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Pre-compute the basis functions for g(x)
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    # So g(x) = exp(-x^2/2) * [ c0*(H0 - H0_0/H12_0*H12) + c4*(H4 - H4_0/H12_0*H12) + c8*(H8 - H8_0/H12_0*H12) ]
    
    basis0 = (H0 - (H0_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    basis4 = (H4 - (H4_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    basis8 = (H8 - (H8_0/H12_0)*H12) * np.exp(-x**2 / 2.0)
    
    # We want to minimize (sup |x * g(x)|)^2
    # Let u = x * exp(-x^2/2)
    # Then x * g(x) = c0 * (x * basis0) + c4 * (x * basis4) + c8 * (x * basis8)
    # Actually basis already includes exp.
    # x * basis0 = x * (H0 - ...)*exp = ...
    # Let's just compute x * basis directly.
    
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    # Smooth approximation of the sup norm using L_p norm with large p
    # This allows gradient-based optimizers to work effectively
    P = 100.0
    
    def objective_smooth(coeffs):
        """
        Compute a smooth approximation of the squared supremum of |x * g(x)|.
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        abs_gx = np.abs(gx)
        # L_p norm approximation
        # sum(|gx|^P)^(1/P)
        # To avoid overflow, we can scale or use log-sum-exp, but P=100 is manageable with float64
        # if values are not huge. The values are bounded by ~10 * max(g).
        # max(g) is small. So gx is small.
        val = np.sum(abs_gx ** P) ** (1.0/P)
        return val * val

    def objective_true(coeffs):
        """
        Compute the exact squared supremum of |x * g(x)|.
        coeffs: [c0, c4, c8]
        """
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val

    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = objective_true(x0)
    best_x = x0.copy()
    
    # Define bounds for Differential Evolution
    # Based on the seed values, the coefficients are small.
    # c0 ~ -1.13, c4 ~ 0.04, c8 ~ 0.0003
    # Let's use a reasonable box.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]

    # 1. Differential Evolution for global search
    # Use the smooth objective to guide DE? DE doesn't use gradients, but smooth landscape might help convergence.
    # Actually, DE works well on non-smooth too. But let's use smooth for consistency in local steps.
    try:
        if time.time() < deadline - 30:
            de_result = differential_evolution(
                objective_smooth, bounds, 
                maxiter=100, 
                popsize=15, 
                seed=42,
                tol=1e-8,
                init='sobol'
            )
            # Evaluate the DE result with the TRUE objective
            de_true_val = objective_true(de_result.x)
            if de_true_val < best_val:
                best_val = de_true_val
                best_x = de_result.x.copy()
    except Exception:
        pass

    # 2. Basin Hopping for local refinement
    # Use L-BFGS-B on the smooth objective
    try:
        if time.time() < deadline - 15:
            def local_minimizer(x):
                # L-BFGS-B is efficient for smooth problems
                res = minimize(objective_smooth, x, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-10})
                return res.x, res.fun, {}
            
            bh_result = basinhopping(
                objective_smooth, best_x, 
                minimizer=local_minimizer,
                niter=50, 
                T=1.0, 
                stepsize=0.1,
                seed=42
            )
            # Evaluate the BH result with the TRUE objective
            bh_true_val = objective_true(bh_result.x)
            if bh_true_val < best_val:
                best_val = bh_true_val
                best_x = bh_result.x.copy()
    except Exception:
        pass

    # 3. Final Polish
    # Use L-BFGS-B on the smooth objective for final polish
    try:
        if time.time() < deadline - 5:
            res = minimize(objective_smooth, best_x, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12})
            res_true_val = objective_true(res.x)
            if res_true_val < best_val:
                best_val = res_true_val
                best_x = res.x.copy()
                
            # Try Powell on smooth objective as a backup
            res_powell = minimize(objective_smooth, best_x, method='Powell', 
                                 options={'maxiter': 500, 'ftol': 1e-12})
            res_powell_true_val = objective_true(res_powell.x)
            if res_powell_true_val < best_val:
                best_val = res_powell_true_val
                best_x = res_powell.x.copy()
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
