# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, basinhopping, rosenbrock


def construct():
    # 1. Precompute basis functions on a dense grid
    # The support of the Hermite-Gaussian is effectively within [-10, 10]
    # 50,000 points provide high resolution for the sup-norm calculation.
    x = np.linspace(-10.0, 10.0, 50000)
    h = np.exp(-0.5 * x * x)
    
    # Precompute Hermite polynomials H_0, H_4, H_8, H_12
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at x=0 for the constraint g(0)=0
    # H0(0)=1, H4(0)=-12, H8(0)=168, H12(0)=792
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 168.0
    H12_0 = 792.0

    def objective(coeffs):
        """
        Compute the uncertainty bound C4 = max|g| * max|g_hat|.
        coeffs: [c0, c1, c2]
        c3 is derived to satisfy g(0) = 0.
        """
        c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
        # Derive c3 such that c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Compute g(x)
        # g(x) = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * exp(-x^2/2)
        poly_part = c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12
        g = poly_part * h
        
        # Compute Fourier Transform g_hat(x)
        # For f(x) = H_n(x) e^{-x^2/2}, F[f](x) = (-i)^n H_n(x) e^{-x^2/2}
        # Here n in {0, 4, 8, 12} are all multiples of 4, so (-i)^n = 1.
        # Thus g_hat(x) has the same functional form as g(x) with the same coefficients.
        # Note: The standard Fourier transform definition might include a 1/sqrt(2pi) factor,
        # but it cancels out in the product if applied to both, or is constant.
        # The task asks for the bound on A(f)*A(Fourier(f)).
        # Assuming unitary Fourier transform or consistent normalization, the shape is identical.
        # If the transform is defined as int f(t) e^{-ixt} dt, there is a sqrt(2pi) factor.
        # Let's assume the standard convention where F[exp(-x^2/2)] = exp(-x^2/2).
        # In that case, F[H_n(x)exp(-x^2/2)] = (-i)^n H_n(x)exp(-x^2/2).
        # So g_hat(x) = g(x) for these specific degrees.
        
        A = np.max(np.abs(g))
        B = np.max(np.abs(g)) # Since g_hat has same magnitude profile
        
        # Optimization: If A is 0, return 0.
        if A == 0:
            return 0.0
            
        return A * B

    # 2. Optimization Strategy
    # Initial guess from literature (Goncalves et al.)
    # c0 ~ -1.13, c1 ~ 0.04, c2 ~ 0.0003
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    best_x = x0.copy()
    best_val = objective(x0)

    # Helper for local minimization using L-BFGS-B with numerical gradient
    # L-BFGS-B is generally faster and more robust than Nelder-Mead for smooth functions
    def local_minimize(x_start):
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                return res.fun, res.x
        except Exception:
            pass
        return best_val, best_x

    # Run Basin Hopping with multiple restarts to ensure global minimum
    # We run a few independent BH runs from different starting points
    
    # Start 1: Literature value
    try:
        bh_res = basinhopping(objective, x0, niter=200, 
                              minimizer_kwargs={'method': 'L-BFGS-B'},
                              stepsize=0.8, 
                              accept_test=lambda new, old: new.fun <= old.fun + 1e-9,
                              disp=False)
        if bh_res.fun < best_val:
            best_val = bh_res.fun
            best_x = bh_res.x
    except Exception:
        pass

    # Start 2: Perturbed literature value
    x0_2 = x0 + np.array([0.1, -0.01, 0.0001])
    try:
        bh_res2 = basinhopping(objective, x0_2, niter=200, 
                               minimizer_kwargs={'method': 'L-BFGS-B'},
                               stepsize=0.8, 
                               accept_test=lambda new, old: new.fun <= old.fun + 1e-9,
                               disp=False)
        if bh_res2.fun < best_val:
            best_val = bh_res2.fun
            best_x = bh_res2.x
    except Exception:
        pass

    # Start 3: Random nearby point
    x0_3 = x0 + np.random.normal(0, 0.2, size=3)
    try:
        bh_res3 = basinhopping(objective, x0_3, niter=200, 
                               minimizer_kwargs={'method': 'L-BFGS-B'},
                               stepsize=0.8, 
                               accept_test=lambda new, old: new.fun <= old.fun + 1e-9,
                               disp=False)
        if bh_res3.fun < best_val:
            best_val = bh_res3.fun
            best_x = bh_res3.x
    except Exception:
        pass

    # Final polish with high precision
    try:
        final_res = minimize(objective, best_x, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
        if final_res.fun < best_val:
            best_val = final_res.fun
            best_x = final_res.x
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
