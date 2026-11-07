"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Finds three coefficients [c0, c1, c2] such that the constructed polynomial g(x)
    minimizes the upper bound on the constant C4 in the uncertainty inequality.
    
    The strategy is to perform a local search (Nelder-Mead) around the known seed
    [-113/100, 1/25, 1/3240] using a highly optimized pure-Numpy evaluator.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin below the 210s limit
    
    # Precompute constant coefficients for Hermite polynomials H0, H4, H8, H12
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 65536x^8 - 147456x^6 + 110592x^4 - 23040x^2 + 1344
    # H12(x) = 4096x^12 - 294912x^10 + 5806080x^8 - 50388480x^6 + 214958016x^4 - 451919360x^2 + 103783680
    # Values at x=0:
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 103783680.0

    def evaluate(c0, c1, c2):
        """
        Computes the value r^2 / (2 * pi) for the given coefficients.
        Returns a large penalty if invalid.
        """
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # q(x) = g(x) / x^2
        # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        # Coefficients for x^12, x^10, x^8, x^6, x^4, x^2, x^0 in g(x):
        # x^12: 4096*c3
        # x^10: -294912*c3
        # x^8: 65536*c2 + 5806080*c3
        # x^6: -147456*c2 - 50388480*c3
        # x^4: 16*c1 + 110592*c2 + 214958016*c3
        # x^2: -48*c1 - 23040*c2 - 451919360*c3
        # x^0: c0 + 12*c1 + 1344*c2 + 103783680*c3 (which is 0 by construction)
        
        # Since g(0)=0, the x^0 term is exactly 0.
        # q(x) = g(x)/x^2 is a polynomial of degree 10.
        # Coefficients of q(x) from highest to lowest power (x^10 to x^0):
        q_coeffs = np.array([
            4096.0 * c3,
            -294912.0 * c3,
            65536.0 * c2 + 5806080.0 * c3,
            -147456.0 * c2 - 50388480.0 * c3,
            16.0 * c1 + 110592.0 * c2 + 214958016.0 * c3,
            -48.0 * c1 - 23040.0 * c2 - 451919360.0 * c3,
            0.0
        ])
        
        # Find roots of q(x)
        try:
            roots = np.roots(q_coeffs)
        except Exception:
            return 1e100

        # Filter real roots
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        if len(real_roots) == 0:
            return 1e100

        # Sort real roots
        real_roots.sort()
        
        # Find the largest real root at which q changes sign.
        # We evaluate q at points slightly to the left and right of each root.
        # A sign change means q(x-eps) * q(x+eps) < 0.
        
        # We want the LARGEST such root. So we iterate from largest to smallest.
        eps = 1e-6
        max_sign_change_root = None
        
        for r in reversed(real_roots):
            # Evaluate q at r - eps and r + eps
            # Using Horner's method for speed and stability
            q_left = 0.0
            q_right = 0.0
            x_l = r - eps
            x_r = r + eps
            
            for coeff in q_coeffs:
                q_left = q_left * x_l + coeff
                q_right = q_right * x_r + coeff
                
            if q_left * q_right < 0:
                max_sign_change_root = r
                break
                
        if max_sign_change_root is None:
            return 1e100
            
        r_val = max_sign_change_root
        return (r_val ** 2) / (2.0 * np.pi)

    # Initial guess (the seed)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Define objective for scipy
    def objective(c):
        return evaluate(c[0], c[1], c[2])
        
    # Check initial value
    best_val = objective(x0)
    best_c = x0.copy()
    
    # Perform local optimization
    # Nelder-Mead is robust and doesn't require gradients.
    # We use a relatively tight tolerance to ensure we converge well within the time limit.
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 500})
        
        if res.success or res.fun < best_val:
            best_c = res.x
            best_val = res.fun
            
        # If we have time, we could try a second pass with tighter bounds or different start, 
        # but Nelder-Mead on this smooth function should be very fast.
        
    except Exception as e:
        # If optimization fails, fall back to seed
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_c

    return best_c
# EVOLVE-BLOCK-END
