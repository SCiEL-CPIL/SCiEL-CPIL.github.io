"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    """
    Finds three coefficients [c0, c1, c2] such that the constructed polynomial g(x)
    minimizes the upper bound on the constant C4 in the uncertainty inequality.
    
    The strategy is to perform a global stochastic search (Differential Evolution)
    over the coefficient space to find a polynomial whose largest sign-changing
    root of q(x)=g(x)/x^2 is minimized. This approach avoids getting stuck in
    local minima near the seed that might trap local gradient-based or simplex
    methods.
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
        # Use a slightly tighter tolerance for imaginary parts to avoid false positives
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        if len(real_roots) == 0:
            return 1e100

        # Sort real roots
        real_roots.sort()
        
        # Find the largest real root at which q changes sign.
        # We evaluate q at points slightly to the left and right of each root.
        # A sign change means q(x-eps) * q(x+eps) < 0.
        
        # We want the LARGEST such root. So we iterate from largest to smallest.
        # Use a small eps relative to the root magnitude to ensure we are close enough
        # to detect sign change but not so small that floating point errors dominate.
        max_sign_change_root = None
        
        for r in reversed(real_roots):
            if r <= 0:
                # Roots must be positive for r^2 to be meaningful in this context? 
                # Actually r is a root of q. q(x) ~ x^10. 
                # The problem states "largest real root". 
                # If r is negative, r^2 is same as (-r)^2. 
                # However, usually these inequalities involve |x| or x>0.
                # Let's assume we want the largest magnitude positive root?
                # "largest real root" usually implies the maximum value on the real line.
                # If the largest root is negative, there are no positive roots.
                # But q(x) is even? No, q(x) is a polynomial in x^2?
                # g(x) is a sum of even Hermite polynomials (H0, H4, H8, H12 are all even).
                # So g(x) is even. q(x) = g(x)/x^2 is also even.
                # Therefore roots come in pairs +/-r.
                # "Largest real root" will be the positive one if any exist.
                # If all roots are complex or zero, we handle that.
                continue
                
            # Adaptive epsilon
            eps = max(1e-8, abs(r) * 1e-6)
            
            # Evaluate q at r - eps and r + eps
            x_l = r - eps
            x_r = r + eps
            
            # Horner's method
            q_left = 0.0
            q_right = 0.0
            for coeff in q_coeffs:
                q_left = q_left * x_l + coeff
                q_right = q_right * x_r + coeff
                
            if q_left * q_right < 0:
                max_sign_change_root = r
                break
                
        if max_sign_change_root is None:
            # If no sign change root found, it might be that the polynomial doesn't cross zero
            # in a way that changes sign (e.g. double roots) or we missed it.
            # Return a large penalty.
            return 1e100
            
        r_val = max_sign_change_root
        return (r_val ** 2) / (2.0 * np.pi)

    # Define objective for scipy
    def objective(c):
        val = evaluate(c[0], c[1], c[2])
        return val

    # Initial guess (the seed)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Check initial value
    best_val = objective(x0)
    best_c = x0.copy()
    
    # Define bounds for Differential Evolution
    # The coefficients are likely small. Let's explore a reasonable range.
    # Seed: c0 ~ -1.13, c1 ~ 0.04, c2 ~ 0.0003
    # Let's try bounds of [-5, 5] for c0, [-0.5, 0.5] for c1, [-0.05, 0.05] for c2
    # This is a heuristic choice. DE is robust to bounds if the optimum is inside.
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Perform global optimization
    try:
        # maxiter and popsize control the number of evaluations.
        # popsize=10, maxiter=50 -> 500 evaluations roughly.
        # Each evaluation is fast (<1ms).
        # We can afford more.
        res = differential_evolution(objective, bounds, 
                                     maxiter=100, 
                                     popsize=15,
                                     tol=1e-6,
                                     seed=42,
                                     polish=False) # Skip polish to save time, DE usually finds good spots
        
        if res.fun < best_val:
            best_c = res.x
            best_val = res.fun
            
    except Exception as e:
        # If optimization fails, fall back to seed
        pass

    # Check time
    if time.time() > deadline - 10:
        return best_c

    # If we have time, try a second pass with tighter bounds around the best found
    if time.time() < deadline - 30:
        try:
            # Tighten bounds around the best solution found
            delta = 0.5
            bounds_tight = [
                (best_c[0] - delta, best_c[0] + delta),
                (best_c[1] - delta*0.1, best_c[1] + delta*0.1),
                (best_c[2] - delta*0.01, best_c[2] + delta*0.01)
            ]
            res2 = differential_evolution(objective, bounds_tight, 
                                         maxiter=50, 
                                         popsize=10,
                                         tol=1e-8,
                                         seed=43,
                                         polish=False)
            if res2.fun < best_val:
                best_c = res2.x
                best_val = res2.fun
        except Exception:
            pass

    return best_c
# EVOLVE-BLOCK-END
