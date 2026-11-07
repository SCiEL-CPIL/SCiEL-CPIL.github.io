"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def compute_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) where r is the largest real sign-changing root of q(x) = g(x)/x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12 with c3 chosen so g(0)=0.
    """
    # Hermite polynomial values at 0 and coefficients
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4608x^6 + 20736x^4 - 20736x^2 + 3024
    # H12(x) = 4096x^12 - 147456x^10 + 1658880x^8 - 6635520x^6 + 8028192x^4 - 2677104x^2 + 1032192
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 3024.0
    H12_0 = 1032192.0
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # If g tends to negative limit at infinity (c3 < 0 since leading coeff is 4096*c3), flip sign
    if c3 < 0:
        c0, c1, c2, c3 = -c0, -c1, -c2, -c3
    
    # Build q(x) = g(x)/x^2 as polynomial in y = x^2
    # g(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    # a0 = 0 by construction
    # q(x) = a12*x^10 + a10*x^8 + a8*x^6 + a6*x^4 + a4*x^2 + a2
    # P(y) = a12*y^5 + a10*y^4 + a8*y^3 + a6*y^2 + a4*y + a2
    
    a12 = 4096.0 * c3
    a10 = -147456.0 * c3
    a8 = 256.0 * c2 + 1658880.0 * c3
    a6 = -4608.0 * c2 - 6635520.0 * c3
    a4 = 16.0 * c1 + 20736.0 * c2 + 8028192.0 * c3
    a2 = -48.0 * c1 - 20736.0 * c2 - 2677104.0 * c3
    
    coeffs = [a12, a10, a8, a6, a4, a2]
    
    # Find roots of P(y)
    roots = np.roots(coeffs)
    
    # Filter real positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    pos_roots = real_roots[real_roots > 1e-12]
    
    if len(pos_roots) == 0:
        return 1e100
    
    # Sort in descending order and find the largest sign-changing root
    pos_roots_sorted = np.sort(pos_roots)[::-1]
    
    for y in pos_roots_sorted:
        eps = 1e-8 * max(1.0, abs(y))
        p_minus = np.polyval(coeffs, y - eps)
        p_plus = np.polyval(coeffs, y + eps)
        if p_minus * p_plus < 0:
            r = np.sqrt(y)
            return r**2 / (2.0 * np.pi)
    
    return 1e100

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    # Seed values
    c0_seed, c1_seed, c2_seed = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    best_val = compute_value(c0_seed, c1_seed, c2_seed)
    best_params = np.array([c0_seed, c1_seed, c2_seed])
    
    # Multi-start Nelder-Mead optimization
    rng = np.random.RandomState(42)
    
    # Define objective
    def obj(x):
        return compute_value(x[0], x[1], x[2])
    
    # Starting points: seed + diverse random points
    starts = [best_params.copy()]
    
    # Add diverse starting points
    for i in range(30):
        if time.time() > deadline - 15.0:
            break
        x0 = np.array([
            rng.uniform(-5.0, 5.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-0.1, 0.1)
        ])
        starts.append(x0)
    
    # Also add points near the seed
    for i in range(10):
        if time.time() > deadline - 15.0:
            break
        x0 = best_params + rng.normal(0, 0.5, 3)
        starts.append(x0)
    
    # Run Nelder-Mead from each starting point
    for x0 in starts:
        if time.time() > deadline - 10.0:
            break
        try:
            result = minimize(
                obj,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5.0:
        try:
            result_final = minimize(
                obj,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if result_final.fun < best_val:
                best_val = result_final.fun
                best_params = result_final.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
