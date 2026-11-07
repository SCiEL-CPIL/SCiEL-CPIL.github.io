"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import sys

# Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Values at x=0
H0_0 = 1
H4_0 = 12
H8_0 = 168
H12_0 = 18480

def compute_q_coeffs(c0, c1, c2):
    """Compute coefficients of q(x) = g(x)/x^2 as a degree-10 polynomial."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) coefficients (degree 12 to 0):
    # x^12: c3 * 4096
    # x^10: c3 * (-46080)
    # x^8: c2*256 + c3*332160
    # x^6: c1*0 + c2*(-1344) + c3*(-1003520)
    # x^4: c1*16 + c2*2688 + c3*1424640
    # x^2: c1*(-48) + c2*(-1344) + c3*(-921600)
    # x^0: c0 + c1*12 + c2*168 + c3*18480 = 0 (by construction)
    
    # q(x) = g(x)/x^2, so divide by x^2:
    # x^10: c3 * 4096
    # x^8: c3 * (-46080)
    # x^6: c2*256 + c3*332160
    # x^4: c2*(-1344) + c3*(-1003520)
    # x^2: c1*16 + c2*2688 + c3*1424640
    # x^0: c1*(-48) + c2*(-1344) + c3*(-921600)
    
    coeffs = np.array([
        c3 * 4096,
        0,  # x^9
        c3 * (-46080),
        0,  # x^7
        c2 * 256 + c3 * 332160,
        0,  # x^5
        c2 * (-1344) + c3 * (-1003520),
        0,  # x^3
        c1 * 16 + c2 * 2688 + c3 * 1424640,
        0,  # x^1
        c1 * (-48) + c2 * (-1344) + c3 * (-921600)
    ])
    return coeffs

def evaluate_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) where r is the largest sign-changing root of q."""
    coeffs = compute_q_coeffs(c0, c1, c2)
    
    # Find real roots
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    
    if len(real_roots) == 0:
        return float('inf')
    
    # For each real root, check if q changes sign there
    # q changes sign at a root if the multiplicity is odd
    # For a polynomial, we can check the sign of q on either side
    
    sign_changing_roots = []
    for r in real_roots:
        if abs(r) < 1e-10:
            continue
        # Evaluate q at r - eps and r + eps
        eps = 1e-8 * max(1, abs(r))
        q_left = np.polyval(coeffs, r - eps)
        q_right = np.polyval(coeffs, r + eps)
        if q_left * q_right < 0:
            sign_changing_roots.append(r)
    
    if len(sign_changing_roots) == 0:
        return float('inf')
    
    r = max(sign_changing_roots)
    return r**2 / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Seed
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = evaluate_value(*seed)
    
    def objective(x):
        return evaluate_value(x[0], x[1], x[2])
    
    # Try differential evolution first (global search)
    try:
        result = differential_evolution(
            objective,
            bounds=[(-2, 2), (-2, 2), (-2, 2)],
            maxiter=50,
            popsize=15,
            tol=1e-10,
            seed=42
        )
        if time.time() < deadline - 30:
            if result.fun < best_val:
                best = result.x
                best_val = result.fun
    except Exception:
        pass
    
    # Refine with local optimization
    try:
        result = minimize(
            objective,
            best,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if time.time() < deadline - 10:
            if result.fun < best_val:
                best = result.x
                best_val = result.fun
    except Exception:
        pass
    
    # Try a few random restarts
    np.random.seed(0)
    for i in range(20):
        if time.time() > deadline - 30:
            break
        x0 = np.random.uniform(-2, 2, 3)
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return np.array(best)
# EVOLVE-BLOCK-END
