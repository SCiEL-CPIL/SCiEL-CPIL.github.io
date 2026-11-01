"""Uncertainty bound minimization via multi-start L-BFGS-B with analytic bounds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 190.0  # Safety margin


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n
    # Recurrence: I[n] = I[n-1] * (2n-1) / 2
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses multi-start L-BFGS-B optimization with a final polish.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    # Objective function wrapper
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
        
    best_bound = objective(seed)
    best_params = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Define a set of starting points
    start_points = [seed.copy()]
    
    # Generate random starting points in a reasonable range
    # The seed has a0 ~ -1.13. Let's explore a box around it and some global points.
    for _ in range(100):
        # Mix of local perturbations and global explorations
        if rng.random() < 0.5:
            # Local perturbation
            x = seed + rng.normal(0, 0.5, 3)
        else:
            # Global exploration
            scale = 10 ** rng.uniform(-1, 1) # 0.1 to 10
            x = rng.normal(0, 1, 3) * scale
        start_points.append(x)
        
    # Add structured points on spheres
    for r in [0.5, 1.0, 2.0, 5.0]:
        for theta1 in np.linspace(-np.pi, np.pi, 8):
            for theta2 in np.linspace(0.1, np.pi-0.1, 8):
                x = np.array([
                    r * np.sin(theta2) * np.cos(theta1),
                    r * np.sin(theta2) * np.sin(theta1),
                    r * np.cos(theta2)
                ])
                start_points.append(x)
                
    # Run L-BFGS-B from each start point
    for x0 in start_points:
        if time.time() > _DEADLINE - 15:
            break
            
        # Check initial value
        val0 = objective(x0)
        if val0 < best_bound:
            best_bound = val0
            best_params = x0.copy()
            
        try:
            # L-BFGS-B is robust for this smooth function
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
            
    # Final polish with Nelder-Mead on the best candidate
    if time.time() < _DEADLINE - 5:
        try:
            res_polish = minimize(objective, best_params, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if np.isfinite(res_polish.fun) and res_polish.fun < best_bound:
                best_bound = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Final verification
    final_bound = objective(best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
