"""Uncertainty bound minimization via multi-start search on the unit sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0

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
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
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

def _bound_from_angles(theta, phi):
    """
    Compute the bound for coefficients on the unit sphere.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi)
    """
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    return _compute_bound(a0, a4, a8)

def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses multi-start optimization on the unit sphere.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    # Convert seed to spherical coordinates
    r_seed = np.linalg.norm(seed)
    if r_seed > 1e-15:
        theta_seed = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
        phi_seed = np.arctan2(seed[1], seed[0])
        if phi_seed < 0:
            phi_seed += 2 * np.pi
    else:
        theta_seed = 0.0
        phi_seed = 0.0

    # Generate starting points on the sphere
    # 1. Seed point
    # 2. Fibonacci sphere points for coverage
    # 3. Random points
    
    rng = np.random.default_rng(42)
    
    start_points = []
    
    # Add seed
    start_points.append((theta_seed, phi_seed))
    
    # Add Fibonacci sphere points (approx 100 points)
    N = 100
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(N):
        theta = np.arccos(1 - 2 * (i + 0.5) / N)
        phi = golden_angle * i
        if phi < 0: phi += 2*np.pi
        start_points.append((theta, phi))
        
    # Add random points
    for _ in range(50):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        start_points.append((theta, phi))
        
    # Add points near seed
    for _ in range(20):
        dtheta = rng.normal(0, 0.2)
        dphi = rng.normal(0, 0.2)
        theta = np.clip(theta_seed + dtheta, 0.01, np.pi - 0.01)
        phi = (phi_seed + dphi) % (2 * np.pi)
        start_points.append((theta, phi))

    # Optimization loop
    for theta0, phi0 in start_points:
        if time.time() > _DEADLINE - 2:
            break
            
        # Initial evaluation
        f0 = _bound_from_angles(theta0, phi0)
        if f0 < best_bound:
            best_bound = f0
            a0 = np.sin(theta0) * np.cos(phi0)
            a4 = np.sin(theta0) * np.sin(phi0)
            a8 = np.cos(theta0)
            best_params = np.array([a0, a4, a8])
            
        # Local optimization
        # Use Powell method which is robust for small dimensional problems
        # We need to handle the periodicity of phi. 
        # We can optimize in a box [0, pi] x [0, 2*pi] and rely on the periodic nature.
        # However, if the minimum is near the boundary of phi (0 or 2*pi), 
        # the optimizer might struggle. 
        # To mitigate, we can shift the coordinate system so the seed is in the middle, 
        # or just use a small step size.
        
        # Let's try Nelder-Mead on the angles
        x0 = np.array([theta0, phi0])
        
        def objective(x):
            th, ph = x
            # Wrap phi to [0, 2*pi) to avoid large values
            ph_wrapped = ph % (2 * np.pi)
            # Clip theta to avoid numerical issues at poles
            th_clipped = np.clip(th, 1e-8, np.pi - 1e-8)
            return _bound_from_angles(th_clipped, ph_wrapped)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            if res.fun < best_bound:
                best_bound = res.fun
                th_f, ph_f = res.x
                ph_f = ph_f % (2 * np.pi)
                th_f = np.clip(th_f, 1e-8, np.pi - 1e-8)
                a0 = np.sin(th_f) * np.cos(ph_f)
                a4 = np.sin(th_f) * np.sin(ph_f)
                a8 = np.cos(th_f)
                best_params = np.array([a0, a4, a8])
        except Exception:
            pass

    # Final verification
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
