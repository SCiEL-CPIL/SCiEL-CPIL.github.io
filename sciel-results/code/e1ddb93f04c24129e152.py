"""Improved: spherical parameterization exploiting scale invariance of the C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_hermite
from scipy.optimize import minimize
import time

_DEADLINE = 190

def _hermite_h(n, x):
    """Physicists' Hermite polynomials."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    elif n == 2:
        return 4.0 * x**2 - 2.0
    elif n == 3:
        return 8.0 * x**3 - 12.0 * x
    elif n == 4:
        return 16.0 * x**4 - 48.0 * x**2 + 12.0
    elif n == 7:
        return 128.0*x**7 - 4480.0*x**5 + 40320.0*x**3 - 67200.0*x
    elif n == 8:
        return 256.0*x**8 - 4480.0*x**6 + 26880.0*x**4 - 67200.0*x**2 + 40320.0
    elif n == 11:
        return (2048.0*x**11 - 147456.0*x**9 + 3133440.0*x**7
                - 30198720.0*x**5 + 122305920.0*x**3 - 145152000.0*x)
    elif n == 12:
        return (4096.0*x**12 - 147456.0*x**10 + 2580480.0*x**8
                - 20321280.0*x**6 + 81729600.0*x**4
                - 172972800.0*x**2 + 103219200.0)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 256
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        dH4 = 8.0 * _hermite_h(3, x_quad)
        dH8 = 16.0 * _hermite_h(7, x_quad)
        dH12 = 24.0 * _hermite_h(11, x_quad)
        
        x2 = x_quad**2
        x2w = x2 * w_quad
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH4, dH8, dH12, x2, x2w)
    return _quad_data

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 40320.0
H12_0 = 103219200.0

def _evaluate_from_directions(v):
    """
    Given a 3D direction vector v (nonzero), compute the C4 bound.
    The bound is scale-invariant, so only direction matters.
    Returns (bound, log_bound) or (inf, inf) if invalid.
    """
    if time.time() > _DEADLINE:
        return float('inf'), float('inf')
    
    x_quad, w_quad, H0, H4, H8, H12, dH4, dH8, dH12, x2, x2w = _get_quad_data()
    
    a0, a4, a8 = v
    
    # Determine d from P(0) = 0: a0 + 12*a4 + 40320*a8 + d*H12(0) = 0
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Build P = a0*H0 + a4*H4 + a8*H8 + d*H12
    # and dP - x*P for the Fourier side
    P = a0*H0 + a4*H4 + a8*H8 + d*H12
    P2 = P * P
    
    # Q = dP - x*P
    dP = a4*dH4 + a8*dH8 + d*dH12  # dH0 = 0
    Q = dP - x_quad * P
    Q2 = Q * Q
    
    S0 = np.sum(w_quad * P2)
    if S0 <= 1e-15:
        return float('inf'), float('inf')
    
    Sx = np.sum(x2w * P2)
    if Sx <= 1e-15:
        return float('inf'), float('inf')
    
    Sd = np.sum(w_quad * Q2)
    if Sd <= 1e-15:
        return float('inf'), float('inf')
    
    var_x = Sx / S0
    var_k = Sd / S0
    
    if var_x <= 0 or var_k <= 0:
        return float('inf'), float('inf')
    
    log_bound = np.log(var_x) + np.log(var_k)
    bound = np.exp(log_bound)
    
    return bound, log_bound

def _bound_from_angles(theta, phi):
    """
    Convert spherical angles to direction and compute bound.
    theta in [0, pi], phi in [0, 2*pi)
    v = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    v = np.array([st * np.cos(phi), st * np.sin(phi), ct])
    return _evaluate_from_directions(v)

def construct():
    start_time = time.time()
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_norm = np.linalg.norm(seed)
    if seed_norm < 1e-15:
        seed = np.array([1.0, 0.0, 0.0])
        seed_norm = 1.0
    seed_dir = seed / seed_norm
    
    # Convert seed direction to spherical angles
    sx = seed_dir[0]
    sy = seed_dir[1]
    sz = seed_dir[2]
    
    seed_theta = np.arccos(np.clip(sz, -1.0, 1.0))
    seed_phi = np.arctan2(sy, sx)
    if seed_phi < 0:
        seed_phi += 2.0 * np.pi
    
    seed_bound, seed_log = _evaluate_from_directions(seed)
    best_bound = seed_bound
    best_v = seed.copy()
    best_log = seed_log
    
    if best_log == float('inf'):
        best_log = 999.0
        best_bound = np.exp(best_log)
    
    # === Phase 1: High-resolution grid search on the sphere ===
    # Adaptive grid: start coarse, refine near best
    grid_results = []
    n_theta = 60
    n_phi = 120
    
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
    
    # Vectorized evaluation for grid
    best_grid_log = float('inf')
    best_grid_theta = seed_theta
    best_grid_phi = seed_phi
    
    for i_t, th in enumerate(thetas):
        for i_p, ph in enumerate(phis):
            if time.time() > _DEADLINE - 60:
                break
            st = np.sin(th)
            ct = np.cos(th)
            v = np.array([st * np.cos(ph), st * np.sin(ph), ct])
            bound, log_b = _evaluate_from_directions(v)
            if log_b < best_grid_log:
                best_grid_log = log_b
                best_grid_theta = th
                best_grid_phi = ph
                best_v = v.copy()
                best_bound = bound
                best_log = log_b
    
    # === Phase 2: Local refinement around best grid point ===
    # Use 2D optimization in (theta, phi) space
    def _obj_2d(params):
        if time.time() > _DEADLINE - 20:
            return float('inf')
        th, ph = params
        # Wrap phi
        ph = ph % (2.0 * np.pi)
        st = np.sin(th)
        ct = np.cos(th)
        v = np.array([st * np.cos(ph), st * np.sin(ph), ct])
        _, log_b = _evaluate_from_directions(v)
        return log_b
    
    # Try several starting points: best grid, seed, and nearby perturbations
    starting_points = [
        (best_grid_theta, best_grid_phi),
        (seed_theta, seed_phi),
        (best_grid_theta + 0.1, best_grid_phi + 0.1),
        (best_grid_theta - 0.1, best_grid_phi - 0.1),
        (best_grid_theta, best_grid_phi + np.pi),  # opposite hemisphere
    ]
    
    for sp in starting_points:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_obj_2d, np.array(sp), method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_log:
                best_log = res.fun
                best_bound = np.exp(res.fun)
                th_opt, ph_opt = res.x
                ph_opt = ph_opt % (2.0 * np.pi)
                st = np.sin(th_opt)
                ct = np.cos(th_opt)
                best_v = np.array([st * np.cos(ph_opt), st * np.sin(ph_opt), ct])
        except Exception:
            pass
    
    # === Phase 3: Direct 3D optimization with gradient (scale-invariant) ===
    # Use the fact that the objective is homogeneous of degree 0 in v.
    # We can fix ||v|| = 1 and optimize on the sphere.
    # Use a Lagrange-multiplier approach or just project gradient onto sphere.
    
    def _sphere_objective_and_grad(v):
        """Objective and projected gradient on the unit sphere."""
        if time.time() > _DEADLINE - 15:
            return float('inf'), np.zeros(3)
        
        x_quad, w_quad, H0, H4, H8, H12, dH4, dH8, dH12, x2, x2w = _get_quad_data()
        
        a0, a4, a8 = v
        d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        P = a0*H0 + a4*H4 + a8*H8 + d*H12
        P2 = P * P
        
        dP = a4*dH4 + a8*dH8 + d*dH12
        Q = dP - x_quad * P
        Q2 = Q * Q
        
        S0 = np.sum(w_quad * P2)
        if S0 <= 1e-15:
            return float('inf'), np.zeros(3)
        
        Sx = np.sum(x2w * P2)
        Sd = np.sum(w_quad * Q2)
        
        if Sx <= 1e-15 or Sd <= 1e-15:
            return float('inf'), np.zeros(3)
        
        var_x = Sx / S0
        var_k = Sd / S0
        log_bound = np.log(var_x) + np.log(var_k)
        
        # Compute gradient w.r.t. (a0, a4, a8)
        # d_d_a0 = -H0_0/H12_0, etc.
        d_d0 = -H0_0 / H12_0
        d_d4 = -H4_0 / H12_0
        d_d8 = -H8_0 / H12_0
        
        # P_i for each component
        P0_part = H0 + d_d0 * H12
        P4_part = H4 + d_d4 * H12
        P8_part = H8 + d_d8 * H12
        
        Q0_part = d_d0 * (dH12 - x_quad * H12)  # dH0=0, so Q0 = 0 + d_d0*(dH12 - x*H12)
        Q4_part = dH4 + d_d4 * (dH12 - x_quad * H12)
        Q8_part = dH8 + d_d8 * (dH12 - x_quad * H12)
        
        # dS0/da_i = 2 * sum(w * P * P_i_part)
        dS0_0 = 2.0 * np.sum(w_quad * P * P0_part)
        dS0_4 = 2.0 * np.sum(w_quad * P * P4_part)
        dS0_8 = 2.0 * np.sum(w_quad * P * P8_part)
        
        dSx_0 = 2.0 * np.sum(x2w * P * P0_part)
        dSx_4 = 2.0 * np.sum(x2w * P * P4_part)
        dSx_8 = 2.0 * np.sum(x2w * P * P8_part)
        
        dSd_0 = 2.0 * np.sum(w_quad * Q * Q0_part)
        dSd_4 = 2.0 * np.sum(w_quad * Q * Q4_part)
        dSd_8 = 2.0 * np.sum(w_quad * Q * Q8_part)
        
        # d(log(Sx/S0))/da_i = dSx_i/Sx - dS0_i/S0
        gx_0 = dSx_0 / Sx - dS0_0 / S0
        gx_4 = dSx_4 / Sx - dS0_4 / S0
        gx_8 = dSx_8 / Sx - dS0_8 / S0
        
        gk_0 = dSd_0 / Sd - dS0_0 / S0
        gk_4 = dSd_4 / Sd - dS0_4 / S0
        gk_8 = dSd_8 / Sd - dS0_8 / S0
        
        grad_full = np.array([gx_0 + gk_0, gx_4 + gk_4, gx_8 + gk_8])
        
        # Project gradient onto tangent plane of sphere at v
        # proj = grad_full - (v · grad_full) * v  (since ||v||=1)
        dot = np.dot(v, grad_full)
        grad_tangent = grad_full - dot * v
        
        return log_bound, grad_tangent
    
    # BFGS on the sphere: we can use the projected gradient
    def _sphere_obj_grad(params):
        v = np.array(params, dtype=float)
        vn = np.linalg.norm(v)
        if vn < 1e-10:
            return float('inf'), np.zeros(3)
        v_unit = v / vn
        obj, grad_t = _sphere_objective_and_grad(v_unit)
        if obj == float('inf'):
            return 999.0, np.zeros(3)
        return obj, grad_t
    
    # Start from current best
    best_v_unit = best_v / np.linalg.norm(best_v)
    
    if time.time() < _DEADLINE - 25:
        try:
            res = minimize(_sphere_obj_grad, best_v_unit, method='L-BFGS-B',
                         jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_log:
                best_log = res.fun
                best_bound = np.exp(res.fun)
                best_v = res.x.copy()
        except Exception:
            pass
    
    # === Phase 4: Multiple random restarts on sphere ===
    np.random.seed(12345)
    if time.time() < _DEADLINE - 20:
        for i in range(20):
            if time.time() > _DEADLINE - 20:
                break
            
            # Random direction
            if i < 10:
                # Uniform on sphere
                z1 = np.random.randn()
                z2 = np.random.randn()
                z3 = np.random.randn()
                v_rand = np.array([z1, z2, z3])
                v_rand /= np.linalg.norm(v_rand)
            else:
                # Perturbation of best
                v_rand = best_v / np.linalg.norm(best_v) + np.random.randn(3) * 0.3
                v_rand /= np.linalg.norm(v_rand)
            
            try:
                res = minimize(_sphere_obj_grad, v_rand, method='L-BFGS-B',
                             jac=True, options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-11})
                if res.fun < best_log:
                    best_log = res.fun
                    best_bound = np.exp(res.fun)
                    best_v = res.x.copy()
            except Exception:
                pass
    
    # === Phase 5: Final high-precision polish ===
    if time.time() < _DEADLINE - 10:
        try:
            best_v_unit = best_v / np.linalg.norm(best_v)
            res = minimize(_sphere_obj_grad, best_v_unit, method='L-BFGS-B',
                         jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_log:
                best_log = res.fun
                best_v = res.x.copy()
        except Exception:
            pass
    
    # Also try Nelder-Mead polish in 2D angle space
    if time.time() < _DEADLINE - 5:
        bv = best_v / np.linalg.norm(best_v)
        th = np.arccos(np.clip(bv[2], -1.0, 1.0))
        ph = np.arctan2(bv[1], bv[0])
        if ph < 0:
            ph += 2.0 * np.pi
        
        def _obj_2d_final(params):
            if time.time() > _DEADLINE - 3:
                return float('inf')
            th, ph = params
            ph = ph % (2.0 * np.pi)
            st = np.sin(th)
            ct = np.cos(th)
            v = np.array([st * np.cos(ph), st * np.sin(ph), ct])
            _, log_b = _evaluate_from_directions(v)
            return log_b
        
        try:
            res = minimize(_obj_2d_final, np.array([th, ph]), method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_log:
                best_log = res.fun
                th_opt, ph_opt = res.x
                ph_opt = ph_opt % (2.0 * np.pi)
                st = np.sin(th_opt)
                ct = np.cos(th_opt)
                best_v = np.array([st * np.cos(ph_opt), st * np.sin(ph_opt), ct])
        except Exception:
            pass
    
    # Return the three coefficients (a0, a4, a8) from best direction
    # Scale doesn't matter for the bound, but return the actual coefficient values
    return np.array(best_v)

# EVOLVE-BLOCK-END
