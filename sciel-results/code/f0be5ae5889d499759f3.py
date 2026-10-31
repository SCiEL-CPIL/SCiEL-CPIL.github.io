"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    s = a + b + c
    a /= s
    b /= s
    c /= s
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area."""
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    shifted = -beta * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    pts.append([1/3, H/3])
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return _clamp_all(np.array(pts))

def _simulated_annealing(start_pts, time_budget, rng):
    """
    Simulated Annealing with adaptive step size for global exploration.
    Returns best points found.
    """
    current_pts = start_pts.copy()
    current_val = _min_area(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    # Adaptive step size: start large, shrink over time
    step_size = 0.15
    cooling_rate = 0.995
    min_step = 0.001
    
    start_t = time.time()
    iterations = 0
    
    while time.time() - start_t < time_budget:
        iterations += 1
        
        # Generate perturbation
        perturbation = rng.normal(0, step_size, size=(N, 2))
        new_pts = _clamp_all(current_pts + perturbation)
        new_val = _min_area(new_pts)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / (step_size * 10 + 1e-10)):
            current_pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_pts = current_pts.copy()
                best_val = current_val
        
        # Cool down and adapt step size
        step_size = max(min_step, step_size * cooling_rate)
        
        # Occasionally restart step size to escape local minima
        if iterations % 500 == 0:
            step_size = max(min_step, step_size * 1.5)
    
    return best_pts, best_val

def _nlp_objective(x):
    """NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]"""
    alpha = x[0]
    return -alpha

def _nlp_constraint_func(x):
    """Returns array of (area_i - alpha) for all triangles."""
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    areas = _all_areas(pts)
    return areas - alpha

def _nlp_jacobian(x):
    """Jacobian of the constraints with respect to x."""
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    jac = np.zeros((_N_TRI, 1 + 2 * N))
    jac[:, 0] = -1.0
    for idx, (i, j, k) in enumerate(_TRI_IDX):
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        signed_area_2 = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
        gxi = 0.5 * (yj - yk)
        gyi = 0.5 * (xk - xj)
        gxj = 0.5 * (yk - yi)
        gyj = 0.5 * (xi - xk)
        gxk = 0.5 * (yi - yj)
        gyk = 0.5 * (xj - xi)
        if signed_area_2 < 0:
            gxi, gyi, gxj, gyj, gxk, gyk = -gxi, -gyi, -gxj, -gyj, -gxk, -gyk
        jac[idx, 2*i + 1] = gxi
        jac[idx, 2*i + 2] = gyi
        jac[idx, 2*j + 1] = gxj
        jac[idx, 2*j + 2] = gyj
        jac[idx, 2*k + 1] = gxk
        jac[idx, 2*k + 2] = gyk
    return jac

def _nlp_refine(pts, time_budget):
    """Refine points using NLP (SLSQP) to maximize min area."""
    current_best = _min_area(pts)
    try:
        initial_alpha = current_best
        x0 = np.concatenate(([initial_alpha], pts.flatten()))
        bounds_nlp = [(0.0, 0.5)] + [(0.0, 1.0)] * N + [(0.0, H)] * N
        res_nlp = minimize(
            _nlp_objective,
            x0,
            method='SLSQP',
            bounds=bounds_nlp,
            constraints=[
                {'type': 'ineq', 'fun': _nlp_constraint_func, 'jac': _nlp_jacobian}
            ],
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if res_nlp.success or res_nlp.fun < -initial_alpha:
            nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
            nlp_val = _min_area(nlp_pts)
            if nlp_val > current_best:
                return nlp_pts, nlp_val
    except Exception:
        pass
    return pts, current_best

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated Annealing for global exploration
    initial_pts = _initial_placement_symmetric()
    sa_time = min(40.0, time_left() * 0.4)
    best_pts, best_val = _simulated_annealing(initial_pts, sa_time, rng)
    
    # Phase 2: NLP refinement
    if time_left() > 25.0:
        best_pts, best_val = _nlp_refine(best_pts, time_left() * 0.3)
    
    # Phase 3: Second SA round from refined point (re-explore)
    if time_left() > 20.0:
        sa_time2 = min(15.0, time_left() * 0.3)
        best_pts2, best_val2 = _simulated_annealing(best_pts, sa_time2, rng)
        if best_val2 > best_val:
            best_pts = best_pts2
            best_val = best_val2
    
    # Phase 4: NLP refinement again
    if time_left() > 15.0:
        best_pts, best_val = _nlp_refine(best_pts, time_left() * 0.2)
    
    # Phase 5: L-BFGS-B on soft objective for polishing
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            bounds_de = [(0.0, 1.0)] * N + [(0.0, H)] * N
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=100.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > best_val:
                best_pts = polished_pts
                best_val = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 6: Powell on hard objective
    if time_left() > 8.0:
        try:
            bounds_de = [(0.0, 1.0)] * N + [(0.0, H)] * N
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=bounds_de,
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            final_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            if _min_area(final_pts) > best_val:
                best_pts = final_pts
                best_val = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 7: Final Nelder-Mead polish
    if time_left() > 5.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > best_val:
                best_pts = final_pts
                best_val = _min_area(best_pts)
        except Exception:
            pass
    
    # Phase 8: Final NLP if time permits
    if time_left() > 10.0:
        best_pts, best_val = _nlp_refine(best_pts, time_left() * 0.5)
    
    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
