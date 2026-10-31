"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
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
    # Barycentric coordinates
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b + 0.5c
    # y = c * H
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
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    Minimizes negative soft-min area + penalty for points outside triangle.
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    c = 2 * pts[:, 1] / np.sqrt(3)
    b = pts[:, 0] - 0.5 * c
    a = 1 - b - c
    
    pen_a = np.maximum(0, -a)
    pen_b = np.maximum(0, -b)
    pen_c = np.maximum(0, -c)
    penalty = np.sum(pen_a**2 + pen_b**2 + pen_c**2)
    
    pts_clamped = _clamp_all(pts)
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    return -area_val + 100.0 * penalty

def _get_critical_triangles(pts, k=10):
    """
    Identify the k-smallest area triangles.
    Returns list of (i, j, k) tuples for these triangles.
    """
    areas = _all_areas(pts)
    indices = np.argsort(areas)[:k]
    return _TRI_IDX[indices]

def _nlp_objective(x):
    """
    NLP objective: minimize -alpha where x = [alpha, x1, y1, ..., x11, y11]
    Constraints: each critical triangle area >= alpha
    """
    alpha = x[0]
    return -alpha

def _nlp_constraints_critical(x, critical_tris):
    """
    Constraints for NLP: each critical triangle area >= alpha
    Returns array of (area_i - alpha) for critical triangles only
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    
    # Compute areas for only the critical triangles
    n_crit = len(critical_tris)
    areas = np.zeros(n_crit)
    for idx, (i, j, k) in enumerate(critical_tris):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        dx1 = pj[0] - pi[0]
        dy1 = pj[1] - pi[1]
        dx2 = pk[0] - pi[0]
        dy2 = pk[1] - pi[1]
        areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
    
    return areas - alpha

def _nlp_jacobian_constraints_critical(x, critical_tris):
    """
    Jacobian of critical constraints w.r.t. x
    Constraint i: area_i - alpha >= 0
    d(area_i)/d(alpha) = -1
    d(area_i)/d(point_k) depends on which points are in triangle i
    """
    alpha = x[0]
    pts = x[1:].reshape(-1, 2)
    n_vars = 1 + 2 * N
    n_crit = len(critical_tris)
    
    jac = np.zeros((n_crit, n_vars))
    
    # d(area_i)/d(alpha) = -1 for all i
    jac[:, 0] = -1.0
    
    for idx, (i, j, k) in enumerate(critical_tris):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # Signed cross product
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        sign = np.sign(cross)
        if sign == 0:
            sign = 1.0
        
        # Gradients for point i
        jac[idx, 1 + 2*i] = 0.5 * sign * (pj[1] - pk[1])
        jac[idx, 2 + 2*i] = 0.5 * sign * (pk[0] - pj[0])
        
        # Gradients for point j
        jac[idx, 1 + 2*j] = 0.5 * sign * (pi[1] - pk[1])
        jac[idx, 2 + 2*j] = 0.5 * sign * (pk[0] - pi[0])
        
        # Gradients for point k
        jac[idx, 1 + 2*k] = 0.5 * sign * (pj[1] - pi[1])
        jac[idx, 2 + 2*k] = 0.5 * sign * (pi[0] - pj[0])
    
    return jac

def _nlp_jacobian_objective(x):
    """Jacobian of objective: d(-alpha)/dx = [-1, 0, 0, ..., 0]"""
    n_vars = 1 + 2 * N
    jac = np.zeros(n_vars)
    jac[0] = -1.0
    return jac

def _nlp_jacobian_hessian_lagrange(x, v):
    """Hessian of Lagrangian (zero for linear constraints)"""
    n_vars = 1 + 2 * N
    return np.zeros((n_vars, n_vars))

def _nlp_bounds():
    """
    Bounds for NLP: alpha in [0, 0.5], points in triangle
    """
    bounds = [(0.0, 0.5)]  # alpha
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, H))    # y_i
    return bounds

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # First, get a good initial point using differential evolution
    bounds_de = [(0.0, 1.0)] * N + [(0.0, H)] * N
    best_pts = None
    
    try:
        if time_left() > 30.0:
            res_de = differential_evolution(
                _objective_de,
                bounds_de,
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=12,
                polish=False,
                updating='immediate'
            )
            if res_de.success or res_de.fun < 1e6:
                best_pts = _clamp_all(res_de.x.reshape(-1, 2))
            else:
                best_pts = _initial_placement_symmetric()
        else:
            best_pts = _initial_placement_symmetric()
    except Exception:
        best_pts = _initial_placement_symmetric()
    
    current_best = _min_area(best_pts)
    
    # Phase 1: Iterative NLP with dynamic critical set
    # We iteratively identify critical triangles and solve NLP with only those constraints
    nlp_iterations = 0
    max_nlp_iterations = 5
    
    while time_left() > 15.0 and nlp_iterations < max_nlp_iterations:
        try:
            # Identify current critical triangles
            critical_tris = _get_critical_triangles(best_pts, k=10)
            
            # Initial guess: [alpha, x1, y1, ..., x11, y11]
            alpha_init = _min_area(best_pts) * 0.95
            x0 = np.zeros(1 + 2 * N)
            x0[0] = alpha_init
            x0[1:] = best_pts.flatten()
            
            # Bounds
            nlp_bounds = _nlp_bounds()
            
            # Define constraint function with closure over critical_tris
            def _constraint_func(x, crit_tris=critical_tris):
                return _nlp_constraints_critical(x, crit_tris)
            
            def _constraint_jac(x, crit_tris=critical_tris):
                return _nlp_jacobian_constraints_critical(x, crit_tris)
            
            from scipy.optimize import NonlinearConstraint
            
            nlp_con = NonlinearConstraint(
                _constraint_func,
                0.0,
                np.inf,
                jac=_constraint_jac
            )
            
            res_nlp = minimize(
                _nlp_objective,
                x0,
                method='trust-constr',
                bounds=nlp_bounds,
                constraints=[nlp_con],
                jac=_nlp_jacobian_objective,
                hess=_nlp_jacobian_hessian_lagrange,
                options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            
            if res_nlp.success or (hasattr(res_nlp, 'fun') and res_nlp.fun < 0):
                nlp_pts = _clamp_all(res_nlp.x[1:].reshape(-1, 2))
                new_min = _min_area(nlp_pts)
                if new_min > current_best:
                    best_pts = nlp_pts
                    current_best = new_min
            
            nlp_iterations += 1
            
            # Check if we're making progress
            if nlp_iterations > 1 and current_best < best_pts is not None:
                # No improvement, break
                break
                
        except Exception:
            break
    
    # Phase 2: L-BFGS-B on smooth objective
    if time_left() > 10.0:
        try:
            flat_pts = best_pts.flatten()
            res_lbfgs = minimize(
                lambda x: -_soft_min_area(_clamp_all(x.reshape(-1, 2)), beta=50.0),
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            if _min_area(polished_pts) > current_best:
                best_pts = polished_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass

    # Phase 3: Nelder-Mead on hard objective
    if time_left() > 5.0:
        try:
            res_nm = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass

    # Phase 4: Powell on hard objective
    if time_left() > 5.0:
        try:
            res_powell = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Powell',
                bounds=bounds_de,
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            final_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            if _min_area(final_pts) > current_best:
                best_pts = final_pts
                current_best = _min_area(best_pts)
        except Exception:
            pass

    # Phase 5: Another round of NLP with updated critical set
    if time_left() > 10.0:
        try:
            critical_tris = _get_critical_triangles(best_pts, k=10)
            
            alpha_init = _min_area(best_pts) * 0.95
            x0 = np.zeros(1 + 2 * N)
            x0[0] = alpha_init
            x0[1:] = best_pts.flatten()
            
            nlp_bounds = _nlp_bounds()
            
            def _constraint_func2(x, crit_tris=critical_tris):
                return _nlp_constraints_critical(x, crit_tris)
            
            def _constraint_jac2(x, crit_tris=critical_tris):
                return _nlp_jacobian_constraints_critical(x, crit_tris)
            
            from scipy.optimize import NonlinearConstraint
            
            nlp_con2 = NonlinearConstraint(
                _constraint_func2,
                0.0,
                np.inf,
                jac=_constraint_jac2
            )
            
            res_nlp2 = minimize(
                _nlp_objective,
                x0,
                method='trust-constr',
                bounds=nlp_bounds,
                constraints=[nlp_con2],
                jac=_nlp_jacobian_objective,
                hess=_nlp_jacobian_hessian_lagrange,
                options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            
            if res_nlp2.success or (hasattr(res_nlp2, 'fun') and res_nlp2.fun < 0):
                nlp_pts2 = _clamp_all(res_nlp2.x[1:].reshape(-1, 2))
                if _min_area(nlp_pts2) > current_best:
                    best_pts = nlp_pts2
                    current_best = _min_area(best_pts)
        except Exception:
            pass

    # Final polish
    if time_left() > 3.0:
        try:
            res_nm2 = minimize(
                lambda x: -_min_area(_clamp_all(x.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            final_pts2 = _clamp_all(res_nm2.x.reshape(-1, 2))
            if _min_area(final_pts2) > current_best:
                best_pts = final_pts2
        except Exception:
            pass
            
    return _clamp_all(best_pts)

def _initial_placement_symmetric():
    """Symmetric initial placement."""
    pts = []
    # Center
    pts.append([1/3, H/3])
    # 3 points near vertices
    m = 0.15
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.2
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # 3 more points in a ring
    cx, cy = 1/3, H/3
    r = 0.25
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

# EVOLVE-BLOCK-END
