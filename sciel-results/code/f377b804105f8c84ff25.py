"""Heilbronn-type problem: place N points to maximize min triangle area / hull area."""
import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 145.0  # seconds, safety margin before 150s kill

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    """Scale points so convex hull area = 1. Score is scale-invariant."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(n1, n2, r1, r2, rot1, rot2):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _four_ring_config(n1, n2, n3, n4, r1, r2, r3, r4, rot1, rot2, rot3, rot4):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    pts4 = _regular_polygon(n4, r4, rot4)
    return np.vstack([pts1, pts2, pts3, pts4])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """Compute negative smooth-min score and gradient for optimization."""
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
    
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _optimize_two_ring(n1, n2, start_time, deadline):
    def _two_ring_score(params):
        r_ratio = params[0]
        rot1 = params[1]
        rot2 = params[2]
        
        r1 = 1.0
        r2 = r_ratio * r1
        
        if r2 <= 0.01 or r2 > 2.0:
            return -1e6
        
        pts = _two_ring_config(n1, n2, r1, r2, rot1, rot2)
        sc = _score(pts)
        return -sc
    
    best_score = -1e6
    best_params = None
    
    for r_init in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for rot1_init in [0.0, np.pi/n1, 2*np.pi/n1, np.pi/(2*n1)]:
            for rot2_init in [0.0, np.pi/n2, 2*np.pi/n2, np.pi/(2*n2), np.pi/n1]:
                if time.time() > deadline - 5:
                    break
                
                x0 = np.array([r_init, rot1_init, rot2_init])
                
                try:
                    result = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    
                    sc = -result.fun
                    if sc > best_score:
                        best_score = sc
                        best_params = result.x.copy()
                except Exception:
                    pass
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_params is not None:
        r_ratio, rot1, rot2 = best_params
        pts = _two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
        return pts, best_score
    
    return None, -1e6

def _optimize_three_ring(n1, n2, n3, start_time, deadline):
    def _three_ring_score(params):
        r1 = params[0]
        r2 = params[1]
        rot1 = params[2]
        rot2 = params[3]
        rot3 = params[4]
        
        if r1 <= 0.01 or r2 <= 0.01 or r1 > 2.0 or r2 > 2.0:
            return -1e6
        
        pts = _three_ring_config(n1, n2, n3, r1, r2, 1.0, rot1, rot2, rot3)
        sc = _score(pts)
        return -sc
    
    best_score = -1e6
    best_params = None
    
    for r1_init in [0.3, 0.5, 0.7, 0.9]:
        for r2_init in [0.4, 0.6, 0.8, 1.0, 1.2]:
            if time.time() > deadline - 5:
                break
            
            x0 = np.array([r1_init, r2_init, 0.0, np.pi/n2, np.pi/n3])
            
            try:
                result = scipy.optimize.minimize(
                    _three_ring_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                sc = -result.fun
                if sc > best_score:
                    best_score = sc
                    best_params = result.x.copy()
            except Exception:
                pass
            
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_params is not None:
        r1, r2, rot1, rot2, rot3 = best_params
        pts = _three_ring_config(n1, n2, n3, r1, r2, 1.0, rot1, rot2, rot3)
        return pts, best_score
    
    return None, -1e6

def _optimize_four_ring(n1, n2, n3, n4, start_time, deadline):
    def _four_ring_score(params):
        r1 = params[0]
        r2 = params[1]
        r3 = params[2]
        rot1 = params[3]
        rot2 = params[4]
        rot3 = params[5]
        rot4 = params[6]
        
        if r1 <= 0.01 or r2 <= 0.01 or r3 <= 0.01:
            return -1e6
        
        pts = _four_ring_config(n1, n2, n3, n4, r1, r2, r3, 1.0, rot1, rot2, rot3, rot4)
        sc = _score(pts)
        return -sc
    
    best_score = -1e6
    best_params = None
    
    for r1_init in [0.2, 0.4, 0.6]:
        for r2_init in [0.3, 0.5, 0.7]:
            for r3_init in [0.6, 0.8, 1.0]:
                if time.time() > deadline - 5:
                    break
                
                x0 = np.array([r1_init, r2_init, r3_init, 0.0, np.pi/n2, 2*np.pi/n3, np.pi/n4])
                
                try:
                    result = scipy.optimize.minimize(
                        _four_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    
                    sc = -result.fun
                    if sc > best_score:
                        best_score = sc
                        best_params = result.x.copy()
                except Exception:
                    pass
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_params is not None:
        r1, r2, r3, rot1, rot2, rot3, rot4 = best_params
        pts = _four_ring_config(n1, n2, n3, n4, r1, r2, r3, 1.0, rot1, rot2, rot3, rot4)
        return pts, best_score
    
    return None, -1e6

def _refine_full_space(initial_points, start_time, deadline):
    """Refine points in full 28D space using smooth-min gradient optimization with beta annealing."""
    x0 = _normalize(initial_points).flatten()
    
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    x = x0.copy()
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def _obj_and_grad(x_flat, beta_val=beta):
            return _neg_smooth_score_and_grad(x_flat, beta_val)
        
        try:
            res = scipy.optimize.minimize(
                _obj_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            x = res.x
        except Exception:
            pass
    
    # Final high-beta polish
    if time.time() < deadline - 3:
        try:
            def _obj_and_grad_high(x_flat):
                return _neg_smooth_score_and_grad(x_flat, 2000.0)
            
            res = scipy.optimize.minimize(
                _obj_and_grad_high,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 2)

def _differential_evolution_search(start_time, deadline, pop_size=15):
    """Use differential evolution for global search in the full 28D space."""
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def _de_objective(x_flat):
        points = x_flat.reshape(-1, 2)
        sc = _score(points)
        return -sc
    
    try:
        result = scipy.optimize.differential_evolution(
            _de_objective,
            bounds,
            maxiter=40,
            popsize=pop_size,
            tol=1e-8,
            seed=42,
            polish=False
        )
        
        if result.success or result.fun < 0:
            points = result.x.reshape(-1, 2)
            sc = _score(points)
            return points, sc
    except Exception:
        pass
    
    return None, -1e6

def _dual_heptagon_edge_search(start_time, deadline):
    """
    Structural change: Represent 14 points as union of a regular outer heptagon
    and 7 inner points constrained to lie on the edges of the outer heptagon.
    
    Parameters (10D):
      - r_outer: outer heptagon radius
      - rot_outer: outer heptagon rotation
      - t_0..t_6: position along each edge (0=vertex_i, 1=vertex_{i+1})
    
    This allows asymmetric inner configurations while maintaining the
    heptagonal structure, explored via differential evolution.
    """
    def _build_points(params):
        r_outer = params[0]
        rot_outer = params[1]
        t_vals = params[2:9]
        
        # Outer heptagon vertices
        angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rot_outer
        outer = np.column_stack([r_outer * np.cos(angles), r_outer * np.sin(angles)])
        
        # Inner points on edges: edge i goes from vertex i to vertex (i+1)%7
        inner = np.zeros((7, 2))
        for i in range(7):
            j = (i + 1) % 7
            t = t_vals[i]
            # Clamp t to [0, 1] to stay on the edge
            t = np.clip(t, 0.0, 1.0)
            inner[i] = (1.0 - t) * outer[i] + t * outer[j]
        
        return np.vstack([outer, inner])
    
    def _objective(params):
        r_outer = params[0]
        if r_outer < 0.3 or r_outer > 2.0:
            return 1e6
        pts = _build_points(params)
        sc = _score(pts)
        return -sc
    
    # Differential evolution over the 10D parameter space
    bounds = [(0.5, 1.5)] + [(0.0, 2*np.pi)] + [(0.0, 1.0)] * 7
    
    try:
        result = scipy.optimize.differential_evolution(
            _objective,
            bounds,
            maxiter=60,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.8
        )
        
        if result.fun < 0:
            pts = _build_points(result.x)
            sc = _score(pts)
            return pts, sc
    except Exception:
        pass
    
    # Fallback: try a few Nelder-Mead starts if DE failed
    best_score = -1e6
    best_params = None
    
    starts = []
    for r_out in [1.0, 1.2]:
        for rot in [0.0, np.pi/7]:
            for t_val in [0.3, 0.5, 0.7]:
                params = np.array([r_out, rot] + [t_val]*7)
                starts.append(params)
    
    for x0 in starts:
        if time.time() > deadline - 3:
            break
        try:
            result = scipy.optimize.minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            sc = -result.fun
            if sc > best_score:
                best_score = sc
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        pts = _build_points(best_params)
        return pts, best_score
    
    return None, -1e6

def _hexagon_plus_heptagon_search(start_time, deadline):
    """
    6 points on regular hexagon + 7 points on regular heptagon.
    Optimize: r_hex, rot_hex, r_hept, rot_hept (4 params)
    """
    def _build_points(params):
        r_hex, rot_hex, r_hept, rot_hept = params
        hex_pts = _regular_polygon(6, r_hex, rot_hex)
        hept_pts = _regular_polygon(7, r_hept, rot_hept)
        return np.vstack([hex_pts, hept_pts])
    
    def _objective(params):
        if params[0] < 0.05 or params[2] < 0.05:
            return 1e6
        pts = _build_points(params)
        sc = _score(pts)
        return -sc
    
    best_score = -1e6
    best_params = None
    
    for r_hex in [0.3, 0.5, 0.7, 0.9]:
        for r_hept in [1.0, 1.2, 1.5]:
            for rot_hex in [0.0, np.pi/6, np.pi/3]:
                for rot_hept in [0.0, np.pi/7, 2*np.pi/7]:
                    if time.time() > deadline - 5:
                        break
                    x0 = np.array([r_hex, rot_hex, r_hept, rot_hept])
                    try:
                        result = scipy.optimize.minimize(
                            _objective, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                        )
                        sc = -result.fun
                        if sc > best_score:
                            best_score = sc
                            best_params = result.x.copy()
                    except Exception:
                        pass
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if best_params is not None:
        pts = _build_points(best_params)
        return pts, best_score
    
    return None, -1e6

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
        return sc
    
    # 1. Explore two-ring configurations
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 50:
            break
        pts, sc = _optimize_two_ring(n1, n2, start_time, deadline - 50)
        if pts is not None:
            _update_best(pts)
    
    # 2. Explore three-ring configurations
    three_splits = [(5, 4, 5), (4, 5, 5), (5, 5, 4), (3, 5, 6), (6, 5, 3)]
    
    for n1, n2, n3 in three_splits:
        if time.time() > deadline - 40:
            break
        pts, sc = _optimize_three_ring(n1, n2, n3, start_time, deadline - 40)
        if pts is not None:
            _update_best(pts)
    
    # 3. Explore four-ring configurations
    four_splits = [(3, 4, 4, 3), (4, 3, 4, 3), (3, 3, 4, 4), (4, 4, 3, 3)]
    
    for n1, n2, n3, n4 in four_splits:
        if time.time() > deadline - 30:
            break
        pts, sc = _optimize_four_ring(n1, n2, n3, n4, start_time, deadline - 30)
        if pts is not None:
            _update_best(pts)
    
    # 4. STRUCTURAL: Dual heptagon with edge-constrained inner points (10D DE search)
    if time.time() < deadline - 35:
        pts, sc = _dual_heptagon_edge_search(start_time, deadline - 35)
        if pts is not None:
            _update_best(pts)
    
    # 5. Hexagon + Heptagon (4D parameter search)
    if time.time() < deadline - 25:
        pts, sc = _hexagon_plus_heptagon_search(start_time, deadline - 25)
        if pts is not None:
            _update_best(pts)
    
    # 6. Differential evolution global search in full 28D space
    if time.time() < deadline - 25:
        pts, sc = _differential_evolution_search(start_time, deadline - 25)
        if pts is not None:
            _update_best(pts)
    
    # 7. Refine best candidate in full space with beta annealing
    if best_points is not None and time.time() < deadline - 15:
        refined = _refine_full_space(best_points, start_time, deadline - 5)
        _update_best(refined)
    
    # 8. Fine-tune with small random perturbations
    if best_points is not None and time.time() < deadline - 3:
        rng = np.random.RandomState(42)
        for _ in range(300):
            if time.time() > deadline - 1:
                break
            pts = best_points + 0.01 * rng.randn(N, 2)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
    
    # 9. Final fallback
    if best_points is None:
        best_points = _regular_polygon(N)
        best_score = _score(best_points)
    
    return best_points
