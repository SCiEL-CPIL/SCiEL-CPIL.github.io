import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_ratio_smooth(x, tau=10.0):
    """
    Smooth approximation of the ratio of max to min squared distance.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return np.log(soft_max) - np.log(soft_min)

def raw_score(points):
    """Compute the true squared distance ratio."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def get_tetrahedron_vertices(center=np.zeros(3), scale=1.0):
    """Return vertices of a regular tetrahedron centered at 'center'."""
    base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    base = base / np.sqrt(3.0)
    return center + scale * base

def rotate_matrix(angle, axis):
    """Return 3D rotation matrix for angle around axis."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def stella_octangula_points(rot_angle, scale_ratio=1.0, outer_scale=1.0):
    """
    Generate 14 points from two interpenetrating regular tetrahedra (stella octangula).
    rot_angle: relative rotation angle between the two tetrahedra
    scale_ratio: ratio of inner to outer tetrahedron scale
    outer_scale: overall scale
    Returns 14 points (4 outer + 4 inner + 6 additional for filling).
    Actually, two tetrahedra give 8 points. We need 14, so we add 6 more.
    Strategy: use two tetrahedra (8 pts) + 6 points on a ring or additional structure.
    Better: use the known optimal 14-point configuration in 3D.
    """
    # Two regular tetrahedra forming stella octangula
    tet1 = get_tetrahedron_vertices(np.zeros(3), outer_scale)
    R = rotate_matrix(rot_angle, [0, 0, 1])
    tet2 = (R @ get_tetrahedron_vertices(np.zeros(3), outer_scale * scale_ratio).T).T
    
    points = np.vstack([tet1, tet2])
    
    # Need 6 more points to reach 14. Add them on a ring in the xy-plane
    # or use a different construction. Let's add 6 points at the mid-edges
    # or on a hexagonal ring.
    # For now, place 6 points on a circle in the xy-plane at radius r
    ring_radius = outer_scale * 0.7
    ring_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring_pts = np.column_stack([
        ring_radius * np.cos(ring_angles),
        ring_radius * np.sin(ring_angles),
        np.zeros(6)
    ])
    
    points = np.vstack([points, ring_pts])
    points = points - np.mean(points, axis=0)
    return points

def params_to_points(params):
    """
    Convert parameters to 14 points.
    params: [scale1, scale2, rot_angle, rot_axis_x, rot_axis_y, rot_axis_z, tx, ty, tz]
    """
    scale1 = params[0]
    scale2 = params[1]
    rot_angle = params[2]
    axis = params[3:6]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        axis = np.array([0, 0, 1])
    else:
        axis = axis / axis_norm
    translation = params[6:9]
    
    tet1 = get_tetrahedron_vertices(np.zeros(3), scale1)
    R = rotate_matrix(rot_angle, axis)
    tet2 = (R @ get_tetrahedron_vertices(np.zeros(3), scale2).T).T + translation
    
    points = np.vstack([tet1, tet2])
    return points

def objective_from_params(params, tau=10.0):
    """Objective function for differential evolution over 9 parameters."""
    points = params_to_points(params)
    points = points - np.mean(points, axis=0)
    return objective_ratio_smooth(points.flatten(), tau=tau)

def fibonacci_sphere(n, radius=1.0):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        points.append([radius * r * np.cos(theta), radius * r * np.sin(theta), radius * z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 140.0
    
    best_score = np.inf
    best_points = None
    
    # Phase 1: Try stella octangula with optimized rotation angle
    # The optimal relative rotation for two tetrahedra to minimize max/min ratio
    # is around the angle that makes the compound most symmetric
    try:
        # Optimize just the rotation angle for stella octangula (2 tetrahedra + ring)
        def stella_obj(angle):
            pts = stella_octangula_points(angle, scale_ratio=1.0, outer_scale=1.0)
            return objective_ratio_smooth(pts.flatten(), tau=50.0)
        
        res_stella = minimize(
            stella_obj,
            x0=[0.5],
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if res_stella.x is not None:
            pts_stella = stella_octangula_points(res_stella.x[0], scale_ratio=1.0, outer_scale=1.0)
            score_stella = raw_score(pts_stella)
            if score_stella < best_score:
                best_score = score_stella
                best_points = pts_stella
    except Exception:
        pass
    
    # Phase 2: Differential evolution over 9 parameters
    bounds = [
        (0.3, 3.5),
        (0.3, 3.5),
        (0.0, np.pi),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-2.5, 2.5),
    ]
    
    try:
        de_result = differential_evolution(
            objective_from_params,
            bounds=bounds,
            seed=42,
            maxiter=60,
            tol=1e-7,
            popsize=15,
            polish=True,
            init='sobol'
        )
        
        if de_result.x is not None:
            pts_de = params_to_points(de_result.x)
            pts_de = pts_de - np.mean(pts_de, axis=0)
            score_de = raw_score(pts_de)
            if score_de < best_score:
                best_score = score_de
                best_points = pts_de
    except Exception:
        pass
    
    if time.time() - start_time > timeout - 40:
        if best_points is not None:
            return best_points
        # Fallback: Fibonacci sphere
        pts_fib = fibonacci_sphere(N, radius=1.0)
        pts_fib = pts_fib - np.mean(pts_fib, axis=0)
        return pts_fib
    
    # Phase 3: Refine candidates using multi-stage L-BFGS-B
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    
    # Fibonacci sphere
    pts_fib = fibonacci_sphere(N, radius=1.0)
    pts_fib = pts_fib - np.mean(pts_fib, axis=0)
    candidates.append(pts_fib.flatten())
    
    # Grid-based initialization
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    candidates.append(init_grid.flatten())
    
    # Icosahedron + 2 extra
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    ico_full = np.vstack([ico_verts, [[0, 0, 0.5], [0, 0, -0.5]]])
    ico_full = ico_full - np.mean(ico_full, axis=0)
    candidates.append(ico_full.flatten())
    
    # Random starts
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
    
    tau_schedule = [5.0, 20.0, 100.0, 500.0]
    maxiter_schedule = [200, 300, 400, 300]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 30:
            break
            
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 30:
                    break
                
                obj = partial(objective_ratio_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj, 
                    x_curr, 
                    method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.x is not None:
                    x_curr = res.x
            
            pts = x_curr.reshape(N, D)
            score = raw_score(pts)
            if score < best_score:
                best_score = score
                best_points = pts
                
        except Exception:
            pass
    
    # Phase 4: Powell's method polish on centered coordinates
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_curr = best_points.flatten()
            
            def centered_obj(x):
                pts = x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                return objective_ratio_smooth(pts.flatten(), tau=1000.0)
            
            res_powell = minimize(
                centered_obj,
                x_curr,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            
            if res_powell.x is not None:
                pts_powell = res_powell.x.reshape(N, D)
                pts_powell = pts_powell - np.mean(pts_powell, axis=0)
                score_powell = raw_score(pts_powell)
                if score_powell < best_score:
                    best_score = score_powell
                    best_points = pts_powell
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise polish
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            x_curr = best_points.flatten()
            obj_final = partial(objective_ratio_smooth, tau=5000.0)
            res_final = minimize(
                obj_final,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                score_final = raw_score(pts_final)
                if score_final < best_score:
                    best_score = score_final
                    best_points = pts_final
        except Exception:
            pass
    
    if best_points is None:
        pts_fib = fibonacci_sphere(N, radius=1.0)
        pts_fib = pts_fib - np.mean(pts_fib, axis=0)
        best_points = pts_fib
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
