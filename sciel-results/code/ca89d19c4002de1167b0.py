import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for vectorized area calculation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute areas of all triangles formed by pairs of points."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute area of the convex hull of the points."""
    try:
        if len(points) < 3:
            return 0.0
        # Check if points are collinear or identical
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _cartesian_to_polar(points):
    """Convert Cartesian points to polar representation (sorted by angle)."""
    points = points - np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    # Sort by angle
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    # Normalize angles to [0, 2*pi)
    angles_sorted = angles_sorted % (2 * np.pi)
    return np.concatenate([angles_sorted, radii_sorted])

def _polar_objective(polar):
    """Objective for polar-parameterized DE: minimize negative score."""
    points = _polar_to_cartesian(polar)
    # Center the points
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _smooth_objective(x_flat, p=50):
    """Smooth surrogate using log-sum-exp based soft-min for better gradients."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp based soft-min: -log(sum(exp(-p * areas))) / p
    # This gives a smooth approximation to min(areas)
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective: move each point's x and y independently."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations in this coordinate
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell on smooth objective (log-sum-exp based)
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=80),
                best_x,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish on hard objective
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _simulated_annealing(points, deadline):
    """Simulated annealing in Cartesian space with adaptive step size."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    time_budget = deadline - 3.0
    total_time = max(0.5, time_budget - time.time())
    est_iters = int(total_time * 800)
    est_iters = max(500, min(est_iters, 60000))
    
    T0 = 0.03
    T_end = 0.00005
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    for i in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** i)
        
        # Perturb a random point
        idx = np.random.randint(0, N)
        dim = np.random.randint(0, 2)
        step = np.random.normal(0, 0.05)
        
        trial = current.copy()
        trial[idx, dim] += step
        
        trial_score = _score(trial)
        
        if trial_score > current_score:
            current = trial
            current_score = trial_score
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            # Accept with probability exp((trial_score - current_score) / T)
            delta = trial_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current = trial
                    current_score = trial_score
    
    return best, best_s

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # --- Phase 1: Global Search with Differential Evolution in Polar Space ---
    # Polar parameterization helps with rotational symmetry and keeps points in a bounded region naturally.
    if time.time() < deadline - 40:
        try:
            # Bounds for polar: angles in [0, 2*pi], radii in [0, 2]
            bounds_polar = [(0, 2 * np.pi)] * N + [(0, 2.0)] * N
            de_params = {
                'maxiter': 50,
                'popsize': 15,
                'mutation': (0.5, 1.0),
                'recombination': 0.7,
                'tol': 1e-6,
                'seed': 42,
                'polish': False,
                'init': 'sobol',
            }
            
            result = differential_evolution(
                _polar_objective,
                bounds_polar,
                **de_params
            )
            
            if result.success or result.fun < 1e9:
                de_polar = result.x
                de_points = _polar_to_cartesian(de_polar)
                de_points = de_points - np.mean(de_points, axis=0)
                _update_best(de_points)
                
        except Exception as e:
            pass

    # --- Phase 2: Local Refinement from Multiple Starts ---
    start_points = []
    
    # 1. Add DE result if we have it
    if best_points is not None:
        start_points.append(best_points.flatten())
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(regular_13.flatten())
    
    # 3. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 1:
            continue
        for r_in in [0.3, 0.5, 0.7]:
            for offset in [0, np.pi / n_out]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                start_points.append(pts.flatten())
    
    # 4. Random starts
    for _ in range(3):
        start_points.append(np.random.uniform(-1, 1, (N, 2)).flatten())

    # Optimization Pipeline for each start
    for x0_flat in start_points:
        if time.time() > deadline - 15:
            break
            
        current_x = x0_flat.copy()
        current_score = _score(current_x.reshape(N, 2))
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_x.reshape(N, 2).copy()
        
        # Run local polish with a time budget
        time_budget = min(10.0, deadline - time.time() - 5)
        if time_budget > 2.0:
            polished_points, polished_score = _local_polish(current_x.reshape(N, 2), time_budget)
            _update_best(polished_points)
            
            # Also try simulated annealing if time permits
            if time.time() < deadline - 10:
                sa_points, sa_score = _simulated_annealing(polished_points, deadline - 10)
                _update_best(sa_points)

    # Final global check: if we have time, do one more DE run from the best point neighborhood?
    # Or just return best.
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
