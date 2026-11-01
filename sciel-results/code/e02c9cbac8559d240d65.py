import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
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
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp (numerically stable)
    # We want to minimize the soft min, which is equivalent to maximizing the soft max of negative areas?
    # No, min(x) = -max(-x).
    # Soft min(x) = -1/beta * log(sum(exp(-beta * x)))
    # Let's use the standard soft min formula:
    # m = min(x)
    # soft_min = m - (1/beta) * log(sum(exp(-beta * (x - m))))
    
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(np.unique(points, axis=0)) < 3:
            return 0.0
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
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

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _angles_to_points(angles):
    """Convert angles to points on unit circle."""
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _neg_soft_score_angles(angles_flat, beta=100.0):
    """
    Negative soft-score for optimization in angle space.
    We assume points are on the unit circle.
    """
    angles = angles_flat
    points = _angles_to_points(angles)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        # We want to maximize smta/ha.
        # Since ha is constant for points on unit circle (hull is regular polygon approx),
        # maximizing smta is sufficient, but let's be precise.
        return -(smta / ha)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return deadline - time.time()

    # Phase 1: Initial Candidates
    # 1. Regular 14-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = _angles_to_points(angles_reg)
    _update_best(pts_reg, _score(pts_reg))
    
    # 2. Random initializations
    rng = np.random.RandomState(42)
    
    # We will run multiple optimization instances
    # To speed up, we limit the number of restarts based on time.
    
    # Pre-compute some good starting points:
    # - Regular polygon
    # - Slightly perturbed regular polygon
    # - Random sorted angles
    
    start_angles_list = [angles_reg]
    
    # Add a few random sorted angle sets
    for _ in range(10):
        rand_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Ensure no two points are too close
        diffs = np.diff(np.concatenate([rand_angles, [rand_angles[0] + 2*np.pi]]))
        if np.min(diffs) > 0.1:
            start_angles_list.append(rand_angles)

    # Phase 2: Optimize in Angle Space
    # We use L-BFGS-B to optimize the angles.
    # Since angles are periodic, we can't directly use L-BFGS-B without care,
    # but for small perturbations from a good start, it works well.
    # Alternatively, we can optimize the *differences* (gaps) between angles.
    # Let gaps g_i >= 0, sum(g_i) = 2*pi.
    # This is a constrained optimization.
    # Simpler: Just optimize angles directly, assuming they stay sorted and don't wrap around too much.
    # Given the symmetry, the optimal solution is likely close to regular, so angles won't jump.
    
    # Let's try optimizing the gaps instead, it's more robust to rotation.
    # g = [g_0, ..., g_{N-1}]
    # angles[0] = 0 (fixed by rotation invariance)
    # angles[i] = sum(g[0:i])
    # Constraint: sum(g) = 2*pi, g > 0.
    # We can parameterize g using softmax or just use a penalty.
    # Actually, L-BFGS-B can handle bounds. But sum(g)=2*pi is an equality constraint.
    # Let's stick to angles for simplicity, but initialize them sorted.
    
    for init_angles in start_angles_list:
        if _time_left() < 10:
            break
            
        # Progressive beta for soft min
        current_angles = init_angles.copy()
        
        for beta in [50.0, 100.0, 200.0, 500.0]:
            if _time_left() < 5:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x, b=beta: _neg_soft_score_angles(x, b),
                    current_angles,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                new_angles = res.x
                # Sort angles to ensure consistent ordering for next step?
                # The optimizer might scramble the order if points cross.
                # But for small steps, they shouldn't.
                # Let's sort them to be safe for the next iteration.
                new_angles = np.sort(new_angles)
                
                pts = _angles_to_points(new_angles)
                sc = _score(pts)
                if sc > best_score:
                    _update_best(pts, sc)
                    current_angles = new_angles
            except Exception:
                pass
        
        # Nelder-Mead polish on exact score in angle space
        if _time_left() > 5:
            try:
                res_nm = scipy.optimize.minimize(
                    lambda x: _neg_score(_angles_to_points(x).flatten()),
                    current_angles,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_angles = np.sort(res_nm.x)
                pts = _angles_to_points(new_angles)
                sc = _score(pts)
                if sc > best_score:
                    _update_best(pts, sc)
            except Exception:
                pass

    # Phase 3: If time remains, try a few random restarts with local optimization
    if _time_left() > 20 and best_points is not None:
        # Generate more diverse random starts
        for _ in range(5):
            if _time_left() < 10:
                break
            rand_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            
            try:
                res = scipy.optimize.minimize(
                    lambda x: _neg_soft_score_angles(x, 100.0),
                    rand_angles,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                new_angles = np.sort(res.x)
                pts = _angles_to_points(new_angles)
                sc = _score(pts)
                if sc > best_score:
                    _update_best(pts, sc)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_angles_to_points(angles_reg))
    else:
        best_points = _normalize(best_points)
        
    return best_points
