import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

# Pre-compute indices for all triples (i, j, k) with i < j < k
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """
    Calculate areas of all triangles formed by the points.
    points: (N, 2) array
    returns: (286,) array of areas
    """
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        if len(points) < 3:
            return 0.0
        # Check for degeneracy
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / hull area.
    """
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

def _neg_score(points):
    """
    Negative score for minimization.
    """
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _smooth_objective(x_flat, p=50):
    """
    Smooth surrogate using log-sum-exp based soft-min for better gradients.
    x_flat: (2N,) array of Cartesian coordinates
    """
    points = x_flat.reshape(N, 2)
    # Center the points to remove translation invariance and keep numbers small
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

def _coordinate_descent_polish(points, deadline, max_iter=1000):
    """
    Coordinate descent on the hard objective: move each point's x and y independently.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Start with larger steps, then refine
    steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in steps:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for coord in range(2):
                    # Try positive and negative moves
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * step
                        trial_s = _score(trial)
                        if trial_s > best_s + 1e-14:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_pts, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # 1. Initial diverse starting points
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Random points
    np.random.seed(42)
    pts_random = np.random.randn(N, 2)
    
    # Two-ring configuration (6 outer, 6 inner, 1 center)
    # Outer ring
    ang_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts_outer = np.column_stack([np.cos(ang_outer), np.sin(ang_outer)])
    # Inner ring
    ang_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts_inner = np.column_stack([0.5 * np.cos(ang_inner), 0.5 * np.sin(ang_inner)])
    # Center
    pts_center = np.array([[0.0, 0.0]])
    pts_ring = np.vstack([pts_outer, pts_inner, pts_center])
    
    candidates = [pts_regular, pts_random, pts_ring]
    
    best_points = None
    best_score = 0.0
    
    # Evaluate initial candidates
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
    
    # 2. Global Search: Differential Evolution on Smooth Objective
    # Bounds: points should be within a reasonable range, e.g., [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(
                _smooth_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-5,
                popsize=15,
                polish=False, # We will polish manually
                updateinterval=0, # Disable progress printing for speed
                workers=1
            )
            de_points = de_res.x.reshape(N, 2)
            de_score = _score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
        except Exception:
            pass

    # 3. Local Refinement on Best Point
    # Use the best point found so far (either initial, DE result, or previous best)
    current_points = best_points.copy()
    
    # Phase 1: Powell on Smooth Objective
    if time.time() < deadline - 15:
        try:
            res_powell = minimize(
                lambda x: _smooth_objective(x, p=80),
                current_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            powell_points = res_powell.x.reshape(N, 2)
            powell_score = _score(powell_points)
            if powell_score > best_score:
                best_score = powell_score
                best_points = powell_points.copy()
                current_points = powell_points.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on Hard Objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                current_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            nm_points = res_nm.x.reshape(N, 2)
            nm_score = _score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
                current_points = nm_points.copy()
        except Exception:
            pass

    # Phase 3: Coordinate Descent Polish on Hard Objective
    if time.time() < deadline - 2:
        try:
            cd_points, cd_score = _coordinate_descent_polish(current_points, deadline)
            if cd_score > best_score:
                best_score = cd_score
                best_points = cd_points.copy()
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    points = construct()
    score = _score(points)
    print(f"Score: {score}")
