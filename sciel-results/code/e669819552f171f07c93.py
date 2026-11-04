"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - \
                        (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0])
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    cha = _convex_hull_area(pts)
    if cha < 1e-10:
        return 0.0
    return _min_triangle_area(pts) / cha

def _negative_score(pts):
    return -_score(pts)

def _points_from_angles(angles):
    """Convert angles to points on unit circle."""
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _evaluate_angles(angles):
    pts = _points_from_angles(angles)
    return _negative_score(pts)

def _local_search(angles, max_iter=5000, step=0.01):
    """Simple coordinate descent / local search on angles."""
    best = _evaluate_angles(angles)
    current = angles.copy()
    for it in range(max_iter):
        improved = False
        for i in range(len(current)):
            for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                trial = current.copy()
                trial[i] += delta
                trial[i] = trial[i] % (2 * np.pi)
                val = _evaluate_angles(trial)
                if val < best:
                    best = val
                    current = trial
                    improved = True
                    break
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-8:
                break
    return current, -best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Start with evenly spaced angles on unit circle
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts0 = _points_from_angles(angles0)
    best_score = _score(pts0)
    best_pts = pts0.copy()
    
    # Local search from the evenly-spaced configuration
    angles_ls, score_ls = _local_search(angles0, max_iter=3000, step=0.05)
    if score_ls > best_score:
        best_score = score_ls
        best_pts = _points_from_angles(angles_ls)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_pts
    
    # Try differential evolution with limited budget
    try:
        from scipy.optimize import differential_evolution
        bounds = [(0, 2 * np.pi)] * N
        result = differential_evolution(
            _evaluate_angles,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube'
        )
        if -result.fun > best_score:
            best_score = -result.fun
            best_pts = _points_from_angles(result.x)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        return best_pts
    
    # Final local polish
    angles_current = np.arctan2(best_pts[:, 1], best_pts[:, 0])
    angles_current[angles_current < 0] += 2 * np.pi
    angles_final, score_final = _local_search(angles_current, max_iter=2000, step=0.01)
    if score_final > best_score:
        best_score = score_final
        best_pts = _points_from_angles(angles_final)
    
    return best_pts

# EVOLVE-BLOCK-END
