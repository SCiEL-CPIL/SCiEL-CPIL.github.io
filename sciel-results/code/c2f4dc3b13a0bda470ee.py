"""Heilbronn triangles: maximize the minimum area of triangles formed by 11 points in a unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
TIME_LIMIT = 120.0
START_TIME = time.time()


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _check_time():
    if _remaining_time() < 2.0:
        return True
    return False


def _area_of_triangle(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_area(points):
    """Compute the minimum area among all triangles formed by the points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area_of_triangle(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a


def _min_area_fast(points):
    """Vectorized minimum area computation."""
    n = len(points)
    # All pairs
    min_a = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            # Cross product with all k > j
            diffs = points[j+1:] - pi
            cross = (pj[0] - pi[0]) * diffs[:, 1] - (pj[1] - pi[1]) * diffs[:, 0]
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_a:
                    min_a = local_min
    return min_a


def _objective_flat(x):
    """Objective for optimization: negative min area (to minimize)."""
    points = x.reshape(-1, 2)
    # Check if all points are inside the triangle
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For a point (px, py) to be inside:
    #   py >= 0
    #   py <= sqrt(3)/2
    #   py <= sqrt(3) * px  (below the line from (0,0) to (0.5, sqrt(3)/2))
    #   py <= sqrt(3) * (1 - px)  (below the line from (1,0) to (0.5, sqrt(3)/2))
    h = np.sqrt(3) / 2
    px = points[:, 0]
    py = points[:, 1]
    
    # Check boundaries with some margin
    if np.any(py < -1e-6) or np.any(py > h + 1e-6):
        return 1e6
    if np.any(py > np.sqrt(3) * px + 1e-6):
        return 1e6
    if np.any(py > np.sqrt(3) * (1 - px) + 1e-6):
        return 1e6
    
    return -_min_area_fast(points)


def _soft_min_area(points, temperature=0.01):
    """Differentiable approximation of min area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                areas.append(_area_of_triangle(points[i], points[j], points[k]))
    areas = np.array(areas)
    # Soft minimum: -T * log(sum(exp(-areas/T)))
    # For small T, this approximates min(areas)
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -temperature * log_sum
    return soft_min


def _objective_smooth(x, temperature=0.01):
    """Smooth objective for L-BFGS-B."""
    points = x.reshape(-1, 2)
    h = np.sqrt(3) / 2
    px = points[:, 0]
    py = points[:, 1]
    
    # Penalty for being outside
    penalty = 0
    if np.any(py < 0):
        penalty += 100 * np.sum(py[py < 0] ** 2)
    if np.any(py > h):
        penalty += 100 * np.sum((py[py > h] - h) ** 2)
    if np.any(py > np.sqrt(3) * px):
        penalty += 100 * np.sum((py[py > np.sqrt(3) * px] - np.sqrt(3) * px[py > np.sqrt(3) * px]) ** 2)
    if np.any(py > np.sqrt(3) * (1 - px)):
        penalty += 100 * np.sum((py[py > np.sqrt(3) * (1 - px)] - np.sqrt(3) * (1 - px)[py > np.sqrt(3) * (1 - px)]) ** 2)
    
    return -_soft_min_area(points, temperature) + penalty


def _random_initialization(rng):
    """Generate a random valid point configuration."""
    points = []
    h = np.sqrt(3) / 2
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0 and u > 0.005 and v > 0.005:
            px = u + 0.5 * v
            py = h * v
            points.append([px, py])
    return np.array(points)


def _grid_initialization():
    """Generate a structured grid-like initial configuration."""
    points = []
    h = np.sqrt(3) / 2
    # Try a 3-4-4 arrangement or similar
    # Row 1 (bottom): 4 points
    for i in range(4):
        px = 0.15 + i * 0.23
        py = 0.05
        points.append([px, py])
    # Row 2 (middle): 4 points
    for i in range(4):
        px = 0.2 + i * 0.2
        py = h * 0.4
        points.append([px, py])
    # Row 3 (top): 3 points
    for i in range(3):
        px = 0.25 + i * 0.25
        py = h * 0.7
        points.append([px, py])
    return np.array(points[:N])


def _optimize_from_initial(points, method='L-BFGS-B', max_iter=500):
    """Optimize from a given initial configuration."""
    x0 = points.flatten()
    
    if method == 'L-BFGS-B':
        try:
            result = minimize(
                _objective_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if _check_time():
                return points
            best_points = result.x.reshape(-1, 2)
            return best_points
        except Exception:
            return points
    else:
        return points


def _refine_with_local_search(points, rng, max_iter=100):
    """Local search refinement."""
    current = points.copy()
    current_val = _min_area_fast(current)
    h = np.sqrt(3) / 2
    
    for iteration in range(max_iter):
        if _check_time():
            break
        # Pick a random point to move
        idx = rng.randint(0, N - 1)
        old_pt = current[idx].copy()
        
        # Try several small perturbations
        best_new_pt = old_pt
        best_new_val = current_val
        
        for _ in range(20):
            delta_x = rng.gauss(0, 0.02)
            delta_y = rng.gauss(0, 0.02)
            new_pt = old_pt + np.array([delta_x, delta_y])
            
            # Check bounds
            if new_pt[1] < 0 or new_pt[1] > h:
                continue
            if new_pt[1] > np.sqrt(3) * new_pt[0]:
                continue
            if new_pt[1] > np.sqrt(3) * (1 - new_pt[0]):
                continue
            
            test_points = current.copy()
            test_points[idx] = new_pt
            val = _min_area_fast(test_points)
            
            if val > best_new_val:
                best_new_val = val
                best_new_pt = new_pt
        
        if best_new_val > current_val:
            current[idx] = best_new_pt
            current_val = best_new_val
    
    return current


def construct():
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.RandomState(42)
    
    # Start with a reasonable initial configuration
    best_points = _grid_initialization()
    best_val = _min_area_fast(best_points)
    
    # Try a few random initializations
    for i in range(3):
        if _check_time():
            break
        pts = _random_initialization(rng)
        val = _min_area_fast(pts)
        if val > best_val:
            best_val = val
            best_points = pts
    
    # Optimize from the best initial point using L-BFGS-B
    if _remaining_time() > 30:
        optimized = _optimize_from_initial(best_points, method='L-BFGS-B', max_iter=300)
        opt_val = _min_area_fast(optimized)
        if opt_val > best_val:
            best_val = opt_val
            best_points = optimized
    
    # Local search refinement
    if _remaining_time() > 15:
        refined = _refine_with_local_search(best_points, rng, max_iter=200)
        ref_val = _min_area_fast(refined)
        if ref_val > best_val:
            best_val = ref_val
            best_points = refined
    
    # Try a few more random restarts with optimization if time allows
    for i in range(5):
        if _check_time():
            break
        pts = _random_initialization(rng)
        
        # Quick local search
        refined = _refine_with_local_search(pts, rng, max_iter=50)
        val = _min_area_fast(refined)
        
        if val > best_val:
            best_val = val
            best_points = refined
    
    # Final local search on the best
    if _remaining_time() > 10:
        final = _refine_with_local_search(best_points, rng, max_iter=100)
        final_val = _min_area_fast(final)
        if final_val > best_val:
            best_points = final
    
    return np.array(best_points)
