# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_strictly_convex(points: np.ndarray) -> bool:
    """Check if the polygon defined by points is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return False

    # Check cross products for consistent sign (convexity)
    # For a CCW convex polygon, cross product of consecutive edges should be positive
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    # All cross products must be non-zero and of the same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return False
        
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    else:
        return False


def _harmonic_perturbation(polygon: np.ndarray, freq: int = 3, amp: float = 0.02) -> np.ndarray:
    """Add a harmonic perturbation to the polygon vertices."""
    n = len(polygon)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Radial perturbation: modulate radius by harmonic
    radii = np.sqrt((polygon ** 2).sum(axis=1))
    perturbed_radii = radii * (1 + amp * np.sin(freq * angles))
    theta = np.arctan2(polygon[:, 1], polygon[:, 0])
    new_polygon = np.stack([perturbed_radii * np.cos(theta), perturbed_radii * np.sin(theta)], axis=1)
    return new_polygon


def _coordinate_descent(polygon: np.ndarray, max_iter: int = 100, step: float = 0.01, deadline: float = None) -> np.ndarray:
    """Coordinate descent optimization on individual vertex coordinates."""
    n = len(polygon)
    current = polygon.copy()
    current_score = _score_polygon(current)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for coord in range(2):
                # Try moving vertex i in coordinate coord
                for direction in [-1, 1]:
                    trial = current.copy()
                    trial[i, coord] += direction * step
                    if not np.all(np.isfinite(trial)):
                        continue
                    if not _is_strictly_convex(trial):
                        continue
                    trial_score = _score_polygon(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    return current

def _score_polygon(polygon: np.ndarray) -> float:
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    if n != 20:
        return 0.0
        
    # Check convexity first to avoid wasting time on invalid shapes
    if not _is_strictly_convex(polygon):
        return 0.0

    # Normalize: subtract centroid, divide by mean distance
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    min_vertex_score = 1.0
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        # Remove self-distance
        d = np.delete(d, i)
        d = np.sort(d)
        
        # Check if any distance < 0.05
        if d[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        if D < 0.1:
            return 0.0
        
        # Select four distances
        if k == 0:
            selected = d[0:4]
        elif k == 17:  # k=17 means d[17] and d[18] are the closest gap
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        vertex_score = min(1.0, 1.0 / raw)
        min_vertex_score = min(min_vertex_score, vertex_score)
    
    return min_vertex_score

def _superellipse_params_to_polygon(params: np.ndarray) -> np.ndarray:
    """Convert superellipse parameters [a, b, n, theta] to a 20x2 polygon."""
    a, b, n_exp, theta = params
    if a <= 0 or b <= 0 or n_exp <= 0:
        return None
    
    # Generate 20 points on the superellipse
    # t is the angle parameter
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    
    # Superellipse equation: |x/a|^n + |y/b|^n = 1
    # Parametric: x = a * sign(cos(t)) * |cos(t)|^(2/n), y = b * sign(sin(t)) * |sin(t)|^(2/n)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n_exp)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n_exp)
    
    points = np.stack([x, y], axis=1)
    
    # Rotate by theta
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    points = points @ rot.T
    
    return points

def _neg_superellipse_score(params: np.ndarray) -> float:
    """Negative score for superellipse parameters."""
    polygon = _superellipse_params_to_polygon(params)
    if polygon is None:
        return 100.0
    if not _is_strictly_convex(polygon):
        return 10.0 + np.sum(np.abs(polygon)) * 0.1 # Penalty for non-convex
    score = _score_polygon(polygon)
    return -score


def _neg_score_smooth(params: np.ndarray) -> float:
    """Smooth negative score for local optimization."""
    polygon = params.reshape(20, 2)
    if not np.all(np.isfinite(polygon)):
        return 1000.0
    
    n = 20
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    if np.any(np.abs(cross_products) < 1e-10):
        return 100.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 100.0

    score = _score_polygon(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses superellipse parameterization to find a good initial shape,
    then refines with local coordinate optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = 0.0
    best_polygon = None
    
    # 1. Search for best superellipse parameters
    # Initial guesses for [a, b, n_exp, theta]
    # a, b are semi-axes, n_exp is the exponent, theta is rotation
    initial_guesses = [
        [1.0, 1.0, 2.0, 0.0],      # Circle
        [1.0, 1.0, 4.0, 0.0],      # Superellipse n=4
        [1.0, 1.0, 6.0, 0.0],      # Superellipse n=6
        [1.0, 1.0, 8.0, 0.0],      # Superellipse n=8
        [1.0, 1.0, 10.0, 0.0],     # Superellipse n=10
        [1.0, 0.8, 4.0, 0.0],      # Ellipse-like
        [1.2, 0.8, 4.0, 0.0],      # Stretched
        [1.0, 1.0, 5.0, np.pi/4],  # Rotated
    ]
    
    best_superellipse_score = -100.0
    best_superellipse_params = None
    
    for guess in initial_guesses:
        if time.time() > deadline - 100:
            break
        try:
            result = minimize(_neg_superellipse_score, guess, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if result.fun < best_superellipse_score:
                best_superellipse_score = result.fun
                best_superellipse_params = result.x
        except Exception:
            pass
    
    # Generate polygon from best superellipse
    if best_superellipse_params is not None:
        polygon = _superellipse_params_to_polygon(best_superellipse_params)
        if polygon is not None and _is_strictly_convex(polygon):
            score = _score_polygon(polygon)
            if score > best_score:
                best_score = score
                best_polygon = polygon
    
    # Also try regular polygon as a fallback/alternative
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    regular_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    reg_score = _score_polygon(regular_polygon)
    if reg_score > best_score:
        best_score = reg_score
        best_polygon = regular_polygon

    if best_polygon is None:
        # Fallback to regular polygon if all else fails
        best_polygon = regular_polygon
        best_score = reg_score

    # 2. Local coordinate optimization (Polish)
    # Start from the best polygon found so far
    current_polygon = best_polygon.copy()
    
    # Run local optimization using smooth score
    if time.time() < deadline - 50:
        try:
            result = minimize(_neg_score_smooth, current_polygon.flatten(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            candidate = result.x.reshape(20, 2)
            if _is_strictly_convex(candidate) and np.all(np.isfinite(candidate)):
                cand_score = _score_polygon(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_polygon = candidate
        except Exception:
            pass
    
    # 2b. Coordinate descent refinement
    if time.time() < deadline - 30:
        try:
            cd_result = _coordinate_descent(best_polygon, max_iter=50, step=0.005, deadline=deadline - 25)
            cd_score = _score_polygon(cd_result)
            if cd_score > best_score:
                best_score = cd_score
                best_polygon = cd_result
        except Exception:
            pass

    # 3. Try harmonic perturbations of the best polygon if time allows
    if time.time() < deadline - 20:
        for freq in [2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            for amp in [0.01, 0.02, 0.03]:
                if time.time() > deadline - 10:
                    break
                try:
                    perturbed = _harmonic_perturbation(best_polygon, freq=freq, amp=amp)
                    if _is_strictly_convex(perturbed):
                        p_score = _score_polygon(perturbed)
                        if p_score > best_score:
                            best_score = p_score
                            best_polygon = perturbed
                            # Quick polish with coordinate descent
                            if time.time() < deadline - 15:
                                cd_res = _coordinate_descent(best_polygon, max_iter=30, step=0.003, deadline=deadline - 10)
                                cd_sc = _score_polygon(cd_res)
                                if cd_sc > best_score:
                                    best_score = cd_sc
                                    best_polygon = cd_res
                except Exception:
                    pass
    
    # 4. Also try random perturbations of the regular polygon
    if time.time() < deadline - 15:
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            noise = np.random.normal(0, 0.03, (20, 2))
            perturbed = regular_polygon + noise
            if _is_strictly_convex(perturbed):
                p_score = _score_polygon(perturbed)
                if p_score > best_score:
                    best_score = p_score
                    best_polygon = perturbed
                    # Quick polish
                    if time.time() < deadline - 12:
                        try:
                            res2 = minimize(_neg_score_smooth, perturbed.flatten(), method='Nelder-Mead',
                                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                            cand2 = res2.x.reshape(20, 2)
                            if _is_strictly_convex(cand2):
                                s2 = _score_polygon(cand2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_polygon = cand2
                        except Exception:
                            pass

    return best_polygon


# EVOLVE-BLOCK-END
