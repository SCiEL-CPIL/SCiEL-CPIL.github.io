# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _all_triangle_indices(n):
    """Precompute all C(n,3) triangle index triples."""
    return list(combinations(range(n), 3))

def _compute_areas(pts, tri_indices):
    """Compute areas of all triangles given points and precomputed indices.
    Returns array of shape (n_triangles,)."""
    n = len(pts)
    areas = np.empty(len(tri_indices))
    for idx, (i, j, k) in enumerate(tri_indices):
        p1, p2, p3 = pts[i], pts[j], pts[k]
        areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return areas

def _score(pts):
    """Compute the Heilbronn score: min_triangle_area / hull_area."""
    n = pts.shape[0]
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return -1e9
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if a_hull < 1e-10:
            return -1e9
    except Exception:
        return -1e9
    
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area / a_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    
    Uses a two-phase approach:
    1. Smooth gradient-based optimization (L-BFGS-B) with log-sum-exp approximation
    2. Local refinement with Nelder-Mead on the true objective
    """
    n = 13
    start_time = time.time()
    max_time = 610.0  # Safety margin before 630s kill
    
    # Precompute triangle indices
    tri_indices = _all_triangle_indices(n)
    n_tris = len(tri_indices)
    
    # Helper: smooth objective for gradient optimization
    # Uses log-sum-exp to approximate -min(area) smoothly
    def smooth_objective(x, beta=50.0):
        """Minimize -smooth_min_area / hull_area.
        We want to maximize min_area / hull_area, so minimize -(min_area / hull_area).
        Using log-sum-exp: log(sum(exp(-beta * areas))) ~ -min(areas)/beta for large beta
        So smooth_min ~ -log(sum(exp(-beta*areas)))/beta
        """
        pts = x.reshape(n, 2)
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e9
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-10:
                return 1e9
        except Exception:
            return 1e9
        
        # Compute all areas
        areas = np.empty(n_tris)
        for idx, (i, j, k) in enumerate(tri_indices):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        
        # Log-sum-exp approximation of min area
        # min(areas) ≈ -(1/beta) * log(sum(exp(-beta * areas)))
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        smooth_min_area = -log_sum_exp / beta
        
        # We want to maximize smooth_min_area / a_hull
        # So minimize -(smooth_min_area / a_hull)
        return -(smooth_min_area / a_hull)
    
    def true_objective(x):
        """True objective: -min_area / hull_area."""
        pts = x.reshape(n, 2)
        s = _score(pts)
        if s < 0:
            return 1e9
        return -s
    
    # Best tracking
    best_score = -1e9
    best_points = None
    
    # Initializations
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(points_reg.copy())
    
    # 2. Regular 13-gon with slight perturbation
    rng = np.random.default_rng(seed=42)
    initializations.append(points_reg.copy() + rng.normal(0, 0.1, (n, 2)))
    
    # 3. Random points in unit disk
    for _ in range(3):
        angles_r = rng.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(rng.uniform(0, 1, n))
        pts = np.column_stack((radii * np.cos(angles_r), radii * np.sin(angles_r)))
        s = _score(pts)
        if s > 0:
            initializations.append(pts)
    
    # 4. Grid-like initialization
    grid_pts = np.array([
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
        [0.5, 2.5], [1.5, 2.5], [1, 2.5]
    ])
    initializations.append(grid_pts)
    
    for init_pts in initializations:
        if time.time() - start_time > max_time - 30:
            break
        
        x0 = init_pts.flatten()
        
        # Phase 1: Gradient-based optimization with smooth objective
        # Try multiple beta values (annealing)
        betas = [20.0, 50.0, 100.0, 200.0]
        x_current = x0.copy()
        
        for beta in betas:
            if time.time() - start_time > max_time - 20:
                break
            
            # Check time budget for this phase
            phase_start = time.time()
            phase_budget = min(120.0, max_time - 10 - (time.time() - start_time))
            if phase_budget < 5:
                break
            
            def callback(xk):
                nonlocal best_score, best_points
                if time.time() - start_time > max_time - 5:
                    raise RuntimeError("Time limit")
                # Evaluate true score periodically
                s = _score(xk.reshape(n, 2))
                if s > best_score:
                    best_score = s
                    best_points = xk.reshape(n, 2).copy()
            
            try:
                result = minimize(
                    lambda x: smooth_objective(x, beta=beta),
                    x_current,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8},
                    callback=callback
                )
                x_current = result.x
            except RuntimeError:
                pass
            except Exception:
                pass
            
            # Update best
            s = _score(x_current.reshape(n, 2))
            if s > best_score:
                best_score = s
                best_points = x_current.reshape(n, 2).copy()
        
        # Phase 2: Local refinement with Nelder-Mead on true objective
        if time.time() - start_time > max_time - 15:
            break
        
        nm_start = time.time()
        nm_budget = min(60.0, max_time - 5 - (time.time() - start_time))
        if nm_budget < 3:
            break
        
        def nm_callback(xk):
            nonlocal best_score, best_points
            if time.time() - start_time > max_time - 3:
                raise RuntimeError("Time limit")
            s = _score(xk.reshape(n, 2))
            if s > best_score:
                best_score = s
                best_points = xk.reshape(n, 2).copy()
        
        try:
            nm_result = minimize(
                true_objective,
                x_current,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10},
                callback=nm_callback
            )
            s = _score(nm_result.x.reshape(n, 2))
            if s > best_score:
                best_score = s
                best_points = nm_result.x.reshape(n, 2).copy()
        except RuntimeError:
            pass
        except Exception:
            pass
    
    # Final validation
    if best_points is None or best_score <= 0:
        return points_reg
    
    final_score = _score(best_points)
    if final_score <= 0:
        return points_reg
    
    return best_points

# EVOLVE-BLOCK-END
