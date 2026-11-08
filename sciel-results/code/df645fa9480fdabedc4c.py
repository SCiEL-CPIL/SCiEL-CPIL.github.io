import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    elapsed = lambda: time.time() - start_time

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            # Center points to improve numerical stability
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=50.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        # Since hull area is 1, score is just min_area
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        # If score is 0, return a large positive number (bad)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 1 center + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
        
    # 9. 5 outer + 8 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 10. 4 outer + 9 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r)
        initializations.append(np.vstack([outer, inner]))
    
    # 11. Nested polygons with more radius options
    for n_outer in [13, 12, 11, 10, 9]:
        n_inner = 13 - n_outer
        if n_inner == 0:
            continue
        for r in [0.2, 0.4, 0.6, 0.8]:
            outer = _regular_polygon(n_outer, 1.0)
            if n_inner == 1:
                inner = np.array([[0.0, 0.0]])
            else:
                inner = _regular_polygon(n_inner, r)
            initializations.append(np.vstack([outer, inner]))
    
    # 12. Grid-based initializations
    for scale in [0.8, 1.0, 1.2]:
        # 4x4 grid minus one corner
        xs = np.array([-1.5, -0.5, 0.5, 1.5]) * scale
        ys = np.array([-1.5, -0.5, 0.5, 1.5]) * scale
        grid = np.array([[x, y] for x in xs for y in ys])
        # Take 13 points from the 16-point grid
        idx = list(range(13))
        initializations.append(grid[idx])
    
    # 13. Random but structured: mix of concentric rings
    rng = np.random.RandomState(42)
    for trial in range(10):
        n1 = rng.randint(3, 8)
        n2 = rng.randint(2, 13 - n1)
        n3 = 13 - n1 - n2
        r1 = rng.uniform(0.3, 0.6)
        r2 = rng.uniform(0.6, 0.9)
        r3 = 1.0
        pts1 = _regular_polygon(n1, r1) if n1 > 0 else np.array([])
        pts2 = _regular_polygon(n2, r2) if n2 > 0 else np.array([])
        pts3 = _regular_polygon(n3, r3) if n3 > 0 else np.array([])
        if len(pts1) > 0 and len(pts2) > 0 and len(pts3) > 0:
            init = np.vstack([pts1, pts2, pts3])
        elif len(pts1) > 0 and len(pts2) > 0:
            init = np.vstack([pts1, pts2])
        elif len(pts1) > 0 and len(pts3) > 0:
            init = np.vstack([pts1, pts3])
        elif len(pts2) > 0 and len(pts3) > 0:
            init = np.vstack([pts2, pts3])
        else:
            continue
        if len(init) == 13:
            initializations.append(init)
    
    # Add some random perturbations to top symmetric configs to break symmetry if beneficial
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # Sort initializations by initial score to prioritize good starting points
    scored_inits = []
    for init in initializations:
        s = compute_score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Time budget allocation
    total_time = time_limit - 5  # Reserve 5s for final checks
    per_init_time = total_time / len(scored_inits)
    
    # Optimize each initialization
    for idx, (init_score, init) in enumerate(scored_inits):
        if elapsed() > time_limit - 10:
            break
            
        # If initial score is already very good, spend less time
        if init_score > 0.08:
            per_init_budget = min(per_init_time, 3.0)
        elif init_score > 0.06:
            per_init_budget = min(per_init_time, 5.0)
        else:
            per_init_budget = per_init_time
        
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        # Start with low beta to explore, increase to refine
        betas = [10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
        
        for beta in betas:
            if elapsed() > time_limit - 8:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Powell with hard objective (robust local search)
        if elapsed() > time_limit - 5:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 80, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Step 3: Nelder-Mead final polish
        if elapsed() > time_limit - 3:
            break
        if best_points is not None:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
