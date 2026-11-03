import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
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
        """Soft-min objective for L-BFGS."""
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
        return -soft_min_area

    def _hard_objective(points_flat):
        """Hard-min objective for Powell/Nelder-Mead."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
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
        
    # Add random perturbations to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    def _refine_critical_moves(points, max_iters=50):
        """
        Custom local search: perturb vertices of the minimum-area triangles.
        """
        current_pts = points.copy()
        current_score = compute_score(current_pts)
        if current_score <= 0:
            return current_pts, current_score
            
        rng_local = np.random.RandomState(int(time.time() * 1000) % 10000)
        
        for _ in range(max_iters):
            if time.time() > start_time + time_limit - 2:
                break
                
            # Get areas to find critical triangles
            pts_norm = _normalize_points(current_pts)
            x = pts_norm[:, 0]
            y = pts_norm[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            # Find indices of the smallest triangle
            min_idx = np.argmin(areas)
            # Also consider the next few smallest to avoid getting stuck in a single triangle's local min
            # Get unique vertices involved in the 5 smallest triangles
            smallest_indices = np.argsort(areas)[:5]
            critical_verts = set()
            for idx in smallest_indices:
                critical_verts.add(tri_i[idx])
                critical_verts.add(tri_j[idx])
                critical_verts.add(tri_k[idx])
            
            # Randomly perturb these critical vertices
            new_pts = current_pts.copy()
            for v in critical_verts:
                # Small Gaussian perturbation
                noise = rng_local.randn(2) * 0.01
                new_pts[v] += noise
            
            new_score = compute_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # If no improvement, try a slightly larger perturbation or different random seed
                # Here we just continue, relying on the randomness
                pass
                
        return current_pts, current_score

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective
        betas = [20.0, 50.0, 100.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 5:
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
        
        # Step 2: Powell with hard objective
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            current = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Step 3: Critical Move Refinement
        if time.time() > start_time + time_limit - 1:
            break
        current, score = _refine_critical_moves(current, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_points = current.copy()
            
        # Step 4: Final Polish with Powell on the refined point
        if time.time() > start_time + time_limit - 0.5:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 20, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            final_score = compute_score(candidate)
            if final_score > best_score:
                best_score = final_score
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = initializations[0]

    return best_points
