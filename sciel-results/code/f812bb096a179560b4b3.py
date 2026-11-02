"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# Pre-compute grids at module level to avoid re-creation overhead
def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

GRID_COARSE = _make_fib_grid(3000)
GRID_FINE = _make_fib_grid(10000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n, offset=0.0):
        i = np.arange(n) + 0.5 + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_grid(pts, grid=None):
        """Fast covering radius via grid evaluation."""
        if grid is None:
            grid = GRID_COARSE
        dots = grid @ pts.T  # shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # shape (G, N)
        min_dists = np.min(dists, axis=1)  # shape (G,)
        return float(np.max(min_dists))

    def compute_covering_radius_hull(pts):
        """Compute exact covering radius via convex hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            norms = np.linalg.norm(normal, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = normal[valid] / norms[valid, np.newaxis]
                centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                dots = np.sum(cc * centroids, axis=1)
                cc[dots < 0] *= -1
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            return np.inf

    def repulsion_objective(pts, alpha=1.0):
        """
        Objective: minimize max covering radius + alpha * sum of inverse squared distances.
        This encourages spreading points while keeping covering radius small.
        """
        # Compute min distance from each grid point to nearest config point
        dots = GRID_COARSE @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        cover_rad = np.max(min_dists)
        
        # Pairwise distances between config points (for repulsion)
        # Use dot products for efficiency
        pair_dots = np.clip(pts @ pts.T, -1.0, 1.0)
        pair_dots = np.maximum(pair_dots, 0.99)  # avoid log(0) issues
        # Only consider off-diagonal
        np.fill_diagonal(pair_dots, 1.0)
        # Inverse squared angular distance ~ 1/(1-dot)^2 for small angles
        repulsion = np.sum(1.0 / (1.0 - pair_dots) ** 2)
        # Subtract diagonal contribution
        repulsion -= N  # remove self-terms
        
        return cover_rad + alpha * repulsion * 1e-4

    def repulsion_gradient(pts, alpha=1.0):
        """Gradient of repulsion objective."""
        grad = np.zeros_like(pts)
        
        # Covering radius gradient: for each grid point, find nearest config point
        dots = GRID_COARSE @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        cover_rad = np.max(min_dists)
        
        # Find which grid points achieve the max
        max_mask = min_dists > cover_rad - 1e-10
        if np.any(max_mask):
            for i in np.where(max_mask)[0]:
                j = min_indices[i]
                g = GRID_COARSE[i]
                p = pts[j]
                dot = np.dot(g, p)
                one_minus_dot_sq = max(1e-12, 1.0 - dot * dot)
                sqrt_oms = np.sqrt(one_minus_dot_sq)
                # d/dp arccos(g.p) = -(g - p*(g.p)) / sqrt(1-(g.p)^2)
                grad[j] += -(g - dot * p) / sqrt_oms
        
        # Repulsion gradient
        pair_dots = np.clip(pts @ pts.T, -1.0, 1.0)
        pair_dots = np.maximum(pair_dots, 0.99)
        np.fill_diagonal(pair_dots, 1.0)
        
        # d/dp_i sum_j 1/(1-dot_ij)^2 = sum_j 2*(p_i - p_j*dot_ij) / (1-dot_ij)^3
        # But we need to be careful with the constraint that points are on sphere
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dot = pair_dots[i, j]
                one_minus_dot = max(1e-10, 1.0 - dot)
                # Gradient contribution from pair (i,j) to point i
                # d/dp_i [1/(1-p_i.p_j)^2] = 2*(p_i - p_j*dot) / (1-dot)^3
                grad[i] += 2.0 * (pts[i] - pts[j] * dot) / (one_minus_dot ** 3)
            # Project to tangent space
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        grad *= alpha * 1e-4
        return grad

    def local_search_facet(pts, max_iters=200):
        """Local search: find worst facet and move its vertices to reduce covering radius."""
        pts = pts.copy()
        best_rad = compute_covering_radius_grid(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Find worst region using grid
            dots = GRID_COARSE @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)
            
            # Find the grid point with maximum distance
            worst_idx = np.argmax(min_dists)
            worst_dist = min_dists[worst_idx]
            nearest_pt_idx = min_indices[worst_idx]
            worst_point = GRID_COARSE[worst_idx]
            
            # Find the 3 config points closest to the worst point (these form the "facet")
            all_dists = dists[worst_idx]
            top3 = np.argsort(all_dists)[:3]
            
            # Try moving these points toward the worst point
            improved = False
            for step in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
                new_pts = pts.copy()
                for idx in top3:
                    p = new_pts[idx]
                    # Move toward worst_point
                    direction = worst_point - np.dot(p, worst_point) * p
                    direction /= max(np.linalg.norm(direction), 1e-10)
                    new_p = p + step * direction
                    new_p /= np.linalg.norm(new_p)
                    new_pts[idx] = new_p
                
                new_rad = compute_covering_radius_grid(new_pts)
                if new_rad < best_rad - 1e-12:
                    pts = new_pts
                    best_rad = new_rad
                    improved = True
                    break
            
            if not improved:
                # Try individual moves
                for idx in top3:
                    for step in [0.01, 0.005, 0.002, 0.001]:
                        new_pts = pts.copy()
                        p = new_pts[idx]
                        direction = worst_point - np.dot(p, worst_point) * p
                        direction /= max(np.linalg.norm(direction), 1e-10)
                        new_p = p + step * direction
                        new_p /= np.linalg.norm(new_p)
                        new_pts[idx] = new_p
                        
                        new_rad = compute_covering_radius_grid(new_pts)
                        if new_rad < best_rad - 1e-12:
                            pts = new_pts
                            best_rad = new_rad
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try random perturbations to escape local minima
                rng = np.random.RandomState(it)
                for _ in range(3):
                    idx = rng.randint(N)
                    rand_vec = rng.randn(3)
                    tangent = rand_vec - np.dot(rand_vec, pts[idx]) * pts[idx]
                    t_norm = np.linalg.norm(tangent)
                    if t_norm < 1e-10:
                        continue
                    tangent /= t_norm
                    
                    new_pts = pts.copy()
                    new_p = pts[idx] + 0.005 * tangent
                    new_p /= np.linalg.norm(new_p)
                    new_pts[idx] = new_p
                    
                    new_rad = compute_covering_radius_grid(new_pts)
                    if new_rad < best_rad - 1e-12:
                        pts = new_pts
                        best_rad = new_rad
                        break
                else:
                    break
        
        return pts, best_rad

    def polish_with_hull(pts):
        """Final polish using exact hull-based covering radius."""
        pts = pts.copy()
        best_rad = compute_covering_radius_hull(pts)
        
        for it in range(50):
            if time.time() > deadline - 3.0:
                break
            
            try:
                hull = ConvexHull(pts)
                faces = hull.simplices
                a = pts[faces[:, 0]]
                b = pts[faces[:, 1]]
                c = pts[faces[:, 2]]
                
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                norms = np.linalg.norm(normal, axis=1)
                
                radii = np.zeros(len(faces))
                valid = norms > 1e-10
                
                if np.any(valid):
                    cc = normal[valid] / norms[valid, np.newaxis]
                    centroids = (a[valid] + b[valid] + c[valid]) / 3.0
                    dots = np.sum(cc * centroids, axis=1)
                    cc[dots < 0] *= -1
                    cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                    r = np.arccos(cos_r)
                    radii[valid] = r
                
                if np.any(~valid):
                    a_deg = a[~valid]
                    b_deg = b[~valid]
                    c_deg = c[~valid]
                    d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                    d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                    d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                    radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                
                max_idx = np.argmax(radii)
                rad = radii[max_idx]
                face = faces[max_idx]
                i, j, k = face
                
                # Compute circumcenter
                a_pt = pts[i]
                b_pt = pts[j]
                c_pt = pts[k]
                ab = b_pt - a_pt
                ac = c_pt - a_pt
                c_dir = np.cross(ab, ac)
                c_norm = np.linalg.norm(c_dir)
                
                if c_norm < 1e-10:
                    break
                    
                c_dir /= c_norm
                centroids = (a_pt + b_pt + c_pt) / 3.0
                if np.dot(c_dir, centroids) < 0:
                    c_dir *= -1
                
                antipode = -c_dir
                
                improved = False
                for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                    new_pts = pts.copy()
                    for idx in [i, j, k]:
                        p = new_pts[idx]
                        new_p = p + step * antipode
                        new_p /= np.linalg.norm(new_p)
                        new_pts[idx] = new_p
                    
                    new_rad = compute_covering_radius_hull(new_pts)
                    if new_rad < rad - 1e-12:
                        pts = new_pts
                        best_rad = new_rad
                        improved = True
                        break
                
                if not improved:
                    break
                    
            except Exception:
                break
        
        return pts, best_rad

    best_points = None
    best_rad = np.inf

    # Phase 1: Multiple random initializations with L-BFGS-B on repulsion objective
    for trial in range(5):
        if time.time() > deadline - 30.0:
            break
        
        offset = trial * 0.1
        points = fibonacci_spiral(N, offset=offset).astype(np.float64)
        
        def obj_wrapper(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            return repulsion_objective(pts, alpha=1.0)
        
        def grad_wrapper(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            g = repulsion_gradient(pts, alpha=1.0)
            return g.ravel()
        
        try:
            result = minimize(
                obj_wrapper,
                points.ravel(),
                jac=grad_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 100}
            )
            candidate = result.x.reshape(N, 3)
            norms = np.linalg.norm(candidate, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            candidate = candidate / norms
            
            rad = compute_covering_radius_grid(candidate)
            if rad < best_rad:
                best_rad = rad
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
        best_rad = compute_covering_radius_grid(best_points)

    # Phase 2: Local search on grid
    if time.time() < deadline - 20.0:
        refined, rad = local_search_facet(best_points.copy(), max_iters=300)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Phase 3: Second local search pass
    if time.time() < deadline - 10.0:
        refined, rad = local_search_facet(best_points.copy(), max_iters=200)
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Phase 4: Final polish with exact hull computation
    if time.time() < deadline - 5.0:
        refined, rad = polish_with_hull(best_points.copy())
        if rad < best_rad:
            best_rad = rad
            best_points = refined.copy()

    # Final validation
    final_rad = compute_covering_radius_hull(best_points)
    if np.isfinite(final_rad):
        best_rad = final_rad

    return best_points

# EVOLVE-BLOCK-END
