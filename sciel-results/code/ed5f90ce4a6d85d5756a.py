"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _covering_radius_deg(pts):
    """Compute covering radius in degrees. Returns inf if invalid."""
    pts = np.asarray(pts)
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = pts / norms[:, None]
    
    # Check not all in a closed hemisphere
    # If all points are in a hemisphere, there exists a direction d such that d·p >= 0 for all p
    # We can check via: the origin is not in the convex hull of the points
    # Quick check: if max over i of |sum of signs| ... simpler: use linear programming or
    # check if there's a vector with all positive dot products.
    # A faster heuristic: compute the convex hull and check if origin is inside.
    # Actually, for 130 points on sphere, they're almost certainly not in a hemisphere.
    # Let's do a quick check: if the maximum angular distance from any point to its nearest neighbor
    # is > 90 degrees, it might be in a hemisphere. But let's just proceed.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    # For each facet (simplex), find the circumcenter
    # The covering radius is the max over facets of the angular distance from facet circumcenter to its vertices
    # But only if the circumcenter is on the "outside" (the cap is empty)
    # Actually, the covering radius equals the max over all facets of the circumradius of the facet,
    # provided the circumcenter lies on the unit sphere (which it does for points on the sphere)
    # and the cap centered at the circumcenter with that radius doesn't contain any other points.
    
    # For a set of points on the unit sphere, the covering radius is the maximum, over all facets of the convex hull,
    # of the circumradius of that facet, where the circumcenter is the point on the sphere equidistant to all facet vertices.
    # The circumradius in angular distance is arccos(circumcenter · vertex) for any vertex.
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]  # shape (k, 3), k = 3 or 4
        k = len(simplex)
        
        # Find circumcenter on the unit sphere equidistant to all vertices
        # For a simplex of k points on the unit sphere, the circumcenter is the normalized
        # solution to: c · v_i = r for all i, |c| = 1
        # This means c is in the span of the vertices and c·v_i is constant.
        # If k=3: c is proportional to the cross product of (v2-v1) x (v3-v1) normalized... 
        # Actually for 3 points on unit sphere, circumcenter = normalize(v1 + v2 + v3) only if they form an equilateral triangle.
        # General approach: solve for c = sum a_i v_i such that c·v_j = constant for all j, |c|=1.
        
        # For a simplex on the unit sphere, the circumcenter c satisfies c·v_i = cos(theta) for all i.
        # c is in the affine hull of the vertices. Write c = M @ a where M has rows v_i^T.
        # We need M @ a · v_j = cos(theta) for all j, i.e., (M @ M^T) @ a = cos(theta) * ones
        # and |M @ a| = 1.
        
        G = v @ v.T  # Gram matrix, shape (k, k)
        # Solve G @ a = cos(theta) * ones
        # Then c = v.T @ a, and |c|^2 = a^T G a = cos(theta)^2 * ones^T G^{-1} ones
        # So cos(theta) = 1 / sqrt(ones^T G^{-1} ones)
        
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # Degenerate facet, skip
            continue
        
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 0:
            continue
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            # All points coincident
            continue
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        rad_deg = np.degrees(theta)
        if rad_deg > max_rad:
            max_rad = rad_deg
    
    return max_rad


def _smooth_covering_radius(pts, sigma=0.05):
    """Smooth approximation of the covering radius for gradient-based optimization."""
    pts = np.asarray(pts)
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None]
    
    # Compute pairwise angular distances
    # d_ij = arccos(clip(pts_i · pts_j, -1, 1))
    dot = pts @ pts.T
    dot = np.clip(dot, -1.0, 1.0)
    dist = np.arccos(dot)  # angular distance matrix
    
    # For each point, find the nearest neighbor distance (excluding self)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    
    # The covering radius is approximately half the max nearest-neighbor distance
    # But this is a lower bound. A better smooth approximation:
    # Use a soft-min over all points of the distance to the nearest configuration point.
    # We sample points on the sphere and compute max distance.
    
    # For efficiency, use the fact that the covering radius is related to the
    # maximum of the Voronoi cell circumradii. A smooth proxy:
    # f = max_i (nearest[i]) / 2  -- this is a lower bound on covering radius
    # Better: use the maximum over a set of test points.
    
    # Use a Fibonacci spiral of test points
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Distance from each test point to each configuration point
    dot_t = test_pts @ pts.T
    dot_t = np.clip(dot_t, -1.0, 1.0)
    dist_t = np.arccos(dot_t)  # (M, N)
    
    # For each test point, distance to nearest configuration point
    min_dist = np.min(dist_t, axis=1)
    
    # Smooth max: use log-sum-exp
    # f = (1/sigma) * log(sum(exp(sigma * min_dist)))
    scaled = sigma * min_dist
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp / sigma
    
    return smooth_max


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-10] = 1.0
    return pts / norms[:, None]


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(best_pts)
    
    # L-BFGS optimization on smooth covering radius
    # We optimize in Cartesian coordinates with normalization
    def objective(x):
        if time.time() > deadline:
            return 1e6
        p = _normalize(x.reshape(N, 3))
        return _smooth_covering_radius(p, sigma=0.03)
    
    try:
        result = minimize(
            objective,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50}
        )
        if time.time() < deadline:
            opt_pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius_deg(opt_pts)
            if cr < best_cr:
                best_pts = opt_pts
                best_cr = cr
    except Exception:
        pass
    
    # Greedy local polishing: iteratively move points to reduce covering radius
    # For each point, try to move it to reduce the max circumradius of its Voronoi cell
    t_polish_start = time.time()
    polish_deadline = min(deadline, t_polish_start + 60.0)
    
    for iteration in range(500):
        if time.time() > polish_deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > polish_deadline:
                break
            
            # Try small perturbations of point idx
            best_local = best_pts[idx].copy()
            best_local_cr = best_cr
            
            # Random directions
            for _ in range(20):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                for step in [0.01, 0.005, 0.001]:
                    candidate = best_pts.copy()
                    candidate[idx] = best_pts[idx] + step * direction
                    candidate[idx] /= np.linalg.norm(candidate[idx])
                    
                    cr = _covering_radius_deg(candidate)
                    if cr < best_local_cr:
                        best_local = candidate[idx]
                        best_local_cr = cr
            
            if best_local_cr < best_cr:
                best_pts[idx] = best_local
                best_cr = best_local_cr
                improved = True
        
        if not improved:
            break
    
    # Final normalization
    best_pts = _normalize(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
