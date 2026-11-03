"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull Delaunay triangulation."""
    pts = np.asarray(pts, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    # Convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    # The hull simplices (triangles) give the Delaunay triangulation on the sphere
    # For each triangle, compute the circumradius (angular distance from circumcenter to vertices)
    # The circumcenter of a spherical triangle is in the direction of the normalized sum of vertices
    # (for acute triangles). The covering radius is the max over all triangles of the max angular distance
    # from the triangle's circumcenter to its vertices.
    simplices = hull.simplices
    max_cr = 0.0
    for tri in simplices:
        v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Circumcenter direction: normalized (v0+v1+v2) works for acute triangles
        # More robust: solve for point equidistant from all three
        # For now use the sum direction
        c = v0 + v1 + v2
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            # Degenerate, use midpoint
            c = (v0 + v1) / 2.0
            cn = np.linalg.norm(c)
            if cn < 1e-10:
                continue
        c = c / cn
        # Angular distances from c to each vertex
        d0 = np.arccos(np.clip(np.dot(c, v0), -1.0, 1.0))
        d1 = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c, v2), -1.0, 1.0))
        cr = max(d0, d1, d2)
        if cr > max_cr:
            max_cr = cr
    return np.degrees(max_cr)

def _smooth_objective(x, pts_init=None):
    """Smooth objective: mean of top-k angular distances for gradient-based opt."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    # Compute all pairwise angular distances
    # Use dot product for speed
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)  # (N, N)
    # For each point, the max distance to any other point is an upper bound
    # But for covering radius, we need the Voronoi cell circumradius
    # Use a simpler smooth proxy: for each point, the max angular distance to its neighbors
    # Actually, let's use: for each point i, compute the max angle to all other points
    # This is not quite the covering radius but is a smooth upper bound
    max_angles = np.max(angles, axis=1)
    # Use mean of top-k for smoothness
    k = min(20, N)
    topk = np.sort(max_angles)[::-1][:k]
    return np.mean(topk)

def _grad_smooth_objective(x, pts_init=None):
    """Gradient of smooth objective."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    max_angles = np.max(angles, axis=1)
    k = min(20, N)
    topk_idx = np.argsort(max_angles)[::-1][:k]
    # Gradient is complex; use numerical or just return None and let L-BFGS use finite differences
    return None

def _fibonacci_init(seed=0):
    """Golden-angle spiral initialization."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _optimize_from_start(pts0, maxiter=200, deadline=130.0):
    """Run L-BFGS-B optimization from a starting point."""
    x0 = pts0.flatten()
    
    def objective(x):
        return _smooth_objective(x)
    
    result = minimize(
        objective, x0, method='L-BFGS-B',
        maxiter=maxiter,
        options={'maxfun': 500, 'ftol': 1e-10}
    )
    return result.x.reshape(-1, 3)

def construct():
    global _start_time
    best_pts = None
    best_cr = 180.0
    
    # Try multiple initializations
    inits = []
    for seed in range(8):
        inits.append(_fibonacci_init(seed))
    for seed in range(8):
        inits.append(_random_init(seed))
    
    # Evaluate initial points
    for pts in inits:
        if time.time() > _DEADLINE - 30:
            break
        cr = _covering_radius_deg(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Optimize from best initializations
    # Sort inits by their initial covering radius
    init_scores = []
    for pts in inits:
        if time.time() > _DEADLINE - 40:
            break
        cr = _covering_radius_deg(pts)
        init_scores.append((cr, pts))
    init_scores.sort(key=lambda x: x[0])
    
    for cr, pts in init_scores[:6]:
        if time.time() > _DEADLINE - 20:
            break
        try:
            opt_pts = _optimize_from_start(pts, maxiter=100, deadline=_DEADLINE)
            # Normalize
            opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
            cr_opt = _covering_radius_deg(opt_pts)
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Final local refinement: gradient descent on the true covering radius using subgradient
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts = best_pts.copy()
        for it in range(50):
            if time.time() > _DEADLINE - 2:
                break
            # Compute subgradient of covering radius
            hull = ConvexHull(pts)
            simplices = hull.simplices
            max_cr = 0.0
            worst_tri = None
            for tri in simplices:
                v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                c = v0 + v1 + v2
                cn = np.linalg.norm(c)
                if cn < 1e-10:
                    c = (v0 + v1) / 2.0
                    cn = np.linalg.norm(c)
                    if cn < 1e-10:
                        continue
                c = c / cn
                d0 = np.arccos(np.clip(np.dot(c, v0), -1.0, 1.0))
                d1 = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c, v2), -1.0, 1.0))
                cr = max(d0, d1, d2)
                if cr > max_cr:
                    max_cr = cr
                    worst_tri = (tri, c)
            if worst_tri is None:
                break
            tri, c = worst_tri
            # Move the worst triangle's vertices away from circumcenter
            step = 0.001
            for idx in tri:
                # Move point away from circumcenter
                pts[idx] -= step * c
            # Renormalize
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            cr_new = _covering_radius_deg(pts)
            if cr_new < best_cr:
                best_cr = cr_new
                best_pts = pts.copy()
    
    # Final normalization
    if best_pts is not None:
        best_pts = np.asarray(best_pts, dtype=np.float64)
        best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
        # Ensure no coincident points
        # Check for near-coincident points and perturb
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-6:
                    best_pts[j] += 1e-5 * np.random.randn(3)
                    best_pts[j] /= np.linalg.norm(best_pts[j])
        return best_pts
    
    # Fallback
    return _fibonacci_init(0)
# EVOLVE-BLOCK-END
