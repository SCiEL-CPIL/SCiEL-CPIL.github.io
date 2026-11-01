"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius by finding the largest circumradius of facets
    of the convex hull. We approximate by checking all pairs of nearest neighbors
    and using the Voronoi cell approach.
    
    For efficiency, we compute the angular distance from each point to its
    nearest neighbor and use that as a lower bound, then refine.
    """
    pts = _normalize(pts)
    # Compute all pairwise angular distances
    # dot[i,j] = cos(angle)
    dot = pts @ pts.T
    np.clip(dot, -1.0, 1.0, out=dot)
    
    # For each point, find the maximum distance to the "farthest" point in its Voronoi cell
    # Approximation: the covering radius is related to the maximum over all points
    # of the angular radius of the largest empty cap centered at that point.
    
    # Better approach: the covering radius equals the maximum circumradius of
    # the convex hull facets. For N=50, we can use scipy's ConvexHull.
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the 3 (or 2) points of the facet
            facet_pts = pts[simplex]
            # Compute circumcenter of the facet
            if len(simplex) == 3:
                # Triangle circumcenter
                a, b, c = facet_pts
                # Using formula for circumcenter of triangle in 3D
                # Project to 2D: find orthonormal basis for the plane
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                norm_n = np.linalg.norm(normal)
                if norm_n < 1e-12:
                    continue
                normal /= norm_n
                # Find circumcenter
                # |x - a|^2 = |x - b|^2 => 2*x.(b-a) = |b|^2 - |a|^2 + 2*a.(b-a)
                # Since all on unit sphere, |a|=|b|=|c|=1
                # 2*x.(b-a) = 0 => x.(b-a) = 0
                # 2*x.(c-a) = 0 => x.(c-a) = 0
                # So x is in the plane and perpendicular to both (b-a) and (c-a) from a
                # Actually, circumcenter satisfies:
                # (x-a).(b-a) = |b-a|^2/2
                # (x-a).(c-a) = |c-a|^2/2
                # Let x = a + s*(b-a) + t*(c-a)
                # (a + s*ab + t*ac - a).(b-a) = |ab|^2/2
                # s*|ab|^2 + t*(ac.ab) = |ab|^2/2
                # Similarly: s*(ab.ac) + t*|ac|^2 = |ac|^2/2
                ab_dot_ab = np.dot(ab, ab)
                ac_dot_ac = np.dot(ac, ac)
                ab_dot_ac = np.dot(ab, ac)
                det = ab_dot_ab * ac_dot_ac - ab_dot_ac * ab_dot_ac
                if abs(det) < 1e-12:
                    continue
                s = (ab_dot_ab * ac_dot_ac - ab_dot_ac * ac_dot_ac) / (2 * det)
                t = (ab_dot_ab * ab_dot_ac - ac_dot_ac * ab_dot_ac) / (2 * det)
                # Wait, let me redo this properly
                # System:
                # s*ab_dot_ab + t*ab_dot_ac = ab_dot_ab / 2
                # s*ab_dot_ac + t*ac_dot_ac = ac_dot_ac / 2
                s = (ab_dot_ab * ac_dot_ac - ab_dot_ac * ac_dot_ac) / (2 * det)
                t = (ac_dot_ac * ab_dot_ab - ab_dot_ac * ab_dot_ab) / (2 * det)
                # Hmm, let me just use Cramer's rule
                # [ab_dot_ab  ab_dot_ac] [s]   [ab_dot_ab/2]
                # [ab_dot_ac  ac_dot_ac] [t] = [ac_dot_ac/2]
                s = (ab_dot_ab/2 * ac_dot_ac - ab_dot_ac * ac_dot_ac/2) / det
                t = (ab_dot_ab * ac_dot_ac/2 - ab_dot_ac/2 * ac_dot_ac) / det
                x = a + s * ab + t * ac
                # The circumradius is |x| (distance from origin to circumcenter)
                # Actually, the angular radius is arccos(|x|) since points are on unit sphere
                # Wait: the circumcenter of the spherical triangle... 
                # The cap has angular radius = arccos(|x|) where x is the circumcenter
                # No: if x is the circumcenter in 3D, then |x - a| = |x - b| = |x - c| = R
                # And |a| = 1. The angular distance from x/|x| to a is arccos(x.a / |x|)
                # But x.a = x.a, and |x-a|^2 = |x|^2 + 1 - 2*x.a = R^2
                # The angular radius of the cap is arccos(cos(theta)) where cos(theta) = x.a/|x|
                # Actually simpler: the covering radius for this facet is the angular distance
                # from the circumcenter to any vertex, which is arccos(x · a / |x|)
                # But x · a = (a + s*ab + t*ac) · a = 1 + s*ab.a + t*ac.a
                # ab.a = (b-a).a = b.a - 1
                # ac.a = (c-a).a = c.a - 1
                # This is getting complicated. Let me use a different approach.
                
                # The angular circumradius of a spherical triangle with vertices on unit sphere:
                # The circumcenter in 3D is x. The angular radius is arccos(|x|) if x is 
                # the point on the sphere... no.
                
                # Let me just compute: the cap centered at x/|x| with angular radius 
                # arccos(x·a/|x|) covers a. The covering radius contribution is this angle.
                if np.linalg.norm(x) < 1e-12:
                    continue
                x_unit = x / np.linalg.norm(x)
                cos_angle = np.dot(x_unit, a)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                max_r = max(max_r, angle)
            elif len(simplex) == 2:
                # Edge case
                a, b = facet_pts
                mid = (a + b) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm < 1e-12:
                    continue
                mid_unit = mid / mid_norm
                cos_angle = np.dot(mid_unit, a)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                max_r = max(max_r, angle)
        
        return np.degrees(max_r)
    except Exception:
        # Fallback: use nearest neighbor distance
        min_dists = np.min(1.0 - dot, axis=1)  # 1 - cos(theta) ~ theta^2/2 for small theta
        # Better: compute actual angular distances
        angles = np.arccos(np.clip(1.0 - np.min(1.0 - dot, axis=1), -1.0, 1.0))
        return np.degrees(np.max(angles))

def _soft_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    For each point on a fine grid, compute min distance to any config point,
    then use log-sum-exp to approximate the max.
    """
    pts = _normalize(pts)
    # Sample points on the sphere
    n_samples = 2000
    # Fibonacci sphere for sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    sample_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each sample point to each config point
    # cos(angle) = sample @ config.T
    cos_angles = sample_pts @ pts.T  # shape (n_samples, N)
    np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
    
    # Angular distances
    angles = np.arccos(cos_angles)  # shape (n_samples, N)
    
    # For each sample, min angle to any config point
    min_angles = np.min(angles, axis=1)  # shape (n_samples,)
    
    # Soft max using log-sum-exp
    # log(sum(exp(beta * min_angles))) / beta
    max_val = np.max(min_angles)
    if max_val * beta > 700:
        # Overflow protection
        shifted = min_angles - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        log_sum_exp = np.log(np.sum(np.exp(beta * min_angles))) / beta
    
    return log_sum_exp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Generate initial configuration using golden angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Step 2: Repulsion relaxation for better initial spread
    def repulsion_energy(pts):
        """Sum of inverse distances (or log distances) to encourage spread."""
        pts = _normalize(pts)
        dot = pts @ pts.T
        np.fill_diagonal(dot, 2.0)  # Avoid division by zero
        # Use log(1 - dot) as a repulsion potential (large when close)
        # Or use 1/(1-dot) 
        dists_sq = 2.0 - dot
        dists_sq = np.maximum(dists_sq, 1e-10)
        return -np.sum(np.log(dists_sq))
    
    # Do a few iterations of gradient descent on repulsion
    for it in range(50):
        if time.time() > deadline:
            break
        pts = _normalize(pts)
        dot = pts @ pts.T
        np.fill_diagonal(dot, 2.0)
        dists_sq = 2.0 - dot
        dists_sq = np.maximum(dists_sq, 1e-10)
        # Gradient of -sum(log(dists_sq)) w.r.t. pts
        # d/dx_i log(2 - x_i.x_j) = -2*(x_j - x_i)/(2 - x_i.x_j) ... 
        # Let's just use numerical gradient or a simple update
        # Force on point i: sum_j (x_i - x_j) / |x_i - x_j|^2 * something
        # For log potential: force = sum_j (x_i - x_j) / (2 - x_i.x_j)
        forces = np.zeros_like(pts)
        for i_idx in range(N):
            for j_idx in range(N):
                if i_idx == j_idx:
                    continue
                d = 2.0 - dot[i_idx, j_idx]
                forces[i_idx] += (pts[i_idx] - pts[j_idx]) / max(d, 1e-10)
        
        lr = 0.01
        pts = pts + lr * forces
        pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Step 3: L-BFGS optimization of soft covering radius
    def objective(x):
        if time.time() > deadline:
            return 0
        pts = _normalize(x.reshape(-1, 3))
        return _soft_covering_radius(pts, beta=50.0)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < _soft_covering_radius(_normalize(result.x.reshape(-1,3)), beta=50.0):
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_pts = new_pts
                best_radius = new_radius
    except Exception:
        pass
    
    # Step 4: Lloyd's algorithm for local refinement
    if time.time() < deadline - 10:
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            pts = _normalize(best_pts)
            
            # Assign each point on a fine grid to nearest config point
            n_samples = 3000
            i = np.arange(n_samples) + 0.5
            z = 1.0 - 2.0 * i / n_samples
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            sample_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
            # Find nearest config point for each sample
            cos_angles = sample_pts @ pts.T
            np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
            nearest = np.argmax(cos_angles, axis=1)
            
            # Update each config point to be the centroid of its Voronoi cell
            new_pts = np.zeros_like(pts)
            for k in range(N):
                mask = nearest == k
                if np.sum(mask) > 0:
                    new_pts[k] = np.mean(sample_pts[mask], axis=0)
                else:
                    new_pts[k] = pts[k]
            
            # Normalize
            new_pts = _normalize(new_pts)
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_pts = new_pts
                best_radius = new_radius
            else:
                # No improvement, break
                break
    
    # Step 5: Final polish with tighter L-BFGS
    if time.time() < deadline - 5:
        def objective_tight(x):
            if time.time() > deadline:
                return 0
            pts = _normalize(x.reshape(-1, 3))
            return _soft_covering_radius(pts, beta=100.0)
        
        x0 = best_pts.flatten()
        try:
            result = minimize(objective_tight, x0, method='L-BFGS-B',
                             options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_pts = new_pts
                best_radius = new_radius
        except Exception:
            pass
    
    # Ensure final result is valid
    best_pts = _normalize(best_pts)
    
    # Final check
    if time.time() < deadline:
        final_radius = _covering_radius(best_pts)
        if final_radius < best_radius:
            best_radius = final_radius
    
    return best_pts

# EVOLVE-BLOCK-END
