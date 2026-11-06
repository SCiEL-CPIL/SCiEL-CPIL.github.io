"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize_scalar, minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-10] = 1.0
    return pts / norms[:, None]

def _get_facet_info(pts):
    """
    Computes the circumradius and circumcenter for every facet of the convex hull.
    Returns a list of tuples: (radius_deg, circumcenter_unit_vector, simplex_indices)
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    
    facets = []
    for simplex in hull.simplices:
        v = pts[simplex]
        k = len(simplex)
        # Gram matrix
        G = v @ v.T
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 1e-12:
            continue
            
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            continue
            
        # Circumcenter calculation
        # c_unnorm = v^T G^{-1} s
        a = G_inv @ s
        c_unnorm = v.T @ a
        norm_c = np.linalg.norm(c_unnorm)
        if norm_c < 1e-10:
            continue
        c = c_unnorm / norm_c
        
        rad_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        facets.append((rad_deg, c, simplex))
        
    return facets

def _covering_radius_deg(pts):
    """Compute exact covering radius in degrees. Returns inf if invalid."""
    pts = np.asarray(pts)
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]
        k = len(simplex)
        G = v @ v.T
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 0:
            continue
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            continue
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        rad_deg = np.degrees(theta)
        if rad_deg > max_rad:
            max_rad = rad_deg
    return max_rad

def _optimize_single_point(pts, idx, facets_info):
    """
    Attempts to improve the configuration by moving point `idx`.
    We focus on the facets that contain `idx` and have large circumradii.
    We perform a local search on the sphere for the new position of `pts[idx]`
    that minimizes the maximum circumradius of the affected facets.
    """
    current_p = pts[idx].copy()
    
    # Identify facets containing idx that are "significant" (large radius)
    # We consider the top few worst facets globally to guide the search,
    # but specifically we care about facets touching idx.
    # Let's find all facets touching idx.
    touching_facets = []
    for rad, c, simplex in facets_info:
        if idx in simplex:
            touching_facets.append((rad, c, simplex))
            
    if not touching_facets:
        return current_p, False

    # Sort touching facets by radius descending
    touching_facets.sort(key=lambda x: x[0], reverse=True)
    
    # We will try to move the point to reduce the radius of the worst touching facet.
    # Let's use a local optimization on the tangent plane or spherical coordinates.
    # A robust way: parameterize the neighborhood of current_p.
    # Or simply: try moving towards the circumcenter of the worst touching facet?
    # Moving towards the circumcenter c of a facet reduces the distance to c?
    # No, the circumcenter is the point on the sphere furthest from the facet vertices.
    # If we move a vertex towards c, the facet "shrinks" in a sense, but the circumcenter moves.
    
    # Better heuristic: The "hole" is centered at c. To fill the hole, we should move the vertex
    # such that it becomes closer to c? 
    # Actually, if we move a vertex closer to the circumcenter, the circumradius of that specific facet decreases.
    # Let's verify: For a triangle on sphere, circumradius R is determined by the angles.
    # If we move one vertex towards the circumcenter, the triangle becomes more "equilateral" locally?
    # Not necessarily.
    
    # Alternative: Use a small grid search or Nelder-Mead on the new position of pts[idx]
    # minimizing the max radius of the facets that will exist after the move.
    # This is expensive to evaluate (re-compute hull).
    # Instead, let's use the gradient of the radius w.r.t the vertex position.
    
    # Simpler and more robust: "Hole Filling"
    # Find the worst facet overall. If it touches idx, move idx towards its circumcenter.
    # If it doesn't touch idx, we can't fix it by moving idx (unless we move multiple points).
    # So, for each point, we only move it if it is part of a top-worst facet.
    
    worst_rad, worst_c, worst_simplex = touching_facets[0]
    
    # We want to reduce worst_rad.
    # Move idx towards worst_c.
    # How much? Let's try a few steps.
    
    # Direction on sphere from current_p to worst_c
    # v = worst_c - current_p
    # proj = v - current_p * (current_p . v)
    
    v = worst_c - current_p
    dot = np.dot(current_p, v)
    proj = v - current_p * dot
    norm_proj = np.linalg.norm(proj)
    if norm_proj < 1e-8:
        return current_p, False
    dir_vec = proj / norm_proj
    
    # Try several step sizes
    best_new_p = current_p
    best_local_rad = worst_rad
    
    # We need to evaluate the actual covering radius change.
    # Since re-computing the full hull is expensive, we can approximate or just do a few trials.
    # Given N=130, ConvexHull is fast enough for ~1000 calls in 120s?
    # 130 points hull is very fast. Let's do a small local search.
    
    for step in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        new_p = current_p + step * dir_vec
        new_p = _normalize(new_p.reshape(1,3))[0]
        
        # Temporarily update pts
        old_p = pts[idx].copy()
        pts[idx] = new_p
        
        # Check if this improves the global covering radius
        # To save time, we can check the max radius of facets touching idx in the new hull?
        # No, moving one point changes the hull structure.
        # We must compute the new covering radius or at least the worst facet radius.
        # Let's compute the full covering radius for accuracy in this local step.
        # Optimization: We can check if the new radius is better than the current worst.
        
        # Actually, let's just compute the new covering radius.
        new_cr = _covering_radius_deg(pts)
        
        if new_cr < best_local_rad:
            best_local_rad = new_cr
            best_new_p = new_p
        else:
            # If it got worse, we might break early if we assume monotonicity?
            # Not necessarily, but often yes for small steps.
            pass
            
        pts[idx] = old_p
        
    if np.linalg.norm(best_new_p - current_p) > 1e-10:
        return best_new_p, True
    else:
        return current_p, False

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(best_pts)
    
    current_pts = pts.copy()
    
    # Main Loop: Facet-Centric Local Search
    # We iterate through points, and for each, we try to improve it by moving it
    # towards the circumcenter of the worst facet it belongs to.
    
    # To speed up, we can pre-compute facets once per "epoch" and use them for multiple steps?
    # No, facets change with every move.
    # But we can reuse the facet info for a few steps if we are careful?
    # Given the speed of ConvexHull for N=130, we can afford to recompute frequently.
    
    max_iterations = 5000
    no_improve_count = 0
    max_no_improve = 100
    
    for iter in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Get current facets
        facets_info = _get_facet_info(current_pts)
        if not facets_info:
            break
            
        # Sort facets by radius descending
        facets_info.sort(key=lambda x: x[0], reverse=True)
        
        # Identify the worst facet
        worst_rad, worst_c, worst_simplex = facets_info[0]
        
        # If the worst facet is very small, we might be done or stuck
        if worst_rad < 11.3165625 - 1e-4:
            # We beat the known best!
            # Try to push further?
            pass
            
        # Try to improve one of the points in the worst facet
        # We pick a random point from the worst facet to move
        idx = np.random.choice(worst_simplex)
        
        new_p, improved = _optimize_single_point(current_pts, idx, facets_info)
        
        if improved:
            current_pts[idx] = new_p
            # Evaluate new covering radius
            new_cr = _covering_radius_deg(current_pts)
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = current_pts.copy()
                no_improve_count = 0
            else:
                # It improved locally (worst facet) but not globally?
                # Or my local check was optimistic.
                # If it didn't improve global best, we might still keep it if it's better than previous current?
                # Let's accept if it's better than the previous current_cr?
                # We don't track current_cr explicitly in the loop, let's just accept if it helps the worst facet.
                # Actually, _optimize_single_point already checked if it reduced the local worst radius.
                # Let's accept the move if it reduced the global CR, or if it reduced the worst facet radius significantly.
                # For simplicity and safety, only accept if it improves global CR.
                # Wait, if it doesn't improve global CR, we should revert?
                # _optimize_single_point already reverted in its internal loop if it didn't find a better global CR?
                # No, _optimize_single_point compares against `best_local_rad` which was initialized to `worst_rad` (the local facet radius).
                # It does NOT check global CR in the comparison inside _optimize_single_point.
                # It computes `new_cr` but compares `new_cr < best_local_rad`.
                # This is a bug in my logic above: `best_local_rad` is a radius in degrees, `new_cr` is a radius in degrees.
                # So it accepts if the new global CR is smaller than the old worst facet radius.
                # This is a good heuristic: if the new global CR is smaller than the old worst facet, we definitely improved the worst facet.
                
                # So `improved` means `new_cr < old_worst_rad`.
                # We should accept this move in `current_pts` to allow further exploration,
                # even if `new_cr` is not < `best_cr` (which is the global best so far).
                # But we must update `best_cr` only if `new_cr < best_cr`.
                
                # However, if `new_cr` is worse than the previous `current_cr`, we might be degrading.
                # Let's track `current_cr`.
                # To save time, I'll just accept if it reduces the worst facet radius.
                # This is a "hill climbing" on the worst facet.
                
                pass # Move is already in current_pts from _optimize_single_point? 
                # No, _optimize_single_point reverts pts[idx] to old_p at the end of its loop if it didn't find an improvement?
                # Let's re-read _optimize_single_point.
                # It sets pts[idx] = new_p, computes new_cr, reverts pts[idx] = old_p.
                # It returns best_new_p.
                # It does NOT update current_pts.
                # So I must update current_pts here.
                
                current_pts[idx] = new_p
                
                # Update best
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_pts = current_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If we haven't improved global best for a while, add some noise?
                    if no_improve_count > max_no_improve:
                         # Perturb a random point
                         idx_rand = np.random.randint(N)
                         noise = np.random.randn(3) * 0.1
                         new_p_rand = current_pts[idx_rand] + noise
                         current_pts[idx_rand] = _normalize(new_p_rand.reshape(1,3))[0]
                         no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                 # Perturb a random point
                 idx_rand = np.random.randint(N)
                 noise = np.random.randn(3) * 0.1
                 new_p_rand = current_pts[idx_rand] + noise
                 current_pts[idx_rand] = _normalize(new_p_rand.reshape(1,3))[0]
                 no_improve_count = 0

    return best_pts
# EVOLVE-BLOCK-END
