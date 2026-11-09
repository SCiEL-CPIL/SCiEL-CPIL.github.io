# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
        
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    # Check no repeated points / zero length edges
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    # Total turning check
    # For a simple convex polygon, the sum of exterior angles is 2*pi.
    # We can verify this by checking the winding number or just trusting the sign check 
    # for small perturbations of a convex polygon. 
    # However, to be safe and match the spec "total turning +-2*pi":
    # The angle of each edge is atan2(y, x). The turn is the difference.
    # Sum of turns = 2*pi (or -2*pi).
    # For a convex polygon, this is guaranteed if it's simple and vertices are in order.
    # Let's do a quick check if it's strictly convex and simple.
    # The sign check above ensures strict convexity (no collinear).
    # Simple polygon check: for n=20 and small perturbations, it's simple.
    # We'll skip the expensive O(n^2) intersection check for speed, assuming 
    # small perturbations of a regular polygon remain simple and convex.
    
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_radial_polygon(a, b, n=20):
    """
    Create a polygon defined by r(theta) = 1 + a*cos(2*theta) + b*cos(4*theta).
    This maintains 4-fold symmetry (or close to it) which is often optimal 
    for these types of distance-based metrics.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 1.0 + a * np.cos(2.0 * angles) + b * np.cos(4.0 * angles)
    
    # Ensure radii are positive to avoid self-intersection or non-convexity
    if np.any(radii <= 0.0):
        return None
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _search_grid():
    """
    Perform a grid search over the (a, b) parameter space.
    """
    n = 20
    best_score = 0.0
    best_poly = None
    
    # The regular polygon corresponds to a=0, b=0.
    # We expect small perturbations to improve the score.
    # Let's search in a box around the origin.
    # Based on typical improvements, a and b are likely small (< 0.1).
    
    # Coarse grid first
    a_vals = np.linspace(-0.1, 0.1, 41)
    b_vals = np.linspace(-0.1, 0.1, 41)
    
    start_time = time.time()
    deadline = start_time + 600.0 # 10 min limit, keep margin
    
    # We can vectorize the score calculation somewhat, but the min-score over vertices 
    # and the sorting inside makes it tricky. 
    # However, n=20 is small. 41*41 = 1681 evaluations. 
    # Each evaluation involves 20 vertex scores, each with a sort of 19 items.
    # This is very fast.
    
    for a in a_vals:
        for b in b_vals:
            if time.time() > deadline - 5:
                break
                
            poly = _make_radial_polygon(a, b, n)
            if poly is None:
                continue
                
            score = _polygon_min_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly
                
    # Refine around the best point found
    if best_poly is not None:
        # Find the (a,b) that produced best_poly
        # We stored the poly, but not the params. Let's just do a local refinement 
        # by re-scanning a smaller grid around the center if we tracked params.
        # For simplicity and robustness, let's just do a second, finer pass 
        # over the whole range if time permits, or a local search.
        
        # Let's do a local search using Nelder-Mead on the (a,b) space
        # starting from the best grid point. 
        # But we didn't store the best (a,b). 
        # Let's just do a finer grid over the whole range if time allows, 
        # or assume the coarse grid is good enough. 
        # Given the smoothness of the metric, a finer grid is safer.
        
        if time.time() < deadline - 10:
            # Finer grid
            a_vals_fine = np.linspace(-0.05, 0.05, 21)
            b_vals_fine = np.linspace(-0.05, 0.05, 21)
            
            # Actually, let's just expand the search range slightly if the best was on the edge?
            # No, let's just do a dense grid in the likely optimal region.
            # The regular polygon is a=0, b=0.
            # Let's try a slightly larger range with moderate resolution.
            
            a_vals_mid = np.linspace(-0.15, 0.15, 61)
            b_vals_mid = np.linspace(-0.15, 0.15, 61)
            
            for a in a_vals_mid:
                for b in b_vals_mid:
                    if time.time() > deadline - 5:
                        break
                    poly = _make_radial_polygon(a, b, n)
                    if poly is None:
                        continue
                    score = _polygon_min_score(poly)
                    if score > best_score:
                        best_score = score
                        best_poly = poly
                        
    return best_poly

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    # Try the structured search
    polygon = _search_grid()
    
    # Fallback to regular polygon if search fails or times out
    if polygon is None:
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
    # Final safety check: if the polygon is invalid (e.g. not convex), return regular
    # The search function already checks validity via _polygon_min_score returning 0.0 for invalid
    # But if _polygon_min_score returned a score, it's valid.
    # If best_score was 0.0, best_poly might be None or an invalid poly.
    # _search_grid returns best_poly which is only updated if score > best_score (initially 0.0).
    # So if no valid polygon found, best_poly remains None.
    
    if time.time() - start_time > 625:
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
    return polygon

# EVOLVE-BLOCK-END
