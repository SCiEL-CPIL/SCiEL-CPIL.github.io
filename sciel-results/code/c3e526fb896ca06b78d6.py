"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_facet_circumradii(points, hull):
    """
    Compute the circumradius (angular distance to circumcenter) for each facet.
    Returns an array of circumradii (in radians) for each simplex in hull.simplices.
    """
    circumradii = np.zeros(len(hull.simplices))
    for i, simplex in enumerate(hull.simplices):
        p1, p2, p3 = points[simplex]
        
        # Calculate the circumcenter of the triangle formed by p1, p2, p3 on the sphere.
        # The circumcenter is the intersection of the sphere with the plane passing through p1, p2, p3.
        # The normal to the plane is n = (p2 - p1) x (p3 - p1).
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            circumradii[i] = 0.0
            continue
        n = n / n_norm
        
        # The circumcenter c is on the line t*n.
        # Since p1 is on the sphere and the plane, c = t*n.
        # |c|^2 = 1 => t^2 = 1 => t = +/- 1.
        # We need the center such that it is on the same side of the triangle as the origin?
        # Actually, the circumcenter of a spherical triangle is the point equidistant from p1, p2, p3.
        # The vector c is parallel to n.
        # Let's find t such that c = t*n is the circumcenter.
        # The distance from c to p1 is the circumradius R.
        # cos(R) = c . p1 = t * (n . p1).
        # Also |c - p1|^2 = 1 + 1 - 2 c.p1 = 2 - 2t(n.p1) = 2(1 - cos R).
        # This is consistent.
        # We need to determine the sign of t.
        # The circumcenter should be "inside" the spherical triangle.
        # A robust way: project the origin onto the plane? No.
        # Let's check the dot product of n with a point in the triangle, e.g., centroid.
        # If n . centroid > 0, then the normal points towards the "interior" of the sphere relative to the plane?
        # Actually, for a convex hull containing the origin, the normal n calculated as (p2-p1)x(p3-p1) might point inwards or outwards.
        # We want the circumcenter to be on the sphere.
        # Let's try t = 1 and t = -1.
        # The circumradius is arccos(|c . p1|).
        # Since c is a unit vector, c . p1 = cos(angle).
        # We want the angle to be the circumradius.
        # The circumcenter is the unique point on the sphere equidistant from p1, p2, p3.
        # There are two such points (antipodal). We want the one that is the "center" of the cap.
        # For a small triangle, the circumcenter is close to the centroid.
        # Centroid g = (p1+p2+p3)/3.
        # If n . g > 0, use c = n, else c = -n.
        g = (p1 + p2 + p3) / 3.0
        if np.dot(n, g) < 0:
            c = -n
        else:
            c = n
            
        cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
        circumradii[i] = np.arccos(cos_r)
        
    return circumradii

def _optimize_covering(points, n_iterations=5000, lr_init=0.05, lr_min=0.001, force_strength=1.0):
    """
    Optimize points to minimize covering radius.
    Uses gradient of the max circumradius (approximated by sum of squared circumradii or moving vertices of worst facets).
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = np.inf
    
    # Initial hull
    try:
        hull = ConvexHull(points)
        circumradii = _compute_facet_circumradii(points, hull)
        best_radius = np.max(circumradii)
    except Exception:
        return points, np.inf

    for it in range(n_iterations):
        # Recompute hull and circumradii
        try:
            hull = ConvexHull(points)
            circumradii = _compute_facet_circumradii(points, hull)
            current_max_radius = np.max(circumradii)
            
            if current_max_radius < best_radius:
                best_radius = current_max_radius
                best_points = points.copy()
        except Exception:
            # If hull fails, break
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        n_points = len(points)
        
        # For each facet, if its circumradius is large, push its vertices away from the circumcenter
        for i, simplex in enumerate(hull.simplices):
            r = circumradii[i]
            # Only consider facets that contribute significantly to the max radius
            # Use a threshold or weight by r
            if r < 0.1: # Skip very small facets
                continue
                
            p1, p2, p3 = points[simplex]
            
            # Recalculate circumcenter c for this facet
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            g = (p1 + p2 + p3) / 3.0
            if np.dot(n, g) < 0:
                c = -n
            else:
                c = n
                
            # The force on each vertex should be in the direction of the gradient of the distance to c.
            # To reduce the circumradius, we want to move the vertices away from c? 
            # No, moving vertices away from c increases the distance to c?
            # Let's think. The circumradius is the distance from c to p1.
            # If we move p1 away from c, the triangle changes, and c changes.
            # A simpler heuristic: move vertices in the direction of the normal to the facet (outwards) if the facet is "flat" (large radius)?
            # Actually, the circumcenter c is the point on the sphere. The "gap" is the cap centered at c.
            # To reduce the gap, we want to move the vertices *towards* the center of the gap? No, that would make the triangle smaller and the circumcenter might move.
            # Standard approach for sphere packing/covering: move points away from their neighbors.
            # Let's use a repulsive force between all points, weighted by distance, plus a specific correction for the worst facets.
            
            # Alternative: Move each point in the direction of the sum of vectors from its neighbors, weighted by distance.
            # This is the "force" approach.
            pass
            
        # Instead of complex facet gradients, use a global repulsive force + local refinement
        # Calculate pairwise repulsive forces
        dots = points @ points.T
        np.fill_diagonal(dots, 0.0)
        dots = np.clip(dots, -0.9999, 0.9999)
        
        # Repulsive force: F_ij = (1 - p_i.p_j) / ||p_i - p_j||^2 * (p_i - p_j) ?
        # Simpler: F_i = sum_{j != i} (p_i - (p_i.p_j)p_j) / (1 - p_i.p_j)^2
        # This is the gradient of the log-energy or similar.
        
        weights = 1.0 / (1.0 - dots) ** 2
        np.fill_diagonal(weights, 0.0)
        
        proj = dots[:, :, None] * points[None, :, :]
        perp = points[:, None, :] - proj
        
        force = np.sum(weights[:, :, None] * perp, axis=1)
        
        # Add facet-based correction
        # For the worst facets, add a force pushing vertices away from the circumcenter
        # This effectively "opens up" the large gaps.
        for i, simplex in enumerate(hull.simplices):
            r = circumradii[i]
            if r > best_radius * 0.9: # Only act on near-worst facets
                p1, p2, p3 = points[simplex]
                n = np.cross(p2 - p1, p3 - p1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-15:
                    continue
                n = n / n_norm
                g = (p1 + p2 + p3) / 3.0
                if np.dot(n, g) < 0:
                    c = -n
                else:
                    c = n
                    
                # Move vertices away from c
                # Direction: p_vertex - c
                for idx in simplex:
                    p_v = points[idx]
                    dir_vec = p_v - c
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm > 1e-15:
                        dir_vec = dir_vec / dir_norm
                        # Project to tangent plane? p_v is on sphere, c is on sphere.
                        # The movement should be on the sphere.
                        # The vector p_v - c is not necessarily tangent to the sphere at p_v.
                        # Tangent component: (p_v - c) - (p_v . (p_v - c)) p_v
                        # Since p_v . p_v = 1, p_v . (p_v - c) = 1 - p_v . c = 1 - cos(r).
                        # So tangent part = (p_v - c) - (1 - cos(r)) p_v
                        # Let's just use the direction of the gradient of the distance.
                        # Actually, moving p_v in the direction of c would reduce the distance to c?
                        # No, c is the circumcenter. We want to change the triangle so that the circumcenter moves or the vertices move such that the max distance decreases.
                        # A known heuristic: move vertices in the direction of the facet normal (outwards) if the facet is large?
                        # Let's try moving vertices towards the centroid of the *neighbors* of the facet?
                        # For now, let's just rely on the global repulsion which tends to equalize distances.
                        # The facet correction can be: move vertex in direction of (c - p_v) ? No, that moves it TO the center, shrinking the triangle?
                        # If we move p1, p2, p3 towards c, the triangle shrinks, and the circumradius might decrease?
                        # Let's test: If triangle is equilateral with side s, circumradius R.
                        # If we shrink it, R decreases.
                        # So moving vertices towards c is a good heuristic to reduce local covering radius.
                        
                        dir_to_c = c - p_v
                        # Project to tangent plane at p_v
                        tang = dir_to_c - np.dot(dir_to_c, p_v) * p_v
                        tang_norm = np.linalg.norm(tang)
                        if tang_norm > 1e-15:
                            tang = tang / tang_norm
                            # Strength proportional to how much this facet exceeds the average
                            strength = (r - np.mean(circumradii))
                            if strength > 0:
                                forces[idx] += strength * tang
                                
        # Combine forces
        total_force = force + forces
        
        # Normalize and apply
        force_norms = np.linalg.norm(total_force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = total_force / force_norms
        
        progress = it / n_iterations
        lr = lr_init * (lr_min / lr_init) ** progress
        
        points = points + lr * force_unit
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return best_points, best_radius

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Multiple starts
    starts = [
        _fibonacci_spiral(N, 0.5),
        _fibonacci_spiral(N, 0.1),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
    ]
    
    for init_pts in starts:
        # Quick electrostatic refinement to spread out
        # Simple repulsion step
        pts = init_pts.copy()
        for _ in range(100):
            dots = pts @ pts.T
            np.fill_diagonal(dots, 0.0)
            dots = np.clip(dots, -0.9999, 0.9999)
            weights = 1.0 / (1.0 - dots) ** 2
            np.fill_diagonal(weights, 0.0)
            proj = dots[:, :, None] * pts[None, :, :]
            perp = pts[:, None, :] - proj
            force = np.sum(weights[:, :, None] * perp, axis=1)
            force_norms = np.linalg.norm(force, axis=1, keepdims=True)
            force_norms = np.maximum(force_norms, 1e-15)
            force_unit = force / force_norms
            pts = pts + 0.05 * force_unit
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
        # Covering radius optimization
        refined_pts, radius = _optimize_covering(pts, n_iterations=3000)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined_pts.copy()
            
    # Final polish
    if best_points is not None:
        best_points, final_radius = _optimize_covering(best_points, n_iterations=2000)
        if final_radius < best_radius:
            best_points = best_points
            best_radius = final_radius
            
    # Normalize
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
