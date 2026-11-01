"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _angular_distance(p, q):
    """Angular distance between two unit vectors in radians."""
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)


def _covering_radius(pts):
    """Compute the covering radius (in radians) of a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the maximum circumradius of the
    facets of the convex hull.
    """
    pts = np.asarray(pts)
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        # Get the points forming this facet
        facet_pts = pts[simplex]
        
        # Compute the circumcenter of the facet on the unit sphere
        # For a spherical triangle, the circumcenter is the point equidistant from all three vertices
        # We can find it by solving for the point that maximizes the minimum distance to all facet vertices
        # Alternatively, we can use the fact that the circumradius is the angular distance from the circumcenter to any vertex
        
        # Method: find the point on the sphere that is equidistant from all vertices of the facet
        # For a triangle with vertices a, b, c on the unit sphere, the circumcenter u satisfies:
        # u·a = u·b = u·c = cos(r)
        # This gives us a linear system if we know r, but we can solve it differently.
        
        # A simpler approach: the circumcenter of a spherical triangle can be found by intersecting
        # the perpendicular bisectors. For a triangle with vertices a, b, c:
        # The perpendicular bisector of arc ab is the set of points u such that u·a = u·b
        # i.e., u·(a-b) = 0
        
        # So we need u such that:
        # u·(a-b) = 0
        # u·(a-c) = 0
        # u·u = 1
        
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        # Build the linear system
        A = np.array([a - b, a - c])
        # We want u perpendicular to both (a-b) and (a-c), so u is along the cross product
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            # Degenerate case
            continue
            
        u = u / u_norm
        
        # Check which side: we want the circumcenter to be on the same side as the facet normal
        # The facet normal is the cross product of edges
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Compute the circumradius as the angular distance from u to any vertex
        r = _angular_distance(u, a)
        
        if r > max_circumradius:
            max_circumradius = r
    
    return max_circumradius


def _repulsive_energy(pts, repulsive_power=2.0, pressure_weight=1.0):
    """Compute a smooth energy that approximates the covering radius minimization.
    
    Combines:
    1. Repulsive forces between points (inverse square of angular distance)
    2. Pressure term pushing points away from facet circumcenters
    """
    pts = np.asarray(pts)
    n = len(pts)
    
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    energy = 0.0
    
    # Repulsive energy: sum of 1/d^2 over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            d = _angular_distance(pts[i], pts[j])
            if d < 1e-10:
                energy += 1e10  # penalty for coincident points
            else:
                energy += 1.0 / (d ** repulsive_power)
    
    # Pressure term: push points away from facet circumcenters
    try:
        hull = ConvexHull(pts)
    except Exception:
        return energy
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        # Find circumcenter
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            continue
            
        u = u / u_norm
        
        # Check side
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Compute circumradius
        r = _angular_distance(u, a)
        
        # Pressure: push all points away from u, weighted by how "worst" this facet is
        for i in range(n):
            d = _angular_distance(pts[i], u)
            if d < 1e-10:
                energy += pressure_weight * 1e10
            else:
                # Stronger pressure for points close to the circumcenter
                energy += pressure_weight * r * (1.0 / (d ** 2))
    
    return energy


def _repulsive_gradient(pts, repulsive_power=2.0, pressure_weight=1.0):
    """Compute the gradient of the repulsive energy with respect to point positions.
    
    Returns a gradient array of shape (n, 3).
    """
    pts = np.asarray(pts)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Repulsive gradient: for each pair (i,j), the force on i is proportional to (p_i - p_j) / d^3
    # (since d = arccos(p_i · p_j), and d(p_i · p_j)/dp_i = p_j, so the gradient involves p_j)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0)
            d = np.arccos(dot)
            if d < 1e-10:
                continue
            # Gradient of 1/d^2 with respect to p_i
            # d(1/d^2)/dp_i = -2 * d(1/d)/dp_i = -2 * (-1/d^2) * d(1/d)/dp_i... 
            # Let's use: d = arccos(p_i · p_j)
            # dd/dp_i = -p_j / sqrt(1 - (p_i·p_j)^2) = -p_j / sin(d)
            # d(1/d^2)/dp_i = -2/d^3 * dd/dp_i = -2/d^3 * (-p_j/sin(d)) = 2*p_j/(d^3 * sin(d))
            sin_d = np.sin(d)
            if sin_d < 1e-10:
                continue
            grad[i] += 2.0 * pts[j] / (d ** 3 * sin_d)
    
    # Pressure gradient: for each facet circumcenter u, push points away
    try:
        hull = ConvexHull(pts)
    except Exception:
        return grad
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            continue
            
        u = u / u_norm
        
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        r = _angular_distance(u, a)
        
        for i in range(n):
            dot = np.clip(np.dot(pts[i], u), -1.0, 1.0)
            d = np.arccos(dot)
            if d < 1e-10:
                continue
            sin_d = np.sin(d)
            if sin_d < 1e-10:
                continue
            # Gradient of r/d^2 with respect to p_i
            # d(r/d^2)/dp_i = r * d(1/d^2)/dp_i = r * 2*u/(d^3 * sin(d))
            grad[i] += pressure_weight * r * 2.0 * u / (d ** 3 * sin_d)
    
    return grad


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius.
    
    Uses a repulsive force system with pressure from facet circumcenters,
    optimized via gradient descent on the sphere.
    """
    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using gradient descent on the sphere
    # We'll use a simple projected gradient descent: move along negative gradient, then re-normalize
    
    repulsive_power = 2.0
    pressure_weight = 0.5
    lr = 0.01
    n_iter = 500
    patience = 50
    best_energy = np.inf
    best_pts = pts.copy()
    no_improve = 0
    
    for it in range(n_iter):
        # Compute gradient
        grad = _repulsive_gradient(pts, repulsive_power, pressure_weight)
        
        # Project gradient onto tangent space (remove component along radial direction)
        # For each point, the tangent space is orthogonal to the point itself
        for i in range(N):
            radial_component = np.dot(grad[i], pts[i]) * pts[i]
            grad[i] -= radial_component
        
        # Update points
        pts_new = pts - lr * grad
        
        # Re-normalize to unit sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check for coincident points
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = _angular_distance(pts_new[i], pts_new[j])
                if d < min_dist:
                    min_dist = d
                if d < 1e-6:
                    # Add small perturbation to separate coincident points
                    pts_new[j] += 0.01 * np.random.randn(3)
                    pts_new[j] /= np.linalg.norm(pts_new[j])
        
        pts = pts_new
        
        # Compute energy (approximate covering radius)
        try:
            cr = _covering_radius(pts)
        except Exception:
            cr = np.pi
        
        if cr < best_energy:
            best_energy = cr
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive learning rate
        if no_improve > patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-6:
                break
        
        # Early stopping if we've converged
        if cr < 0.32:  # ~18.3 degrees in radians
            break
    
    # Final refinement: use local optimization around the best point
    # Try a few random restarts with small perturbations
    best_cr = best_energy
    best_final_pts = best_pts.copy()
    
    for restart in range(5):
        # Perturb the best point
        perturbation = 0.05 * np.random.randn(N, 3)
        pts_trial = best_pts + perturbation
        norms = np.linalg.norm(pts_trial, axis=1, keepdims=True)
        pts_trial = pts_trial / norms
        
        # Optimize from this starting point
        lr_local = 0.005
        for it in range(200):
            grad = _repulsive_gradient(pts_trial, repulsive_power, pressure_weight)
            for i in range(N):
                radial_component = np.dot(grad[i], pts_trial[i]) * pts_trial[i]
                grad[i] -= radial_component
            pts_trial = pts_trial - lr_local * grad
            norms = np.linalg.norm(pts_trial, axis=1, keepdims=True)
            pts_trial = pts_trial / norms
        
        try:
            cr_trial = _covering_radius(pts_trial)
        except Exception:
            cr_trial = np.pi
        
        if cr_trial < best_cr:
            best_cr = cr_trial
            best_final_pts = pts_trial.copy()
    
    # Return the best configuration
    return best_final_pts
# EVOLVE-BLOCK-END
