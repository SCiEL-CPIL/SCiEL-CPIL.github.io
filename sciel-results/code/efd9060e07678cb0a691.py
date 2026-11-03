"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_covering_radius_and_grad(points):
    """
    Compute the covering radius and the gradient direction to minimize it.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    To minimize the covering radius, we should move the points of the facet(s)
    defining the maximum radius towards their circumcenter.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: return a large radius and zero gradient
        return np.inf, np.zeros_like(points)

    max_radius = 0.0
    max_facets = []
    
    # Pre-compute circumradii for all facets
    radii = np.zeros(len(hull.simplices))
    
    for i, simplex in enumerate(hull.simplices):
        facet_pts = points[simplex]
        a, b, c = facet_pts
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            radii[i] = 0.0
            continue
            
        # Circumcenter of the triangle on the sphere
        # The circumcenter is along the normal vector o
        o = cross / norm_cross
        if np.dot(o, a) < 0:
            o = -o
        
        cos_angle = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        radii[i] = angle
        
        if angle > max_radius:
            max_radius = angle
            max_facets = [i]
        elif np.isclose(angle, max_radius, rtol=1e-10):
            max_facets.append(i)

    # Compute gradient: sum of (circumcenter - point) for points in max facets
    # Actually, we want to move points towards the circumcenter.
    # Gradient of radius w.r.t point p_i is roughly (p_i - circumcenter) / |p_i - circumcenter|
    # But we want to MINIMIZE radius, so we move towards circumcenter.
    # Direction = circumcenter - p_i
    
    grad = np.zeros_like(points)
    for idx in max_facets:
        simplex = hull.simplices[idx]
        facet_pts = points[simplex]
        a, b, c = facet_pts
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            continue
            
        o = cross / norm_cross
        if np.dot(o, a) < 0:
            o = -o
            
        # Move a, b, c towards o
        for p in facet_pts:
            # Tangential component of (o - p)
            tang = o - np.dot(o, p) * p
            grad += tang

    # Normalize gradient
    norm_grad = np.linalg.norm(grad, axis=1, keepdims=True)
    norm_grad = np.maximum(norm_grad, 1e-15)
    grad_unit = grad / norm_grad
    
    return max_radius, grad_unit


def _electrostatic_forces(points, exponent=2.0):
    """Compute repulsive forces to maintain separation."""
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    dots = np.clip(dots, -0.9999, 0.9999)
    
    # Force is repulsive, pushing points apart
    # F_i = sum_j w_ij * (p_i - (p_i.p_j)p_j)
    weights = 1.0 / (1.0 - dots) ** exponent
    np.fill_diagonal(weights, 0.0)
    
    proj = dots[:, :, None] * points[None, :, :]
    perp = points[:, None, :] - proj
    
    force = np.sum(weights[:, :, None] * perp, axis=1)
    return force


def _refine(points, n_steps=2000):
    """
    Hybrid refinement:
    1. Electrostatic repulsion to prevent clustering.
    2. Covering radius gradient descent to minimize the max distance.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius_and_grad(points)[0]
    
    for i in range(n_steps):
        # Phase 1: Electrostatic relaxation
        # Use a small learning rate to gently push apart
        elec_force = _electrostatic_forces(points, exponent=2.0)
        elec_norm = np.linalg.norm(elec_force, axis=1, keepdims=True)
        elec_norm = np.maximum(elec_norm, 1e-15)
        elec_grad = elec_force / elec_norm
        
        # Phase 2: Covering radius gradient
        radius, cov_grad = _compute_covering_radius_and_grad(points)
        
        # Combine gradients
        # Weights: prioritize covering radius minimization, but use electrostatics
        # to handle local minima or collisions.
        w_cov = 1.0
        w_elec = 0.1
        
        total_grad = w_cov * cov_grad + w_elec * elec_grad
        
        # Normalize total gradient
        total_norm = np.linalg.norm(total_grad, axis=1, keepdims=True)
        total_norm = np.maximum(total_norm, 1e-15)
        total_grad_unit = total_grad / total_norm
        
        # Learning rate decay
        lr = 0.05 * (0.01 / 0.05) ** (i / n_steps)
        
        # Update
        points = points + lr * total_grad_unit
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Track best
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
    return best_points, best_radius


def _fibonacci_spiral(n, offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    # Start with a good deterministic initialization
    points = _fibonacci_spiral(N)
    
    # Initial electrostatic relaxation to settle into a local minimum of energy
    # This helps distribute points more evenly before the covering radius optimization
    # which can be sensitive to initial configurations.
    elec_force = _electrostatic_forces(points, exponent=2.0)
    for i in range(500):
        elec_norm = np.linalg.norm(elec_force, axis=1, keepdims=True)
        elec_norm = np.maximum(elec_norm, 1e-15)
        elec_grad = elec_force / elec_norm
        lr = 0.01
        points = points + lr * elec_grad
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        elec_force = _electrostatic_forces(points, exponent=2.0)
        
    # Main refinement loop
    points, radius = _refine(points, n_steps=3000)
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
