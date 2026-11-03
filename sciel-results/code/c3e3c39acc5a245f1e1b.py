import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _covering_radius_hull(points):
    """Compute covering radius using convex hull facet circumradii."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        center = normal
        if np.dot(center, p1) < 0 or np.dot(center, p2) < 0 or np.dot(center, p3) < 0:
            center = -center
        
        dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        max_radius = max(max_radius, dist)
    
    return max_radius

def _repulsive_force(pts, power=4.0):
    """Compute repulsive force on each point from all others."""
    forces = np.zeros_like(pts)
    n = len(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            # Force proportional to 1/dist^power, direction away from j
            f = diff / (dist ** (power + 1))
            forces[i] += f
    return forces

def _optimize_repulsive(pts, deadline, iters_per_phase=500, step_sizes=None):
    """Optimize using repulsive force with adaptive step size."""
    if step_sizes is None:
        step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]
    
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_hull(best)
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        
        for _ in range(iters_per_phase):
            if time.time() > deadline:
                break
            
            forces = _repulsive_force(current, power=4.0)
            # Normalize forces
            f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            f_norms[f_norms < 1e-10] = 1.0
            forces = forces / f_norms
            
            new_pts = current + step * forces
            new_pts = _normalize(new_pts)
            
            # Check if all points are distinct
            if np.any(np.linalg.norm(new_pts[:, None, :] - new_pts[None, :, :], axis=2) < 1e-8):
                continue
            
            # Evaluate new configuration
            new_radius = _covering_radius_hull(new_pts)
            
            if new_radius < best_radius - 1e-9:
                best = new_pts.copy()
                best_radius = new_radius
                current = new_pts.copy()
            else:
                # Reject move, keep current
                pass
    
    return best

def _local_refinement_hull(pts, deadline):
    """Local refinement: for each point, try small perturbations to reduce max facet radius."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_hull(best)
    
    # Get current hull
    try:
        hull = ConvexHull(current)
    except Exception:
        return best
    
    # Identify facets and their circumradii
    facet_radii = []
    for simplex in hull.simplices:
        p1 = current[simplex[0]]
        p2 = current[simplex[1]]
        p3 = current[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            facet_radii.append(0.0)
            continue
        normal = normal / normal_norm
        
        center = normal
        if np.dot(center, p1) < 0 or np.dot(center, p2) < 0 or np.dot(center, p3) < 0:
            center = -center
        
        dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        facet_radii.append(dist)
    
    facet_radii = np.array(facet_radii)
    
    # For several iterations, find the largest facet and try to reduce it
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Find largest facet
        max_idx = np.argmax(facet_radii)
        if facet_radii[max_idx] < 1e-6:
            break
        
        simplex = hull.simplices[max_idx]
        p1_idx, p2_idx, p3_idx = simplex
        
        # Try moving each of the three points slightly toward the facet center
        v1 = current[p2_idx] - current[p1_idx]
        v2 = current[p3_idx] - current[p1_idx]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        center = normal
        if np.dot(center, current[p1_idx]) < 0:
            center = -center
        
        # Try small moves of each vertex toward center
        improved = False
        for idx in [p1_idx, p2_idx, p3_idx]:
            for step in [0.01, 0.005, 0.002]:
                new_pts = current.copy()
                direction = center - new_pts[idx]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    continue
                direction = direction / dir_norm
                new_pts[idx] = new_pts[idx] + step * direction
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
                
                new_radius = _covering_radius_hull(new_pts)
                if new_radius < best_radius - 1e-9:
                    best = new_pts.copy()
                    best_radius = new_radius
                    current = new_pts.copy()
                    improved = True
                    # Recompute hull and facet radii
                    try:
                        hull = ConvexHull(current)
                        facet_radii = []
                        for simp in hull.simplices:
                            a = current[simp[0]]
                            b = current[simp[1]]
                            c = current[simp[2]]
                            n1 = b - a
                            n2 = c - a
                            nm = np.cross(n1, n2)
                            nm_norm = np.linalg.norm(nm)
                            if nm_norm < 1e-10:
                                facet_radii.append(0.0)
                                continue
                            nm = nm / nm_norm
                            cc = nm
                            if np.dot(cc, a) < 0:
                                cc = -cc
                            d = np.arccos(np.clip(np.dot(cc, a), -1.0, 1.0))
                            facet_radii.append(d)
                        facet_radii = np.array(facet_radii)
                    except Exception:
                        break
                    break
            if improved:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Repulsive force optimization
    try:
        optimized = _optimize_repulsive(pts, deadline - 40.0, iters_per_phase=200)
        if np.all(np.isfinite(optimized)):
            pts = optimized
    except Exception:
        pass
    
    # Phase 2: Local refinement targeting largest facets
    try:
        refined = _local_refinement_hull(pts, deadline - 5.0)
        if np.all(np.isfinite(refined)):
            pts = refined
    except Exception:
        pass
    
    # Final validation
    pts = _normalize(pts)
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
    
    return pts
