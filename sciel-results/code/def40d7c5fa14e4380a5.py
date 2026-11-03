"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n

def _facet_circumradii(pts, hull=None):
    """Compute circumradius (in radians) for each facet of the convex hull."""
    if hull is None:
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    for k, simplex in enumerate(simplices):
        p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        det = aa * bb - ab * ab
        if abs(det) < 1e-15:
            radii[k] = np.pi
            continue
        t = 0.5 * aa * (bb - ab) / det
        s = (0.5 * aa - t * ab) / aa
        d = s * a + t * b
        C = p0 + d
        cn = np.linalg.norm(C)
        if cn < 1e-12:
            radii[k] = np.pi
            continue
        C_unit = C / cn
        r0 = np.arccos(np.clip(np.dot(C_unit, p0), -1, 1))
        r1 = np.arccos(np.clip(np.dot(C_unit, p1), -1, 1))
        r2 = np.arccos(np.clip(np.dot(C_unit, p2), -1, 1))
        radii[k] = max(r0, r1, r2)
    return radii

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    radii = _facet_circumradii(pts)
    if radii is None:
        return 180.0
    return np.degrees(np.max(radii))

def _covering_radius_rad(pts):
    """Compute covering radius in radians via convex hull facet circumradii."""
    radii = _facet_circumradii(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _smooth_surrogate(pts, beta=200.0):
    """
    Smooth surrogate for max facet circumradius using log-sum-exp.
    Returns (value, gradient w.r.t. pts).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)
    
    radii = _facet_circumradii(pts, hull)
    if radii is None:
        return np.pi, np.zeros_like(pts)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    total = np.sum(weights)
    if total < 1e-30:
        return max_r, np.zeros_like(pts)
    
    # Soft max value
    soft_max = max_r + np.log(total) / beta
    
    # Gradient of soft max w.r.t. each radius
    # d(soft_max)/d(radii_k) = weights[k] / total
    grad_radii = weights / total
    
    # Now compute gradient w.r.t. pts for each facet
    # For each facet, the circumradius depends on the three vertices
    # We need d(radii_k)/d(pts_i) for each vertex i in facet k
    
    grad_pts = np.zeros_like(pts)
    simplices = hull.simplices
    
    for k in range(len(simplices)):
        if grad_radii[k] < 1e-15:
            continue
        simplex = simplices[k]
        p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        det = aa * bb - ab * ab
        if abs(det) < 1e-15:
            continue
        t = 0.5 * aa * (bb - ab) / det
        s = (0.5 * aa - t * ab) / aa
        d = s * a + t * b
        C = p0 + d
        cn = np.linalg.norm(C)
        if cn < 1e-12:
            continue
        C_unit = C / cn
        r0 = np.arccos(np.clip(np.dot(C_unit, p0), -1, 1))
        r1 = np.arccos(np.clip(np.dot(C_unit, p1), -1, 1))
        r2 = np.arccos(np.clip(np.dot(C_unit, p2), -1, 1))
        facet_r = max(r0, r1, r2)
        
        # Determine which vertex gives the max radius
        # Gradient of arccos(dot(C_unit, p_i)) w.r.t. p_i
        # = -C_unit / sqrt(1 - dot(C_unit, p_i)^2)
        # But we also need to account for C_unit changing with p_i
        
        # Simplified: use numerical gradient for each vertex
        eps = 1e-5
        for vi, idx in enumerate(simplex):
            # Numerical gradient of facet_r w.r.t. pts[idx]
            for axis in range(3):
                trial = pts.copy()
                trial[idx][axis] += eps
                trial[idx] = trial[idx] / np.linalg.norm(trial[idx])
                r_trial = _facet_circumradii(trial, hull)[k] if _facet_circumradii(trial, hull) is not None else np.pi
                grad_component = (r_trial - facet_r) / eps
                grad_pts[idx][axis] += grad_radii[k] * grad_component
    
    return soft_max, grad_pts

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=100, learning_rate=0.01):
    """
    Refine points using electrostatic repulsion to spread them out.
    This helps escape local minima by encouraging uniform distribution.
    """
    pts = _normalize(pts.copy())
    for it in range(n_iters):
        # Compute repulsive forces
        # Force on point i from point j is proportional to (p_i - p_j) / |p_i - p_j|^3
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                force = diff / (dist ** 3)
                forces[i] += force
        
        # Normalize forces and take a small step
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-10:
                forces[i] = forces[i] / f_norm
        
        # Update points
        new_pts = pts + learning_rate * forces
        new_pts = _normalize(new_pts)
        
        # Check if covering radius improved
        r_old = _covering_radius_rad(pts)
        r_new = _covering_radius_rad(new_pts)
        if r_new < r_old:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = pts + learning_rate * 0.1 * forces
            new_pts = _normalize(new_pts)
            r_new = _covering_radius_rad(new_pts)
            if r_new < r_old:
                pts = new_pts
            learning_rate *= 0.95
    
    return pts

def _lloyd_step(pts, n_iters=10):
    """Lloyd's algorithm: move each point to centroid of its Voronoi cell."""
    pts = _normalize(pts.copy())
    for _ in range(n_iters):
        try:
            hull = ConvexHull(pts)
            new_pts = pts.copy()
            for i in range(N):
                facet_circumcenters = []
                for simplex in hull.simplices:
                    if i in simplex:
                        p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                        a = p1 - p0
                        b = p2 - p0
                        aa = np.dot(a, a)
                        bb = np.dot(b, b)
                        ab = np.dot(a, b)
                        det = aa * bb - ab * ab
                        if abs(det) < 1e-15:
                            continue
                        t = 0.5 * aa * (bb - ab) / det
                        s = (0.5 * aa - t * ab) / aa
                        d = s * a + t * b
                        C = p0 + d
                        cn = np.linalg.norm(C)
                        if cn < 1e-12:
                            continue
                        facet_circumcenters.append(C / cn)
                
                if facet_circumcenters:
                    centroid = np.mean(facet_circumcenters, axis=0)
                    cn = np.linalg.norm(centroid)
                    if cn > 1e-12:
                        centroid = centroid / cn
                        new_pts[i] = 0.7 * pts[i] + 0.3 * centroid
                        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        except Exception:
            pass
        pts = _normalize(new_pts)
    return pts

def _nelder_mead_polish(pts, max_iters=200, tol=1e-6):
    """Polish the configuration using Nelder-Mead on the worst points."""
    pts = _normalize(pts.copy())
    best_r = _covering_radius_rad(pts)
    
    for it in range(max_iters):
        # Find the worst point (part of the largest facet)
        try:
            hull = ConvexHull(pts)
            radii = _facet_circumradii(pts, hull)
            if radii is None:
                break
            worst_idx = np.argmax(radii)
            worst_facet = hull.simplices[worst_idx]
            worst_r = radii[worst_idx]
        except Exception:
            break
        
        # Try small perturbations of the points in the worst facet
        improved = False
        for idx in worst_facet:
            for scale in [0.01, 0.005, 0.001]:
                for axis in range(3):
                    for sign in [-1, 1]:
                        trial = pts.copy()
                        trial[idx][axis] += sign * scale
                        trial[idx] = trial[idx] / np.linalg.norm(trial[idx])
                        r = _covering_radius_rad(trial)
                        if r < best_r - 1e-9:
                            pts = trial
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    best_pts = pts.copy()
    best_r = _covering_radius_rad(pts)
    
    # Phase 1: Electrostatic refinement to spread points
    if time.time() < deadline - 60:
        pts = _electrostatic_refine(pts, n_iters=50, learning_rate=0.005)
        r = _covering_radius_rad(pts)
        if r < best_r:
            best_pts = pts.copy()
            best_r = r
    
    # Phase 2: Lloyd's algorithm
    if time.time() < deadline - 40:
        pts = _lloyd_step(best_pts, n_iters=30)
        r = _covering_radius_rad(pts)
        if r < best_r:
            best_pts = pts.copy()
            best_r = r
    
    # Phase 3: Nelder-Mead polishing
    if time.time() < deadline - 20:
        pts, r = _nelder_mead_polish(best_pts, max_iters=100)
        if r < best_r:
            best_pts = pts.copy()
            best_r = r
    
    # Phase 4: More Lloyd iterations
    if time.time() < deadline - 10:
        pts = _lloyd_step(best_pts, n_iters=20)
        r = _covering_radius_rad(pts)
        if r < best_r:
            best_pts = pts.copy()
            best_r = r
    
    # Phase 5: Final polishing
    if time.time() < deadline - 5:
        pts, r = _nelder_mead_polish(best_pts, max_iters=50)
        if r < best_r:
            best_pts = pts.copy()
            best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
