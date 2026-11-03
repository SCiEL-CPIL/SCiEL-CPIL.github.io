"""Spherical repulsive force simulation for optimal covering of the unit sphere."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = 140.0  # safety margin before 150s kill

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull Delaunay triangulation."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return 180.0
    pts = pts / norms
    
    # Check for coincident points
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    off_diag = dots - np.eye(N)
    if np.min(off_diag) > 1 - 1e-10:
        return 180.0
    
    # Check if origin is inside convex hull (not in a hemisphere)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Compute covering radius as max circumradius over Delaunay triangles
    max_cr = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Spherical circumcenter: point equidistant from v0, v1, v2 on the sphere
        u = v1 - v0
        v = v2 - v0
        w = np.cross(u, v)
        wn = np.linalg.norm(w)
        if wn < 1e-15:
            # Degenerate, use midpoint of longest edge
            d01 = np.linalg.norm(v0 - v1)
            d02 = np.linalg.norm(v0 - v2)
            d12 = np.linalg.norm(v1 - v2)
            if d01 >= d02 and d01 >= d12:
                c = (v0 + v1) / 2.0
            elif d02 >= d01 and d02 >= d12:
                c = (v0 + v2) / 2.0
            else:
                c = (v1 + v2) / 2.0
            cn = np.linalg.norm(c)
            if cn < 1e-15:
                continue
            c = c / cn
        else:
            c = w / wn
            # Choose the hemisphere containing the face
            if np.dot(c, v0) < 0:
                c = -c
        
        # Angular distance from c to v0 (should be same for v1, v2)
        cos_d = np.clip(np.dot(c, v0), -1.0, 1.0)
        ang = np.arccos(cos_d)
        if ang > max_cr:
            max_cr = ang
    
    return np.degrees(max_cr)


def _fibonacci_sphere(seed=0):
    """Generate n points on the sphere using the Fibonacci spiral with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, zq = q
    R = np.array([
        [1 - 2*(y*y + zq*zq), 2*(x*y - w*zq), 2*(x*zq + w*y)],
        [2*(x*y + w*zq), 1 - 2*(x*x + zq*zq), 2*(y*zq - w*x)],
        [2*(x*zq - w*y), 2*(y*zq + w*x), 1 - 2*(x*x + y*y)]
    ])
    pts = pts @ R.T
    return pts


def _compute_repulsive_forces(pts, k=1.0):
    """
    Compute repulsive forces between points on the sphere using inverse square law.
    Points act as charged particles; force magnitude ~ 1/r^2 where r is angular distance.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Compute pairwise dot products
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    # Angular distances
    angles = np.arccos(dots)
    
    # For each point, compute net repulsive force
    for i in range(n):
        # Skip self-interaction
        angles_i = angles[i].copy()
        angles_i[i] = np.inf
        
        # Force magnitude: k / angle^2 (repulsive, so pushes away)
        # But we need the direction: away from other points
        # The force on point i from point j is in the direction (pts[i] - pts[j]) normalized
        # Project onto tangent plane at pts[i]
        
        for j in range(n):
            if i == j:
                continue
            ang = angles[i, j]
            if ang < 1e-10:
                continue
            
            # Force magnitude (inverse square of angular distance)
            f_mag = k / (ang * ang)
            
            # Direction: from j to i, projected onto tangent plane at i
            d = pts[i] - pts[j]
            # Project onto tangent plane: remove component along pts[i]
            d_tangent = d - np.dot(d, pts[i]) * pts[i]
            d_norm = np.linalg.norm(d_tangent)
            if d_norm > 1e-15:
                d_tangent = d_tangent / d_norm
                forces[i] += f_mag * d_tangent
    
    return forces


def _simulate_repulsion(pts, deadline, n_steps=2000, initial_temp=0.05, cooling_rate=0.999):
    """
    Simulate repulsive forces with adaptive step size and cooling.
    """
    pts = np.array(pts, dtype=np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    rng = np.random.RandomState(42)
    temp = initial_temp
    
    for step in range(n_steps):
        if time.time() > deadline - 10:
            break
        
        # Compute repulsive forces
        forces = _compute_repulsive_forces(pts, k=1.0)
        
        # Add small random perturbation to escape local minima
        noise = rng.randn(*pts.shape) * temp
        forces += noise
        
        # Adaptive step size: limit movement to avoid instability
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags) if len(force_mags) > 0 else 1.0
        
        # Step size inversely proportional to force magnitude, capped
        step_size = 0.01 / (1.0 + max_force * 0.1)
        
        # Update positions
        pts_new = pts + step_size * forces
        
        # Project back to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts_new = pts_new / norms
        
        # Check for convergence
        move = np.linalg.norm(pts_new - pts, axis=1)
        if np.max(move) < 1e-8:
            break
        
        pts = pts_new
        
        # Cool down
        temp *= cooling_rate
    
    return pts


def _refine_with_local_opt(pts, deadline):
    """
    Refine using local optimization on the worst facets.
    """
    pts = np.array(pts, dtype=np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Compute covering radius and find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_cr = 0.0
        worst_simplex = None
        worst_circumcenter = None
        
        for simplex in hull.simplices:
            if len(simplex) != 3:
                continue
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            u = v1 - v0
            v = v2 - v0
            w = np.cross(u, v)
            wn = np.linalg.norm(w)
            if wn < 1e-15:
                continue
            
            c = w / wn
            if np.dot(c, v0) < 0:
                c = -c
            
            cos_d = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_d)
            if ang > max_cr:
                max_cr = ang
                worst_simplex = simplex
                worst_circumcenter = c
        
        if worst_simplex is None:
            break
        
        # Try to improve the worst facet by moving one of its vertices
        # Move the vertex that is farthest from the circumcenter
        v0, v1, v2 = pts[worst_simplex[0]], pts[worst_simplex[1]], pts[worst_simplex[2]]
        
        # Find which vertex to move
        d0 = np.linalg.norm(v0 - worst_circumcenter)
        d1 = np.linalg.norm(v1 - worst_circumcenter)
        d2 = np.linalg.norm(v2 - worst_circumcenter)
        
        if d0 >= d1 and d0 >= d2:
            idx = worst_simplex[0]
        elif d1 >= d0 and d1 >= d2:
            idx = worst_simplex[1]
        else:
            idx = worst_simplex[2]
        
        # Try small perturbations in the direction away from circumcenter
        best_move = None
        best_new_cr = _covering_radius_deg(pts)
        
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random direction in tangent plane
            random_vec = rng.randn(3)
            tangent = random_vec - np.dot(random_vec, pts[idx]) * pts[idx]
            tangent /= np.linalg.norm(tangent)
            
            # Try moving in this direction
            for step in [0.001, 0.005, 0.01, 0.02]:
                new_pts = pts.copy()
                new_pts[idx] = pts[idx] + step * tangent
                new_pts[idx] /= np.linalg.norm(new_pts[idx])
                
                new_cr = _covering_radius_deg(new_pts)
                if new_cr < best_new_cr - 1e-10:
                    best_new_cr = new_cr
                    best_move = new_pts.copy()
        
        if best_move is not None:
            pts = best_move
        else:
            # No improvement found; break
            break
    
    return pts


def construct():
    start_time = time.time()
    rng = np.random.RandomState(0)
    
    # Generate multiple initializations
    inits = []
    for seed in range(8):
        inits.append(_fibonacci_sphere(seed))
    
    # Evaluate all initializations
    init_scores = []
    for pts in inits:
        if time.time() > _DEADLINE - 60:
            break
        cr = _covering_radius_deg(pts)
        init_scores.append((cr, pts))
    
    if not init_scores:
        return _fibonacci_sphere(0)
    
    # Sort by covering radius
    init_scores.sort(key=lambda x: x[0])
    best_cr, best_pts = init_scores[0]
    
    # Run repulsive force simulation on top initializations
    for cr, pts in init_scores[:4]:
        if time.time() > _DEADLINE - 30:
            break
        try:
            # Simulate repulsion
            sim_pts = _simulate_repulsion(pts, deadline=_DEADLINE, n_steps=1500)
            cr_sim = _covering_radius_deg(sim_pts)
            
            if cr_sim < best_cr:
                best_cr = cr_sim
                best_pts = sim_pts.copy()
            
            # Further refine with local optimization
            if time.time() < _DEADLINE - 10:
                refined_pts = _refine_with_local_opt(sim_pts, deadline=_DEADLINE)
                cr_ref = _covering_radius_deg(refined_pts)
                if cr_ref < best_cr:
                    best_cr = cr_ref
                    best_pts = refined_pts.copy()
                    
        except Exception:
            continue
    
    # Final normalization and validation
    best_pts = np.asarray(best_pts, dtype=np.float64)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return _fibonacci_sphere(0)
    best_pts = best_pts / norms
    
    return best_pts
