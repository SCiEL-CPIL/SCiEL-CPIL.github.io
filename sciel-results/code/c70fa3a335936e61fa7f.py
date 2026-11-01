import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull.
    Returns (radius, worst_pole, worst_face_indices)
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, np.zeros(3), np.array([], dtype=int)
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        max_pole = normals[max_idx]
        max_face = faces[valid][max_idx]
        
        return max_radius, max_pole, max_face
    except Exception:
        return np.pi, np.zeros(3), np.array([], dtype=int)

def _perturb_point(pts, idx, rng, step_size):
    """
    Perturb a single point on the sphere.
    Moves the point in a random tangent direction.
    """
    p = pts[idx].copy()
    # Generate a random vector orthogonal to p
    # p x random_vector gives a tangent vector
    rand_vec = rng.standard_normal(3)
    tangent = np.cross(p, rand_vec)
    norm_t = np.linalg.norm(tangent)
    if norm_t < _EPS:
        # Fallback if cross product is zero (unlikely but possible)
        tangent = np.array([1.0, 0.0, 0.0])
        if abs(p[0]) > 0.9:
            tangent = np.array([0.0, 1.0, 0.0])
        tangent = tangent - np.dot(tangent, p) * p
        norm_t = np.linalg.norm(tangent)
    
    tangent /= norm_t
    
    # Move along tangent
    new_p = p + step_size * tangent
    
    # Re-normalize to sphere
    new_p /= (np.linalg.norm(new_p) + _EPS)
    
    # Update pts
    pts_new = pts.copy()
    pts_new[idx] = new_p
    return pts_new

def _hill_climb_sa(pts, deadline, rng):
    """
    Stochastic Hill Climbing with Simulated Annealing.
    Focuses perturbations on the vertices of the worst facet (largest circumradius).
    """
    current_pts = pts.copy()
    current_radius, current_pole, current_face = _compute_hull_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    # Initial temperature
    T = 0.1  # Radians
    T_min = 1e-4
    alpha = 0.9995 # Cooling rate
    
    # Track time
    start_time = time.time()
    
    # We will perform many iterations. 
    # Each iteration involves:
    # 1. Identifying worst facet (already done or updated)
    # 2. Perturbing one of the 3 vertices of the worst facet
    # 3. Computing new hull radius
    # 4. Accept/Reject
    
    # To speed up, we can cache the hull if possible, but ConvexHull is O(N log N) or similar.
    # For N=130, it's fast enough. ~1000-5000 iterations per second? 
    # Let's aim for as many as possible.
    
    while time.time() < deadline:
        # Identify vertices of the current worst facet
        if len(current_face) == 0:
            # Degenerate case, try random perturbation
            idx = rng.integers(0, N)
        else:
            # Pick one of the 3 vertices of the worst facet
            # Weighted selection? Just pick randomly among the 3.
            idx = current_face[rng.integers(0, 3)]
            
        # Determine step size
        # Step size can be proportional to T or fixed
        step_size = T * 0.5 
        
        # Generate new point
        new_pts = _perturb_point(current_pts, idx, rng, step_size)
        
        # Evaluate new radius
        new_radius, new_pole, new_face = _compute_hull_radius(new_pts)
        
        # Acceptance criterion
        if new_radius < current_radius:
            # Always accept improvement
            current_pts = new_pts
            current_radius = new_radius
            current_pole = new_pole
            current_face = new_face
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
        else:
            # Accept with probability exp(-(new_radius - current_radius) / T)
            delta = new_radius - current_radius
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_pts = new_pts
                current_radius = new_radius
                current_pole = new_pole
                current_face = new_face
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat slightly if stuck? Or just reset to a small value?
            # For now, keep T at min.
            T = T_min
            
        # Periodic check to ensure we don't spend too much time on a single bad local search
        # If no improvement for a long time, maybe restart from best?
        # Simple strategy: if we haven't improved best in the last 500 iterations, 
        # reset current to best and increase T slightly.
        # This requires tracking iteration count and last improvement.
        # Let's add a simple restart mechanism.
        
    return best_pts, best_radius

def _multi_restart_search(deadline):
    """
    Run multiple restarts of SA/Hill Climbing.
    """
    rng = np.random.default_rng(42)
    best_pts = _initial_fib()
    best_r, _, _ = _compute_hull_radius(best_pts)
    
    # Allocate time to restarts
    # Let's do 3-5 restarts depending on time.
    # Each restart gets a chunk of time.
    n_restarts = 4
    time_per_restart = (deadline - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            # Start from Fibonacci
            pts = _initial_fib()
        else:
            # Start from random or slightly perturbed best
            if rng.random() < 0.5:
                pts = _random_init(rng)
            else:
                # Perturb best solution significantly
                pts = best_pts.copy()
                for _ in range(10):
                    idx = rng.integers(0, N)
                    pts = _perturb_point(pts, idx, rng, 0.5)
        
        # Run SA/Hill Climbing for the allocated time
        local_deadline = time.time() + time_per_restart
        pts_polished, r_polished = _hill_climb_sa(pts, local_deadline, rng)
        
        if r_polished < best_r:
            best_r = r_polished
            best_pts = pts_polished.copy()
            
    # Final polish with remaining time
    if time.time() < deadline - 2.0:
        # Use a higher initial temperature for the final push to escape deep local minima?
        # Or just continue SA from best.
        final_deadline = deadline - 1.0
        best_pts, best_r = _hill_climb_sa(best_pts, final_deadline, rng)
        
    return best_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin
    best_pts = _multi_restart_search(deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
