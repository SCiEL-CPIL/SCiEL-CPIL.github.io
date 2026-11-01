import time
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist

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

def _compute_radius(pts):
    """
    Compute the covering radius.
    Returns: (radius_in_radians, worst_facet_indices, pole)
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi, None, None
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        max_r = np.max(radii)
        
        # Identify the worst facet
        worst_idx = np.argmax(radii)
        
        valid_faces = faces[valid]
        worst_face = valid_faces[worst_idx]
        pole = normals[worst_idx]
        
        return max_r, worst_face, pole
        
    except Exception:
        return np.pi, None, None

def _spherical_dist(a, b):
    """Compute angular distance between two unit vectors."""
    dot = np.dot(a, b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _move_point_towards(pts, idx, target, step):
    """
    Move point idx on the sphere towards target by 'step' radians.
    """
    p = pts[idx]
    t = target
    
    # Normalize target
    t_norm = np.linalg.norm(t)
    if t_norm < _EPS:
        return pts
    
    t = t / t_norm
    
    # Geodesic interpolation (Slerp)
    # d = angular distance
    d = _spherical_dist(p, t)
    
    if d < _EPS:
        return pts
        
    # We want to move a fraction 'step' of the distance d
    # Actually, let's just move by a fixed small angle towards t.
    # Or move by 'step' radians.
    # Let's use Slerp with alpha = min(1, step / d) if step is distance?
    # No, step is a parameter. Let's interpret step as the angle to move.
    
    alpha = np.clip(step / d, 0.0, 1.0)
    
    sin_d = np.sin(d)
    if sin_d < _EPS:
        new_p = p
    else:
        new_p = (np.sin((1 - alpha) * d) / sin_d) * p + (np.sin(alpha * d) / sin_d) * t
        
    new_p = new_p / (np.linalg.norm(new_p) + _EPS)
    
    new_pts = pts.copy()
    new_pts[idx] = new_p
    return new_pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Initialize with Fibonacci
    best_pts = _initial_fib()
    best_r, best_face, best_pole = _compute_radius(best_pts)
    
    # Simulated Annealing Parameters
    # We will use a "targeted" SA where we preferentially perturb points
    # involved in the worst facet, but also allow global perturbations.
    
    T = 0.05  # Initial temperature (radians squared? or just scale)
    T_min = 0.001
    cooling_rate = 0.995
    
    # Pre-compute tree for current best? 
    # SA needs fast evaluation. 
    # Strategy: 
    # 1. Pick a point to move.
    # 2. Move it slightly.
    # 3. Evaluate new radius.
    # 4. Accept or reject.
    
    # To make it fast, we can approximate the radius change?
    # No, convex hull is too expensive for thousands of steps if we do it naively?
    # ConvexHull for N=130 is fast (~ms). 
    # 1000 steps * 5ms = 5s. This is feasible.
    
    # However, we can optimize by only checking if the *worst* facet improved
    # or if a new worse facet appeared. 
    # But checking "new worse facet" requires full hull or at least checking neighbors.
    # Let's just do full hull evaluation for simplicity and robustness, 
    # but limit the number of steps to fit in time.
    
    # Estimate: 150s / 0.01s per step = 15000 steps.
    # Let's aim for 5000-10000 steps.
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    # Build KD-tree for initial points to find nearest neighbors for perturbation selection?
    # Or just random index.
    
    n_steps = 0
    stuck_count = 0
    
    while time.time() < deadline - 5.0:
        # Select a point to perturb
        # Bias towards points in the current worst facet
        if best_face is not None and rng.random() < 0.7:
            # Pick one of the 3 vertices of the worst facet
            candidate_indices = best_face
            idx = rng.choice(candidate_indices)
        else:
            idx = rng.integers(0, N)
        
        # Determine perturbation magnitude
        # Scale with temperature
        mag = T * np.sqrt(rng.chisquare(3))
        
        # Generate a random direction on the tangent plane at current_pts[idx]
        p = current_pts[idx]
        
        # Random vector
        v = rng.standard_normal(3)
        # Project to tangent plane: v - (v.p)p
        v = v - np.dot(v, p) * p
        v_norm = np.linalg.norm(v)
        if v_norm < _EPS:
            continue
        v = v / v_norm
        
        # New position
        new_p = p + mag * v
        new_p = new_p / (np.linalg.norm(new_p) + _EPS)
        
        # Apply move
        trial_pts = current_pts.copy()
        trial_pts[idx] = new_p
        
        # Evaluate
        trial_r, trial_face, trial_pole = _compute_radius(trial_pts)
        
        # Acceptance criterion
        delta_r = trial_r - current_r
        if delta_r < 0 or rng.random() < np.exp(-delta_r / T):
            # Accept
            current_pts = trial_pts
            current_r = trial_r
            stuck_count = 0
            
            # Update best
            if current_r < best_r:
                best_r = current_r
                best_pts = current_pts.copy()
                best_face = trial_face
                best_pole = trial_pole
        else:
            stuck_count += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        n_steps += 1
        
        # Optional: If stuck for too long, restart temperature or do a larger jump
        if stuck_count > 100:
            # Random restart of temperature
            T = max(T, 0.02)
            stuck_count = 0
            
        # Periodic global perturbation to escape deep local minima
        if n_steps % 500 == 0 and time.time() < deadline - 20.0:
            # Perturb a random subset of points
            k = 5
            indices = rng.choice(N, k, replace=False)
            for i in indices:
                p = current_pts[i]
                v = rng.standard_normal(3)
                v = v - np.dot(v, p) * p
                v = v / (np.linalg.norm(v) + _EPS)
                new_p = p + 0.1 * v
                new_p = new_p / (np.linalg.norm(new_p) + _EPS)
                current_pts[i] = new_p
            
            # Re-evaluate
            current_r, current_face, current_pole = _compute_radius(current_pts)
            if current_r < best_r:
                best_r = current_r
                best_pts = current_pts.copy()
                best_face = current_face
                best_pole = current_pole
            # Reset temperature slightly
            T = max(T, 0.05)

    # Final polish: A few steps of Lloyd's on the best found?
    # Or just return best.
    # Let's do a quick local optimization on best_pts to squeeze out last bits.
    # We have ~5s left.
    
    if time.time() < deadline - 2.0:
        # Simple gradient-free local polish using small random steps (hill climbing)
        polish_T = 0.005
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            idx = rng.integers(0, N)
            p = best_pts[idx]
            v = rng.standard_normal(3)
            v = v - np.dot(v, p) * p
            v = v / (np.linalg.norm(v) + _EPS)
            new_p = p + polish_T * v
            new_p = new_p / (np.linalg.norm(new_p) + _EPS)
            
            trial_pts = best_pts.copy()
            trial_pts[idx] = new_p
            trial_r, _, _ = _compute_radius(trial_pts)
            
            if trial_r < best_r:
                best_r = trial_r
                best_pts = trial_pts

    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
