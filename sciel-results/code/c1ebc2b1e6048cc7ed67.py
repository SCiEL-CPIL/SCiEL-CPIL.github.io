import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

N = 130
_EPS = 1e-12
_DEADLINE = 145.0

def _initial_fib():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(pts):
    """
    Compute per-facet covering radius and outward pole direction.
    Returns (radii, normals, faces, valid_mask).
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid], valid
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])

def _hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull.
    Returns the maximum angular distance from the sphere to the nearest point.
    """
    radii, _, _, _ = _compute_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return radii.max()

def _least_squares_polish(pts, deadline, max_iter=100):
    """
    Polish using scipy.optimize.least_squares on the facet circumradii.
    This treats the covering problem as minimizing the maximum facet radius by
    minimizing the sum of squares of (radius - target), effectively pushing
    all large facets down.
    """
    pts = pts.astype(float).copy()
    
    def residuals(x):
        p = x.reshape(N, 3)
        # Normalize to sphere
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + _EPS)
        
        radii, _, _, _ = _compute_hull_data(p)
        if len(radii) == 0:
            return np.zeros(1)
        
        # We want to minimize the max radius.
        # A common trick is to minimize (r_i - R)^2 for r_i > R, or just sum of squares of r_i.
        # However, sum of squares of r_i might not reduce the max as effectively as needed if one is huge.
        # Let's use a soft-max approximation or just minimize the sum of squares of the radii.
        # Actually, minimizing sum(r_i^2) is a valid smooth surrogate for min-max.
        return radii**2

    try:
        # Use 'trf' method which supports bounds and is robust
        # We don't have explicit bounds for points on sphere, but least_squares can handle it if we project.
        # However, standard least_squares doesn't project. We need to ensure the points stay on the sphere.
        # A better approach for least_squares is to parameterize the points differently or use a custom step.
        # Since we can't easily constrain the sphere in least_squares directly without a wrapper,
        # we will use a simple iterative projection inside the residual? No, that breaks differentiability.
        
        # Alternative: Use the fact that the problem is on a manifold.
        # Let's stick to a simpler, more robust local optimizer if least_squares is tricky with constraints.
        # Or, we can use scipy.optimize.minimize with BFGS or L-BFGS-B on the flattened points,
        # and project back to the sphere after each step? No, that's not standard.
        
        # Let's use a different strategy for the polish: 
        # Iteratively identify the worst facet and move its vertices towards the pole (Subgradient) 
        # but with adaptive step size and line search. This is more reliable for the sphere constraint.
        
        pass # Placeholder, we will implement a robust subgradient/line search below instead of least_squares if it's too complex to constrain.
        
    except Exception:
        pass
    
    # Fallback to a robust local search: "Facet Push" with line search
    best_pts = pts.copy()
    best_r = _hull_radius(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        radii, normals, faces, valid = _compute_hull_data(best_pts)
        if len(radii) == 0:
            break
            
        curr_r = radii.max()
        if curr_r < best_r:
            best_r = curr_r
            
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        pole = normals[worst_idx]
        
        # Move vertices of the worst facet towards the pole
        cand = best_pts.copy()
        step = 0.05
        
        # Line search: try a few step sizes
        improved = False
        for s in [step, step*0.5, step*0.25]:
            trial = best_pts.copy()
            for v in worst_face:
                tangent_dir = pole - trial[v]
                tangent_dir = tangent_dir - np.dot(tangent_dir, trial[v]) * trial[v]
                norm_t = np.linalg.norm(tangent_dir)
                if norm_t > _EPS:
                    tangent_dir /= norm_t
                else:
                    continue
                trial[v] = trial[v] + s * tangent_dir
                trial[v] = trial[v] / (np.linalg.norm(trial[v]) + _EPS)
            
            new_r = _hull_radius(trial)
            if new_r < curr_r:
                best_pts = trial
                best_r = new_r
                improved = True
                break
            else:
                # If it didn't improve, keep the previous best and try smaller step or break
                pass
        
        if not improved:
            # Try moving a random subset of points slightly to escape local minima
            # This acts as a local perturbation
            if it % 5 == 0:
                rng = np.random.default_rng(it)
                cand = best_pts.copy()
                n_perturb = 5
                indices = rng.choice(N, size=n_perturb, replace=False)
                for i in indices:
                    random_vec = rng.normal(size=3)
                    tangent_dir = random_vec - np.dot(random_vec, cand[i]) * cand[i]
                    norm_t = np.linalg.norm(tangent_dir)
                    if norm_t > _EPS:
                        tangent_dir /= norm_t
                    else:
                        continue
                    cand[i] = cand[i] + 0.02 * tangent_dir
                    cand[i] = cand[i] / (np.linalg.norm(cand[i]) + _EPS)
                
                new_r = _hull_radius(cand)
                if new_r < best_r:
                    best_pts = cand
                    best_r = new_r
                else:
                    # If stuck, accept a worse solution with small probability (Simulated Annealing style)
                    # But since we don't have a temperature schedule easily, just break if stuck for too long
                    pass
                    
    return best_pts, best_r

def _simulated_annealing(pts_init, deadline):
    """
    Global search using Simulated Annealing.
    Moves random points in random tangent directions.
    Accepts worse solutions with probability exp(-delta_r / T).
    """
    pts = pts_init.astype(float).copy()
    curr_r = _hull_radius(pts)
    best_pts = pts.copy()
    best_r = curr_r
    
    T = 0.1
    T_min = 0.001
    cooling = 0.95
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
            
        # Pick a random point
        i = rng.integers(0, N)
        
        # Random tangent direction
        random_vec = rng.normal(size=3)
        tangent_dir = random_vec - np.dot(random_vec, pts[i]) * pts[i]
        norm_t = np.linalg.norm(tangent_dir)
        if norm_t < _EPS:
            continue
        tangent_dir /= norm_t
        
        # Step size proportional to temperature
        step = 0.1 * T
        
        cand = pts.copy()
        cand[i] = cand[i] + step * tangent_dir
        cand[i] = cand[i] / (np.linalg.norm(cand[i]) + _EPS)
        
        new_r = _hull_radius(cand)
        delta_r = new_r - curr_r
        
        if delta_r < 0 or rng.random() < np.exp(-delta_r / T):
            pts = cand
            curr_r = new_r
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
        T *= cooling
        
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Initial point: Fibonacci spiral
    pts_init = _initial_fib()
    
    best_pts = pts_init.copy()
    best_r = _hull_radius(best_pts)
    
    # 2. Multi-restart with different strategies
    # Strategy A: Simulated Annealing from Fibonacci
    if time.time() < deadline - 10.0:
        pts_sa, r_sa = _simulated_annealing(pts_init, deadline - 10.0)
        if r_sa < best_r:
            best_r = r_sa
            best_pts = pts_sa.copy()
            
    # Strategy B: Simulated Annealing from Random
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        random_init = np.zeros((N, 3))
        for i in range(N):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(np.clip(1 - z**2, 0, None))
            random_init[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
            
        pts_sa2, r_sa2 = _simulated_annealing(random_init, deadline - 10.0)
        if r_sa2 < best_r:
            best_r = r_sa2
            best_pts = pts_sa2.copy()
            
    # Strategy C: Polish the best result found so far
    if time.time() < deadline - 2.0:
        pts_pol, r_pol = _least_squares_polish(best_pts, deadline)
        if r_pol < best_r:
            best_r = r_pol
            best_pts = pts_pol.copy()
            
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
    
    return best_pts
