# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses spherical constraint with high-power repulsive forces and
    simulated annealing for escape from local minima.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0  # seconds
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def repulsion_forces(points, power=10.0):
        """
        Compute repulsive forces between all pairs.
        Force magnitude ~ 1/d^power, direction away from other point.
        """
        n_pts = len(points)
        forces = np.zeros_like(points)
        
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-15:
                    # Add small random perturbation
                    diff = np.random.randn(3) * 1e-8
                    dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)
                # Force magnitude: 1/d^power
                force_mag = 1.0 / (dist ** power)
                force_vec = force_mag * diff / dist
                forces[i] += force_vec
                forces[j] -= force_vec
        
        return forces
    
    def relax(points, n_steps=5000, power=10.0, cooling=None):
        """
        Relax points on sphere using repulsive forces with optional
        simulated annealing (random perturbations that decrease over time).
        """
        pts = points.copy()
        rng = np.random.default_rng(42)
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute forces
            forces = repulsion_forces(pts, power)
            
            # Add thermal noise if cooling is provided
            if cooling is not None:
                temp = cooling[0] * (cooling[1] ** step)
                noise = rng.standard_normal((n, d)) * temp
                forces += noise
            
            # Update positions (gradient ascent on repulsive potential)
            lr = 0.01
            pts += lr * forces
            
            # Project back to sphere
            pts = normalize_sphere(pts)
        
        return pts
    
    def objective_sphere(points):
        """Score for points on unit sphere: d_min^2 (since d_max is fixed ~ sqrt(2))."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    # Initialization strategies
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def symmetric_start():
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron = normalize_sphere(icosahedron)
        
        p13 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p14 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        
        return np.vstack([icosahedron, [p13, p14]])
    
    best_points = None
    best_score = -1.0
    
    # Try multiple initializations with different relaxation parameters
    init_configs = [
        fibonacci_sphere(n),
        symmetric_start(),
    ]
    
    for seed in range(30):
        init_configs.append(random_sphere(n, seed))
    
    for init_idx, init_pts in enumerate(init_configs):
        if check_deadline():
            break
        
        # Try different power laws and annealing schedules
        for power in [6.0, 10.0, 15.0, 20.0]:
            if check_deadline():
                break
            
            # Without annealing
            relaxed = relax(init_pts, n_steps=3000, power=power)
            score = objective_sphere(relaxed)
            
            if score > best_score:
                best_score = score
                best_points = relaxed.copy()
            
            # With simulated annealing (high to low temperature)
            cooling = (0.05, 0.999)  # start_temp, decay
            relaxed_sa = relax(init_pts, n_steps=5000, power=power, cooling=cooling)
            score_sa = objective_sphere(relaxed_sa)
            
            if score_sa > best_score:
                best_score = score_sa
                best_points = relaxed_sa.copy()
        
        # If we've found a good score, do more refinement
        if best_score > 0.45 and init_idx % 3 == 0:
            if check_deadline():
                break
            # Long relaxation with moderate power
            refined = relax(best_points, n_steps=10000, power=12.0)
            score_ref = objective_sphere(refined)
            if score_ref > best_score:
                best_score = score_ref
                best_points = refined.copy()
    
    # Final refinement: multiple passes with decreasing power
    if best_points is not None and not check_deadline():
        for power in [8.0, 10.0, 12.0]:
            if check_deadline():
                break
            refined = relax(best_points, n_steps=5000, power=power)
            score = objective_sphere(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
