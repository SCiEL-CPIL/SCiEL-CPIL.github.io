"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses the convex hull facets to find the largest empty circle center.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point on the unit sphere equidistant from a, b, c
        # x · a = x · b = x · c, |x| = 1
        # x · (a - b) = 0, x · (a - c) = 0
        # So x is parallel to (a - b) × (a - c)
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Check both directions and take the one that gives a valid circumcenter
        # (the one closer to the vertices, i.e., smaller angular distance)
        dot_a_pos = np.clip(np.dot(x, a), -1.0, 1.0)
        r_pos = np.arccos(dot_a_pos)
        
        dot_a_neg = np.clip(np.dot(-x, a), -1.0, 1.0)
        r_neg = np.arccos(dot_a_neg)
        
        # The circumradius is the minimum of the two (the closer one)
        # But we need to check which one is actually the circumcenter of the cap
        # The circumcenter should be on the same side as the facet normal
        n = np.cross(b - a, c - a)
        if np.dot(x, n) < 0:
            x = -x
            r = r_neg
        else:
            r = r_pos
        
        max_radius = max(max_radius, r)
    
    # Also check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        max_radius = max(max_radius, min_dist)
    
    return max_radius


def _covering_radius_fallback(pts, n_samples=50000):
    """Fallback covering radius computation via dense sampling."""
    pts = _to_sphere(pts)
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    dots = samples @ pts.T
    min_dots = np.max(dots, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    max_ang = np.arccos(min_dots)
    return np.max(max_ang)


def _objective(pts):
    """The score to minimize: covering radius in degrees."""
    return np.degrees(_covering_radius(pts))


def _repulsion_force(pts, k=5):
    """
    Compute repulsion forces between points.
    Points repel each other with a force that increases as they get closer.
    This is a structural change: using a physical simulation approach.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # Repulsion force: stronger when closer
            force_mag = 1.0 / (dist ** k)
            force = force_mag * diff / dist
            forces[i] += force
            forces[j] -= force
    
    return forces


def _local_optimize(pts, time_limit=30.0):
    """
    Local optimization using repulsion forces and gradient descent.
    This is a different optimization method: instead of random perturbations,
    we use a physical simulation where points repel each other.
    """
    start_time = time.time()
    pts = _to_sphere(pts)
    current_score = _objective(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    
    while time.time() - start_time < time_limit:
        # Compute repulsion forces
        forces = _repulsion_force(pts)
        
        # Add a small random perturbation to escape local minima
        noise = 0.001 * np.random.randn(*pts.shape)
        forces += noise
        
        # Update points
        momentum = 0.9 * momentum + 0.1 * forces
        pts = pts + lr * momentum
        pts = _to_sphere(pts)
        
        new_score = _objective(pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.999
        
        # If stuck, try a larger perturbation
        if time.time() - start_time > time_limit * 0.8:
            idx = np.random.randint(N)
            pts[idx] += 0.1 * np.random.randn(3)
            pts = _to_sphere(pts)
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_score = _objective(pts)
    best_pts = pts.copy()
    
    # Phase 1: Repulsion-based local optimization
    if time.time() < deadline - 60.0:
        pts, score = _local_optimize(pts, time_limit=40.0)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing with random perturbations
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(12345)
        current_pts = best_pts.copy()
        current_score = best_score
        temp = 0.05
        
        while time.time() < deadline - 5.0:
            idx = rng.integers(N)
            delta = rng.normal(scale=0.03, size=3)
            new_pts = current_pts.copy()
            new_pts[idx] = current_pts[idx] + delta
            new_pts = _to_sphere(new_pts)
            
            new_score = _objective(new_pts)
            delta_e = new_score - current_score
            
            # Accept if better, or with probability exp(-delta_e / temp)
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                current_pts = new_pts
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            
            # Cool down
            temp *= 0.995
            if temp < 0.001:
                temp = 0.001
    
    return best_pts
# EVOLVE-BLOCK-END
