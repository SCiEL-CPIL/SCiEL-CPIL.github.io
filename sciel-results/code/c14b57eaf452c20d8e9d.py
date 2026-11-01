import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n, offset=0.5):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _facet_radii(pts):
    """
    Compute the angular circumradius for each hull facet.
    Returns array of radii in radians.
    For each triangular facet, the circumcenter on the sphere is the point
    equidistant from all three vertices. If the circumcenter lies outside the
    spherical triangle, the covering radius for that region is determined by
    the edge with the largest angular length (radius = half that length).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([])
    
    radii = []
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere.
        # The circumcenter is along the normal to the plane containing v1, v2, v3.
        edge1 = v2 - v1
        edge2 = v3 - v1
        n = np.cross(edge1, edge2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        
        c = n / norm_n
        # Ensure c points away from origin (outward normal for hull containing origin)
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        
        # The angular distance from c to each vertex should be equal if c is the circumcenter.
        dots = np.array([np.dot(c, v1), np.dot(c, v2), np.dot(c, v3)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Check if the circumcenter is inside the spherical triangle.
        # A robust check: the circumcenter is inside if it can be expressed as
        # a positive linear combination of v1, v2, v3 (barycentric coordinates all positive).
        # Alternatively, check that the angles are all close to each other (within tolerance)
        # and that the circumcenter is on the same side of each edge as the opposite vertex.
        
        # Simpler check: if the max angle - min angle is small, the circumcenter is likely inside.
        # If not, the worst point is on an edge, and the radius is half the largest edge length.
        angle_spread = np.max(angles) - np.min(angles)
        
        if angle_spread < 1e-4:
            # Circumcenter is inside (or very close to the boundary)
            radius = angles[0]  # All angles should be approximately equal
        else:
            # Circumcenter is outside. The covering radius for this facet is
            # half the largest angular edge length.
            d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
            d13 = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
            radius = max(d12, d23, d13) / 2.0
        
        radii.append(radius)
    
    return np.array(radii) if radii else np.array([])

def _covering_radius_deg(config):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback: dense sampling
        grid = _fib_sample(50000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    radii = _facet_radii(config)
    if len(radii) == 0:
        return np.degrees(np.pi)
    
    return np.degrees(np.max(radii))

def _soft_max_objective(params, deadline, tau=0.01):
    """
    Smooth surrogate objective: soft-max of facet circumradii via log-sum-exp.
    """
    if time.time() > deadline - 10:
        return 1e10
    
    pts = params.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts = pts / norms
    
    radii = _facet_radii(pts)
    if len(radii) == 0:
        return 1e10
    
    scaled = radii / tau
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return tau * log_sum_exp

def _lloyd_step_vectorized(config, grid):
    """One step of Lloyd's algorithm, fully vectorized."""
    dots = grid @ config.T  # (len(grid), N)
    max_idx = np.argmax(dots, axis=1)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_config[i] = _normalize(center)
        else:
            # If a point has no assigned grid points, keep it in place
            new_config[i] = config[i]
    return new_config

def _optimize_from_start(config, deadline):
    """Full optimization pipeline from a starting configuration."""
    start_time = time.time()
    
    # Phase 1: Lloyd's algorithm with a grid
    grid = _fib_sample(20000)
    
    for it in range(100):
        if time.time() > deadline - 80:
            break
        new_config = _lloyd_step_vectorized(config, grid)
        if np.allclose(config, new_config, atol=1e-9):
            break
        config = new_config
    
    # Phase 2: L-BFGS-B optimization on the smooth surrogate objective
    if time.time() < deadline - 40:
        try:
            x0 = config.flatten()
            # Use multiple beta values for soft-max, starting smooth and getting sharper
            for tau in [0.05, 0.02, 0.01, 0.005]:
                if time.time() > deadline - 20:
                    break
                result = minimize(
                    _soft_max_objective,
                    x0,
                    args=(deadline, tau),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < _soft_max_objective(x0, deadline, tau):
                    lbfgs_pts = np.array([_normalize(p) for p in result.x.reshape(N, 3)])
                    lbfgs_radius = _covering_radius_deg(lbfgs_pts)
                    current_radius = _covering_radius_deg(config)
                    if lbfgs_radius < current_radius:
                        config = lbfgs_pts
                        x0 = config.flatten()
        except Exception:
            pass
    
    # Phase 3: Local refinement - perturb each point slightly and keep if it improves
    if time.time() < deadline - 15:
        current_radius = _covering_radius_deg(config)
        config = config.copy()
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Pick a random point to perturb
            idx = np.random.randint(N)
            v = config[idx]
            # Generate a random tangent direction
            rand_dir = np.random.randn(3)
            tangent = rand_dir - np.dot(rand_dir, v) * v
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent /= tangent_norm
            
            # Try a small step
            step = 0.01
            new_v = _normalize(v + step * tangent)
            config[idx] = new_v
            new_radius = _covering_radius_deg(config)
            
            if new_radius >= current_radius:
                # Revert
                config[idx] = v
            else:
                current_radius = new_radius
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds safety margin
    
    best_config = None
    best_radius = np.pi
    
    # Try multiple starting configurations
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            config = _fib_sample(N)
        elif restart == 1:
            config = _fib_sample(N, offset=0.3)
        elif restart == 2:
            config = _fib_sample(N, offset=0.7)
        else:
            # Random start with some structure
            config = _fib_sample(N)
            # Add small random perturbations
            for i in range(N):
                rand_dir = np.random.randn(3)
                tangent = rand_dir - np.dot(rand_dir, config[i]) * config[i]
                config[i] = _normalize(config[i] + 0.05 * tangent)
        
        config = _optimize_from_start(config, deadline)
        
        # Final normalization
        config = np.array([_normalize(p) for p in config])
        
        # Ensure no coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(config[i] - config[j]) < 1e-6:
                    config[j] = config[j] + 1e-4 * np.random.randn(3)
                    config[j] = _normalize(config[j])
        
        radius = _covering_radius_deg(config)
        if radius < best_radius:
            best_radius = radius
            best_config = config.copy()
    
    if best_config is None:
        best_config = _fib_sample(N)
        best_config = np.array([_normalize(p) for p in best_config])
    
    return best_config
