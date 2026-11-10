"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
_DEADLINE = 120.0  # seconds, with margin before 150s kill


def _normalize(pts):
    """Normalize points to unit sphere, handling 1D and 2D inputs."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        n = np.linalg.norm(pts)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / n
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1e-12
    pts[mask] = pts[mask] / norms[mask, None]
    # Replace any zero-norm rows with a default direction
    if np.any(~mask):
        pts[~mask] = np.array([1.0, 0.0, 0.0])
    return pts


def _covering_radius(pts, n_samples=8000):
    """
    Estimate covering radius by sampling the sphere densely.
    Returns the max angular distance (in degrees) from any sample to nearest point.
    Uses vectorized computation for speed.
    """
    pts = _normalize(pts)
    # Generate sample points on sphere using Fibonacci spiral
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances: cos(angle) = dot product
    # For each sample, find min distance to any point
    # Batch computation to avoid memory issues
    batch_size = 2000
    max_cos = -2.0  # will be max of min cos = min angle
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        s = samples[start:end]  # (B, 3)
        # dot products: (B, N)
        dots = s @ pts.T
        # clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # min distance = max cos (closest point)
        min_cos = dots.max(axis=1)  # (B,)
        # angle = arccos(cos), we want max angle = min cos... 
        # Actually: angular distance = arccos(dot). Min distance to nearest = min over j of arccos(dot_ij)
        # = arccos(max over j of dot_ij)
        # So covering radius = max over i of arccos(max_j dot_ij)
        # = arccos(min over i of max_j dot_ij)
        # So we track min over i of (max_j dot_ij)
        batch_min_cos = min_cos.min()
        if batch_min_cos > max_cos:
            max_cos = batch_min_cos
    
    # covering radius in radians
    radius_rad = np.arccos(np.clip(max_cos, -1.0, 1.0))
    return np.degrees(radius_rad)


def _local_optimize(pts, max_iters=5000, tol=1e-8, time_limit=None):
    """
    Local optimization: move points to reduce covering radius.
    Uses subgradient-like approach: for each point, compute which region it covers,
    and move it toward the center of its Voronoi cell (in angular sense).
    """
    pts = _normalize(pts).copy()
    start_time = time.time()
    
    # Precompute a set of "test points" to evaluate covering radius efficiently
    # Use Fibonacci spiral for test points
    n_test = 5000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, n_samples=n_test)
    
    lr = 0.05  # learning rate
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # For each test point, find nearest configuration point
        # dots: (n_test, N)
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        nearest_idx = dots.argmax(axis=1)  # (n_test,)
        
        # For each configuration point, find the test points assigned to it
        # and compute the "farthest" among them - that's the bottleneck
        # Move each point toward the centroid of its farthest assigned points
        
        new_pts = pts.copy()
        for j in range(N):
            assigned = (nearest_idx == j)
            if not np.any(assigned):
                continue
            assigned_pts = test_pts[assigned]
            # Find the farthest assigned point (max angular distance = min dot)
            dots_j = dots[assigned, j]
            # The farthest has min dot product
            far_idx = np.argmin(dots_j)
            far_point = assigned_pts[far_idx]
            
            # Move point j toward far_point
            # Gradient of covering radius w.r.t. point j points toward far_point
            # Update: p_j += lr * (far_point - p_j) / ||far_point - p_j||
            diff = far_point - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[j] = pts[j] + lr * diff / diff_norm
        
        new_pts = _normalize(new_pts)
        new_radius = _covering_radius(new_pts, n_samples=n_test)
        
        if new_radius < best_radius - tol:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Decay learning rate slowly
            lr = max(0.01, lr * 0.999)
        else:
            # Try smaller step
            lr = max(0.005, lr * 0.9)
    
    return best_pts


def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses Fibonacci spiral initialization + local optimization.
    """
    start_time = time.time()
    deadline = _DEADLINE - 5.0  # safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Try multiple random perturbations and pick the best
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, n_samples=5000)
    
    rng = np.random.default_rng(42)
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() - start_time > deadline:
            break
        
        # Perturb initial points
        noise = rng.normal(0, 0.1, size=pts.shape)
        perturbed = _normalize(pts + noise)
        
        # Optimize
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        opt_pts = _local_optimize(perturbed, max_iters=2000, time_limit=remaining - 5)
        opt_radius = _covering_radius(opt_pts, n_samples=5000)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Final optimization on best result if time permits
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        final_pts = _local_optimize(best_pts, max_iters=3000, time_limit=remaining - 5)
        final_radius = _covering_radius(final_pts, n_samples=5000)
        if final_radius < best_radius:
            best_pts = final_pts.copy()
            best_radius = final_radius
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
