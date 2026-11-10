"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

# Constants
_DEADLINE = 145.0  # Safety margin before 150s kill
_N_SAMPLES = 10000  # Number of sample points for covering radius estimation

def _normalize(pts):
    """Normalize points to unit sphere. Handles 1D and 2D inputs."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_samples(n_samples):
    """Generate sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _covering_radius(pts, samples):
    """
    Estimate covering radius by sampling the sphere densely.
    Returns the max angular distance (in degrees) from any sample to nearest point.
    Uses vectorized computation for speed.
    """
    # pts shape: (N, 3)
    # samples shape: (M, 3)
    
    # Compute dot products: (M, N)
    # To save memory, we can process in batches if M*N is large, but 10000*50 = 500k floats is fine.
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each sample, find the max dot product (closest point)
    max_dots = dots.max(axis=1)  # (M,)
    
    # The covering radius is the max angle, which corresponds to the min max_dot
    min_max_dot = np.min(max_dots)
    
    # Convert to degrees
    radius_rad = np.arccos(np.clip(min_max_dot, -1.0, 1.0))
    return np.degrees(radius_rad)

def _local_optimize(pts, samples, max_iters=2000, time_limit=None, start_time=None):
    """
    Local optimization: move points to reduce covering radius.
    Strategy: For each point, find the farthest sample point assigned to it (nearest in config),
    and move the config point towards that sample point.
    """
    if start_time is None:
        start_time = time.time()
        
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, samples)
    
    lr = 0.05  # Learning rate
    
    for it in range(max_iters):
        if time_limit is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        # Compute nearest configuration point for each sample
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        nearest_idx = dots.argmax(axis=1)  # (M,)
        
        # For each configuration point j, find the sample points assigned to it
        # and identify the "farthest" one (min dot product with j)
        new_pts = pts.copy()
        
        # Efficiently find the farthest sample for each point
        # We can do this by iterating over points, but vectorizing the selection is tricky.
        # Since N=50 is small, a loop over N is acceptable and clear.
        for j in range(N):
            mask = (nearest_idx == j)
            if not np.any(mask):
                continue
            
            # Get dots for samples assigned to j
            dots_j = dots[mask, j]
            
            # Find the sample with the minimum dot product (farthest angular distance)
            far_idx_in_mask = np.argmin(dots_j)
            
            # Get the actual sample point
            # We need the global index of this sample in the 'samples' array
            # We can precompute the global indices or just use np.where
            # For speed, let's just use the mask to get the points and pick the one with min dot
            assigned_samples = samples[mask]
            far_sample = assigned_samples[far_idx_in_mask]
            
            # Move point j towards far_sample
            # Gradient direction is roughly (far_sample - pts[j])
            diff = far_sample - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[j] = pts[j] + lr * diff / diff_norm
        
        new_pts = _normalize(new_pts)
        
        # Evaluate new radius
        new_radius = _covering_radius(new_pts, samples)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            # Slightly decay learning rate or keep it constant?
            # Keeping it constant allows for fine adjustments, but might oscillate.
            # Let's decay very slowly.
            lr = max(0.01, lr * 0.999)
        else:
            # If no improvement, reduce learning rate to escape local noise
            lr = max(0.005, lr * 0.9)
            
            # Optional: If stuck for a long time, break?
            # For now, just let it run until time limit.
    
    return best_pts, best_radius

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses Fibonacci spiral initialization + local optimization with multiple starts.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Generate fixed samples for evaluation
    samples = _generate_samples(_N_SAMPLES)
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts_init = _normalize(pts_init)
    
    best_pts = pts_init.copy()
    best_radius = _covering_radius(pts_init, samples)
    
    # Try multiple random perturbations to escape local minima
    rng = np.random.default_rng(42)
    
    # Number of attempts depends on remaining time
    # Each attempt takes some time for optimization.
    # Let's try at least 1 (the initial) and then as many as fit in the budget.
    
    n_attempts = 10 # Max attempts to try
    
    for attempt in range(n_attempts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:  # Leave 20s for final polish/return
            break
        
        if attempt == 0:
            current_pts = pts_init
        else:
            # Perturb the initial points or the best points?
            # Perturbing the best points can help refine, perturbing initial helps explore.
            # Let's perturb the initial Fibonacci spiral with noise.
            noise = rng.normal(0, 0.1, size=pts_init.shape)
            current_pts = _normalize(pts_init + noise)
        
        # Run local optimization
        # Allocate remaining time minus a safety margin for this attempt
        remaining = deadline - (time.time() - start_time)
        # Reserve some time for potential final polish
        time_limit = max(5.0, remaining - 5.0)
        
        opt_pts, opt_radius = _local_optimize(current_pts, samples, max_iters=5000, time_limit=time_limit, start_time=start_time)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
        
        # Check if we should continue
        if time.time() - start_time > deadline - 10:
            break
    
    # Final polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        remaining = deadline - elapsed - 2
        final_pts, final_radius = _local_optimize(best_pts, samples, max_iters=1000, time_limit=remaining, start_time=start_time)
        if final_radius < best_radius:
            best_pts = final_pts.copy()
            best_radius = final_radius
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
