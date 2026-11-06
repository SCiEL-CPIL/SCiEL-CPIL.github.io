"""Circle packing: 200 equal circles in unit square [-0.5, 0.5]^2."""
import numpy as np
import time

N = 200
DEADLINE = 150.0  # wall-clock seconds; evaluator kills at 150s
SAFETY_MARGIN = 5.0  # return at least this many seconds before deadline

def _compute_radius(centers):
    """Compute the radius: min over pairs of dist/2, and min over boundary distances."""
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Boundary distances
    boundary = np.min(np.min(0.5 - np.abs(centers), axis=1), axis=0)
    return min(min_pair, boundary)

def _hex_init(n, rng):
    """Generate n points on a hexagonal lattice, randomly placed/rotated/scaled."""
    # Hex lattice basis
    a = 1.0  # lattice spacing (will be scaled)
    b1 = np.array([a, 0.0])
    b2 = np.array([a * 0.5, a * np.sqrt(3) / 2.0])
    
    # Generate enough lattice points
    extent = 3
    pts = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            p = i * b1 + j * b2
            if abs(p[0]) <= 1.0 and abs(p[1]) <= 1.0:
                pts.append(p)
    pts = np.array(pts)
    
    # Random rotation
    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    # Random translation
    tx = rng.uniform(-0.3, 0.3)
    ty = rng.uniform(-0.3, 0.3)
    pts = pts + np.array([tx, ty])
    
    # Random scale
    scale = rng.uniform(0.5, 1.5)
    pts = pts * scale
    
    # Filter to square
    mask = (np.abs(pts[:, 0]) <= 0.45) & (np.abs(pts[:, 1]) <= 0.45)
    pts = pts[mask]
    
    if len(pts) < n:
        # Fill with random points
        extra = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    # Shuffle and take n
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]

def _optimize(centers, n_iter=5000, rng=None):
    """Vectorized force-based optimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Adaptive parameters
    step = 0.01
    force_power = 2.0  # repulsion exponent
    boundary_force = 1.0
    
    for it in range(n_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Compute forces
        # Pairwise repulsion
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Force: repel if too close, target separation ~ 2r
        # Use a smooth force: F = (target - dist) / dist * direction
        # Actually, use gradient of min distance: push apart closest pairs
        # Better: force proportional to 1/dist^force_power for all pairs
        force = np.zeros_like(centers)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = dist[i, j]
                if d < 1e-10:
                    # Random nudge
                    force[i] += rng.normal(size=2) * 0.01
                    force[j] -= rng.normal(size=2) * 0.01
                    continue
                # Repulsive force
                f = 1.0 / (d ** force_power)
                force[i] += f * diff[i, j] / d
                force[j] -= f * diff[i, j] / d
        
        # Boundary forces: push inward if close to edge
        for i in range(len(centers)):
            for axis in range(2):
                if centers[i, axis] > 0.45:
                    force[i, axis] -= boundary_force * (centers[i, axis] - 0.45) ** 2
                elif centers[i, axis] < -0.45:
                    force[i, axis] += boundary_force * (-0.45 - centers[i, axis]) ** 2
        
        # Normalize and apply
        force_norm = np.linalg.norm(force, axis=1, keepdims=True)
        force_norm = np.maximum(force_norm, 1e-10)
        force = force / force_norm * step
        
        # Adaptive step: reduce if radius decreased
        new_centers = centers + force
        # Clamp to square
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        if new_radius > best_radius:
            centers = new_centers
            best_radius = new_radius
            step = min(step * 1.01, 0.05)
        else:
            step = max(step * 0.95, 0.001)
        
        # Occasionally do a larger kick
        if it % 500 == 0 and step < 0.01:
            kick = rng.normal(size=centers.shape) * 0.05
            centers = centers + kick
            centers = np.clip(centers, -0.499, 0.499)
    
    return centers, best_radius

def _optimize_vectorized(centers, n_iter=10000, rng=None):
    """Fully vectorized force-based optimization - much faster."""
    if rng is None:
        rng = np.random.default_rng()
    
    centers = centers.copy()
    n = len(centers)
    best_radius = _compute_radius(centers)
    
    step = 0.005
    decay = 0.9995
    
    for it in range(n_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Vectorized pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Force: repulsive, 1/dist^2
        # F_ij = diff_ij / dist_ij^3  (since d(1/dist)/d(dist) = -1/dist^2, direction is diff/dist)
        # Actually force on i from j: direction is (c_i - c_j), magnitude 1/dist^2
        # So F_i = sum_j (c_i - c_j) / dist_ij^3
        inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
        inv_dist_cubed = inv_dist ** 3
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        force = np.sum(diff * inv_dist_cubed[:, :, None], axis=1)
        
        # Boundary forces (vectorized)
        margin = 0.48
        for axis in range(2):
            over = np.maximum(0, centers[:, axis] - margin)
            under = np.maximum(0, -margin - centers[:, axis])
            force[:, axis] -= over * 10.0
            force[:, axis] += under * 10.0
        
        # Normalize
        force_norm = np.linalg.norm(force, axis=1, keepdims=True)
        force_norm = np.maximum(force_norm, 1e-10)
        force = force / force_norm * step
        
        new_centers = centers + force
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        if new_radius >= best_radius:
            centers = new_centers
            best_radius = new_radius
        else:
            # Partial step
            new_centers = centers + force * 0.5
            new_centers = np.clip(new_centers, -0.499, 0.499)
            new_radius = _compute_radius(new_centers)
            if new_radius >= best_radius:
                centers = new_centers
                best_radius = new_radius
        
        step *= decay
        if step < 1e-6:
            break
        
        # Periodic kick
        if it % 2000 == 1000:
            kick = rng.normal(size=centers.shape) * 0.03
            centers = centers + kick
            centers = np.clip(centers, -0.499, 0.499)
            step = 0.003
    
    return centers, best_radius

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Hex initialization
        centers = _hex_init(N, rng)
        
        # Optimize
        centers, radius = _optimize_vectorized(centers, n_iter=8000, rng=rng)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # If we have good time, do more restarts
        elapsed = time.time() - start_time
        if elapsed > DEADLINE * 0.7:
            break
    
    # Final refinement on best
    if best_centers is not None and time.time() < DEADLINE - SAFETY_MARGIN:
        best_centers, best_radius = _optimize_vectorized(best_centers, n_iter=10000, rng=rng)
    
    if best_centers is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
