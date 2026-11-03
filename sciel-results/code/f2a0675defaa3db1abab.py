"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds: centers must be inside [-0.5, 0.5]^2
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_wall = np.min(wall_dists)
    return min(min_pair, 2 * min_wall) / 2.0

def _grid_init():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Create a hexagonal lattice initialization."""
    pts = []
    # Hexagonal packing in a square
    m = 8
    for row in range(m):
        for col in range(m):
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            if row % 2 == 1:
                x += 0.5 / m
            pts.append([x, y])
    pts = np.array(pts[:N])
    # Scale to fit
    max_coord = np.max(np.abs(pts))
    if max_coord > 0.45:
        pts = pts * (0.45 / max_coord)
    return pts

def _force_relax(pts, n_iter=500, dt=0.001):
    """Force-directed relaxation to spread points apart."""
    pts = pts.copy()
    for _ in range(n_iter):
        # Compute repulsive forces between all pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Repulsion: force proportional to 1/d^2 along the direction
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dists[i, j]
                    if d < 0.1:
                        f = 0.01 / (d * d)
                        force[i] += f * diff[i, j] / d
        
        # Attraction to center to keep points inside
        for i in range(N):
            force[i] -= 0.001 * pts[i]
        
        # Wall repulsion
        for i in range(N):
            for axis in range(2):
                if pts[i, axis] > 0.4:
                    force[i, axis] -= 0.1 * (pts[i, axis] - 0.4)
                elif pts[i, axis] < -0.4:
                    force[i, axis] -= 0.1 * (pts[i, axis] + 0.4)
        
        pts += dt * force
        # Clamp to bounds
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def _optimize_sa(pts, t_start=0.05, t_end=0.001, n_iter=20000, deadline=None):
    """Simulated annealing optimization."""
    best_pts = pts.copy()
    best_score = _score(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation
        new_pts = current_pts.copy()
        idx = np.random.randint(N)
        step = t_start * (t_end / t_start) ** (i / n_iter)
        new_pts[idx] += np.random.normal(0, step, 2)
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        else:
            temp = t_start * (t_end / t_start) ** (i / n_iter)
            if temp > 1e-10:
                prob = np.exp((new_score - current_score) / temp)
                if np.random.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
    
    return best_pts, best_score

def _local_refine(pts, n_iter=1000, deadline=None):
    """Local refinement using gradient-like steps."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            for axis in range(2):
                # Try small perturbations
                for delta in [0.001, -0.001, 0.0005, -0.0005]:
                    new_pts = best_pts.copy()
                    new_pts[idx, axis] += delta
                    if -0.5 <= new_pts[idx, axis] <= 0.5:
                        new_score = _score(new_pts)
                        if new_score > best_score + 1e-12:
                            best_pts = new_pts
                            best_score = new_score
                            improved = True
        
        if not improved:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Try multiple initializations
    inits = [_grid_init(), _hex_init()]
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force relaxation
        relaxed = _force_relax(init, n_iter=200)
        if time.time() > deadline:
            break
        
        # Simulated annealing
        sa_pts, sa_score = _optimize_sa(relaxed, t_start=0.03, t_end=0.0005, n_iter=15000, deadline=deadline)
        if sa_score > best_score:
            best_pts = sa_pts
            best_score = sa_score
        
        # Local refinement
        if time.time() < deadline - 10.0:
            ref_pts, ref_score = _local_refine(sa_pts, n_iter=500, deadline=deadline)
            if ref_score > best_score:
                best_pts = ref_pts
                best_score = ref_score
    
    # If no valid result, fall back to grid
    if best_pts is None:
        return _grid_init()
    
    return best_pts
# EVOLVE-BLOCK-END
