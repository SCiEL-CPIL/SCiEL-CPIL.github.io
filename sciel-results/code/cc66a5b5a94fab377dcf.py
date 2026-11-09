"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def _score(pts):
    """Compute the radius: min over all pairs of half-distance and over all points of distance to boundary."""
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    # For each point, find nearest neighbor
    dists, _ = tree.query(pts, k=2)
    min_pair_dist = dists[:, 1].min()
    # Distance to boundary
    # Square is [-0.5, 0.5]^2
    dist_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_boundary_dist = dist_to_boundary.min()
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _hexagonal_init(n, seed=0):
    """Initialize n points on a hexagonal lattice within [-0.5, 0.5]^2."""
    rng = np.random.RandomState(seed)
    
    # Hexagonal lattice parameters
    # For a hexagonal packing, the spacing between rows is s * sqrt(3)/2
    # and horizontal spacing is s
    # We want to fit n points in a unit square
    # Approximate: n points in unit square => area per point ~ 1/n
    # Hexagonal packing area per point = s^2 * sqrt(3)/2
    # So s ~ sqrt(2/(n*sqrt(3)))
    
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    
    # Generate hexagonal lattice points
    points = []
    row = 0
    while len(points) < n + 50:  # Generate extra, then trim
        y = -0.5 + (row + 0.5) * s * np.sqrt(3) / 2
        if y > 0.5:
            break
        # Offset every other row
        offset = 0 if row % 2 == 0 else s / 2
        x = -0.5 + offset
        while x < 0.5:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += s
        row += 1
    
    points = np.array(points)
    
    # If we have more than n, randomly select n
    if len(points) > n:
        idx = rng.choice(len(points), size=n, replace=False)
        points = points[idx]
    
    # If we have fewer than n, add random points
    if len(points) < n:
        extra = rng.uniform(-0.4, 0.4, size=(n - len(points), 2))
        points = np.vstack([points, extra])
    
    return points[:n]


def _repulsive_relax(pts, max_iters=500, time_limit=100.0, start_time=None):
    """Apply repulsive force-based relaxation to improve packing."""
    if start_time is None:
        start_time = time.time()
    
    n = len(pts)
    pts = pts.copy()
    
    # Learning rate schedule
    lr = 0.01
    decay = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces: repulsion from all other points + attraction to boundary
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            # Vector from i to all others
            diffs = pts - pts[i]  # (n, 2)
            dists = np.linalg.norm(diffs, axis=1)
            # Avoid division by zero
            dists[i] = 1e-10
            # Repulsive force: 1/d^2 direction away
            # Only consider close neighbors for efficiency
            mask = dists < 0.15  # cutoff
            if np.any(mask):
                # Force magnitude ~ 1/d^2, direction away
                force_mags = 1.0 / (dists[mask] ** 2)
                directions = diffs[mask] / dists[mask, :1]
                forces[i] -= np.sum(force_mags[:, :1] * directions, axis=0)
        
        # Boundary attraction: push points away from walls
        # Distance to each wall
        dist_left = pts[:, 0] + 0.5
        dist_right = 0.5 - pts[:, 0]
        dist_bottom = pts[:, 1] + 0.5
        dist_top = 0.5 - pts[:, 1]
        
        # If too close to a wall, push inward
        wall_force = 0.05
        for i in range(n):
            if dist_left[i] < 0.05:
                forces[i, 0] += wall_force * (0.05 - dist_left[i]) / 0.05
            if dist_right[i] < 0.05:
                forces[i, 0] -= wall_force * (0.05 - dist_right[i]) / 0.05
            if dist_bottom[i] < 0.05:
                forces[i, 1] += wall_force * (0.05 - dist_bottom[i]) / 0.05
            if dist_top[i] < 0.05:
                forces[i, 1] -= wall_force * (0.05 - dist_top[i]) / 0.05
        
        # Update positions
        pts += lr * forces
        
        # Keep points within bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
        pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        # Decay learning rate
        lr *= decay
    
    return pts


def _gradient_ascent(pts, max_iters=3000, time_limit=100.0, start_time=None):
    """Gradient ascent on soft-min distance using log-sum-exp."""
    if start_time is None:
        start_time = time.time()
    
    n = len(pts)
    pts = pts.copy()
    
    # Soft-min parameter
    beta = 50.0  # Higher = sharper min approximation
    lr = 0.005
    decay = 0.9995
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all pairwise distances
        # Use efficient computation
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
        dists = np.linalg.norm(diffs, axis=2)  # (n, n)
        np.fill_diagonal(dists, 1e10)
        
        # Distance to boundary for each point
        dist_to_boundary = np.minimum(
            np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        )
        
        # Combine: for each point i, the limiting distance is min(nearest neighbor dist, boundary dist)
        # For soft-min, we use log-sum-exp on negative distances
        # Objective: maximize min over all constraints
        # Constraint values: for pair (i,j), d_ij/2; for point i, boundary_dist_i
        
        # Compute soft-min of all constraint values
        # All pair distances / 2
        pair_dists_half = dists / 2.0
        # Only upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        pair_vals = pair_dists_half[iu]
        
        # Boundary values
        boundary_vals = dist_to_boundary
        
        # Combine all values
        all_vals = np.concatenate([pair_vals, boundary_vals])
        
        # Soft-min via log-sum-exp: softmin(x) = -logsumexp(-x)
        # Gradient of softmin w.r.t. x_i = -softmax(-x)_i
        
        neg_vals = -all_vals
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        sum_exp = np.sum(exp_vals)
        softmax_vals = exp_vals / sum_exp  # These are the weights
        
        # Gradient of softmin w.r.t. each constraint value = -softmax_vals
        # But we need gradient w.r.t. pts
        
        # Reconstruct: for each pair (i,j), d_ij = |p_i - p_j|
        # d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
        # d(d_ij/2)/d(p_i) = (p_i - p_j) / (2 * d_ij)
        
        # For boundary: d(boundary_dist_i)/d(p_i) depends on which wall is closest
        
        # Build gradient
        grad = np.zeros_like(pts)
        
        # Pair contributions
        pair_grads = -softmax_vals[:len(pair_vals)]  # negative because softmin gradient
        # For each pair (i,j) with weight w, contribution to grad[i] = w * (p_i - p_j) / (2 * d_ij)
        # and grad[j] = w * (p_j - p_i) / (2 * d_ij)
        
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            if idx >= len(pair_grads):
                break
            w = pair_grads[idx]
            d = dists[i, j]
            if d > 1e-10:
                diff = (pts[i] - pts[j]) / (2.0 * d)
                grad[i] += w * diff
                grad[j] -= w * diff
        
        # Boundary contributions
        boundary_grads = -softmax_vals[len(pair_vals):]
        for i in range(n):
            w = boundary_grads[i]
            # Determine which wall is closest
            dl = pts[i, 0] + 0.5
            dr = 0.5 - pts[i, 0]
            db = pts[i, 1] + 0.5
            dt = 0.5 - pts[i, 1]
            
            min_d = min(dl, dr, db, dt)
            
            if min_d == dl:
                grad[i, 0] += w * (-1.0)  # d(dl)/dx = 1, but we want to increase dl => decrease x... wait
                # boundary_dist = min(dl, dr, db, dt)
                # If dl is min, boundary_dist = dl = x_i + 0.5
                # d(boundary_dist)/dx_i = 1
                # So grad += w * 1 in x direction
                grad[i, 0] = w * 1.0  # Replace, since only one wall is active
            elif min_d == dr:
                # dr = 0.5 - x_i, d/dr/dx_i = -1
                grad[i, 0] = w * (-1.0)
            elif min_d == db:
                grad[i, 1] = w * 1.0
            elif min_d == dt:
                grad[i, 1] = w * (-1.0)
        
        # Update
        pts += lr * grad
        
        # Keep within bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
        pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        lr *= decay
    
    return pts


def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s kill
    
    best_pts = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 5
    seeds = [42, 123, 456, 789, 1011]
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 20:
            break
        
        seed = seeds[r % len(seeds)]
        
        # Hexagonal initialization
        pts = _hexagonal_init(N, seed=seed)
        
        # Add small random perturbation to break symmetry
        rng = np.random.RandomState(seed)
        pts += rng.normal(0, 0.005, size=pts.shape)
        pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
        pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        # First: quick repulsive relaxation
        remaining = time_limit - (time.time() - start_time)
        relax_time = min(30.0, remaining * 0.3)
        if relax_time > 5:
            pts = _repulsive_relax(pts, max_iters=200, time_limit=relax_time, start_time=time.time())
        
        # Then: gradient ascent on soft-min
        remaining = time_limit - (time.time() - start_time)
        grad_time = min(remaining * 0.7, 80.0)
        if grad_time > 5:
            pts = _gradient_ascent(pts, max_iters=2000, time_limit=grad_time, start_time=time.time())
        
        # Final check
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
