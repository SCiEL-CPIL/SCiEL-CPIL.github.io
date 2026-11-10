"""Improved circle packing: hexagonal lattice + repulsive relaxation + local search."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
DEADLINE = time.time() + 130  # 130s safety margin before 150s kill


def _compute_radius(centers):
    """Compute the maximum common radius for a set of centers in [-0.5,0.5]^2."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Boundary distance
    dist_boundary = np.min(np.min(0.5 - np.abs(centers), axis=1))
    if dist_boundary <= 0:
        return 0.0
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0, output_type='ndarray')
    if len(pairs) == 0:
        return dist_boundary
    diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    min_pair = np.min(dists) / 2.0
    return min(dist_boundary, min_pair)


def _hex_lattice_init(n_circles):
    """Initialize with a hexagonal lattice pattern in [-0.5, 0.5]^2."""
    # Use hexagonal rows
    # Row spacing: dy, horizontal offset for alternating rows: dx/2
    # Try to fit as many as possible, then trim
    points = []
    # Estimate number of rows needed
    # For hexagonal packing with radius r, spacing = 2r
    # Let's try a range of spacings
    best_points = None
    best_count = 0
    
    for trial in range(30):
        # Vary the number of rows
        n_rows = 12 + trial
        dy = 1.0 / n_rows
        dx = dy / np.sqrt(3)
        n_cols = int(np.floor(1.0 / dx)) + 1
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + (row + 0.5) * dy
            offset = (dx / 2) if (row % 2 == 1) else 0.0
            for col in range(n_cols):
                x = -0.5 + (col + 0.5) * dx + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= n_circles:
            # Take a subset that maximizes minimum distance
            # Simple approach: take every k-th point or use a greedy selection
            if len(pts) == n_circles:
                if len(pts) > best_count:
                    best_count = len(pts)
                    best_points = pts
            else:
                # Use a greedy selection to pick n_circles points
                # Start with all, then iteratively remove the point with smallest min-distance
                remaining = list(range(len(pts)))
                while len(remaining) > n_circles:
                    min_d = np.inf
                    min_idx = -1
                    for i in remaining:
                        if i < len(pts):
                            d = np.min(np.linalg.norm(pts[remaining] - pts[i], axis=1))
                            if d < min_d:
                                min_d = d
                                min_idx = i
                    if min_idx >= 0:
                        remaining.remove(min_idx)
                selected = pts[remaining]
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_points = selected
    
    if best_points is None or len(best_points) < n_circles:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]
    
    return best_points[:n_circles]


def _repulsive_relaxation(centers, n_iter=200, learning_rate=0.01):
    """Relax circles using repulsive forces and boundary constraints."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(n_iter):
        if time.time() > DEADLINE:
            break
        # Compute repulsive forces
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r=0.2, output_type='ndarray')
        
        forces = np.zeros_like(centers)
        for i, j in pairs:
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random push
                d = np.array([np.random.randn(), np.random.randn()])
                dist = np.linalg.norm(d)
            # Repulsive force: stronger when closer
            f = d / (dist * dist) * 0.01
            forces[i] += f
            forces[j] -= f
        
        # Boundary forces: push inward if too close to edge
        for k in range(n):
            for axis in range(2):
                if centers[k, axis] > 0.45:
                    forces[k, axis] -= (centers[k, axis] - 0.45) * 0.5
                elif centers[k, axis] < -0.45:
                    forces[k, axis] += (-0.45 - centers[k, axis]) * 0.5
        
        # Update positions
        centers += learning_rate * forces
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decrease learning rate
        learning_rate *= 0.995
    
    return centers


def _local_search(centers, max_iters=5000):
    """Local hill climbing: randomly perturb a point or swap two nearby points."""
    centers = centers.copy()
    n = len(centers)
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Choose operation
        if np.random.random() < 0.7:
            # Perturb a single point
            i = np.random.randint(n)
            new_center = centers.copy()
            new_center[i] += np.random.randn(2) * 0.02
            new_center[i] = np.clip(new_center[i], -0.5, 0.5)
        else:
            # Swap two points
            i, j = np.random.choice(n, size=2, replace=False)
            new_center = centers.copy()
            new_center[i], new_center[j] = centers[j].copy(), centers[i].copy()
        
        new_r = _compute_radius(new_center)
        if new_r > current_r:
            centers = new_center
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    
    # Step 1: Hexagonal lattice initialization
    centers = _hex_lattice_init(N)
    
    # Step 2: Repulsive relaxation
    centers = _repulsive_relaxation(centers, n_iter=300)
    
    # Step 3: Local search
    centers, best_r = _local_search(centers, max_iters=3000)
    
    # Check if we have time for more iterations
    if time.time() < DEADLINE - 5:
        centers2, best_r2 = _local_search(centers, max_iters=2000)
        if best_r2 > best_r:
            centers = centers2
            best_r = best_r2
    
    # Final validation
    final_r = _compute_radius(centers)
    if final_r <= 0 or not np.all(np.isfinite(centers)):
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centers = np.array([(x, y) for y in ticks for x in ticks])
        centers = centers[:N]
    
    return centers
