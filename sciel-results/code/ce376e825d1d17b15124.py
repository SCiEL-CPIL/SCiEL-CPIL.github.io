"""Pack 300 equal circles in the unit disk.
Strategy: hexagonal lattice initialization + local optimization with O(N) cost per step."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _score(points):
    """Compute the packing score (minimum of pair distances/2 and boundary margins)."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    # Check nearest neighbors for minimum pair distance
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min() / 2.0
    # Check boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = (1.0 - radii).min()
    return min(min_pair_dist, min_boundary)


def _local_optimize(points, time_budget=100.0):
    """Local optimization using gradient-free coordinate descent with adaptive step size.
    Uses O(N) per step by only checking nearest neighbors."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    tree = cKDTree(points)
    
    # Adaptive parameters
    step_size = 0.01
    min_step = 1e-7
    patience = 50  # steps without improvement before reducing step
    no_improve = 0
    start_time = time.time()
    
    # Build neighbor list (k nearest neighbors)
    k_neighbors = 8
    dists, indices = tree.query(points, k=k_neighbors + 1)
    # indices[i, 0] is i itself
    
    while time.time() - start_time < time_budget:
        # Pick a random point to move
        i = np.random.randint(n)
        
        # Compute forces from nearest neighbors and boundary
        force = np.zeros(2)
        
        # Boundary force: push inward if too close to edge
        r_i = np.hypot(points[i, 0], points[i, 1])
        if r_i > 0.0:
            boundary_margin = 1.0 - r_i
            if boundary_margin < 0.05:
                # Push toward center
                force -= points[i] / r_i * (0.05 - boundary_margin) * 2.0
        
        # Neighbor forces: push away from neighbors that are too close
        for j in indices[i, 1:]:
            if j == i:
                continue
            diff = points[i] - points[j]
            d = np.hypot(diff[0], diff[1])
            if d < 0.02:  # Only interact with close neighbors
                # Desired distance is 2*r, but we don't know r exactly
                # Use a target based on current best score
                target = 2.0 * best_score
                if d < target:
                    push = (target - d) / max(d, 1e-10)
                    force += diff * push
        
        # Apply move
        norm_f = np.hypot(force[0], force[1])
        if norm_f > 1e-12:
            move = force / norm_f * step_size
            new_point = points[i] + move
            
            # Check if still inside unit circle
            if np.hypot(new_point[0], new_point[1]) < 1.0:
                old_point = points[i].copy()
                points[i] = new_point
                
                # Quick local score check
                # Check distance to neighbors
                valid = True
                for j in indices[i, 1:]:
                    if j == i:
                        continue
                    d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
                    if d < 2.0 * best_score * 0.99:
                        valid = False
                        break
                
                if valid:
                    # Recompute score periodically
                    if no_improve % 10 == 0:
                        new_score = _score(points)
                        if new_score > best_score:
                            best_score = new_score
                            no_improve = 0
                            # Update tree
                            tree = cKDTree(points)
                            dists, indices = tree.query(points, k=k_neighbors + 1)
                        else:
                            no_improve += 1
                            # Revert if worse
                            points[i] = old_point
                    else:
                        no_improve += 1
                else:
                    points[i] = old_point
                    no_improve += 1
            else:
                no_improve += 1
        
        # Adaptive step size
        if no_improve > patience:
            step_size *= 0.7
            no_improve = 0
            if step_size < min_step:
                break
        
        # Occasionally recompute full score
        if no_improve % 50 == 0:
            current_score = _score(points)
            if current_score > best_score:
                best_score = current_score
            # Rebuild tree periodically
            tree = cKDTree(points)
            dists, indices = tree.query(points, k=k_neighbors + 1)
    
    return points, best_score


def _hex_init():
    """Generate hexagonal lattice points inside unit disk."""
    # Try different spacings
    best_points = None
    best_spacing = 0.03
    
    for spacing in np.linspace(0.025, 0.045, 20):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        
        points = np.array(rows)
        if len(points) >= N:
            # Select N points closest to center for better packing
            radii = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(radii, kind='stable')[:N]]
            score = _score(points)
            if best_points is None or score > best_spacing:
                best_points = points
                best_spacing = score
    
    if best_points is None:
        # Fallback: uniform random
        while True:
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            if _score(points) > 0:
                return points
    
    return best_points


def construct():
    start_time = time.time()
    
    # Initialize with hexagonal lattice
    points = _hex_init()
    best_score = _score(points)
    
    # Local optimization
    points, best_score = _local_optimize(points, time_budget=DEADLINE - (time.time() - start_time) - 5)
    
    # Final polish: try to improve with smaller steps
    if time.time() - start_time < DEADLINE - 10:
        points, score2 = _local_optimize(points, time_budget=min(20, DEADLINE - (time.time() - start_time) - 5))
        if score2 > best_score:
            best_score = score2
    
    # Ensure validity
    points = np.array(points, dtype=np.float64)
    
    # Final check: ensure all points are strictly inside and no overlaps
    radii = np.hypot(points[:, 0], points[:, 1])
    if (radii >= 1.0).any():
        # Project back inside
        for i in range(len(points)):
            if radii[i] >= 1.0:
                points[i] *= 0.999 / radii[i]
    
    return points
