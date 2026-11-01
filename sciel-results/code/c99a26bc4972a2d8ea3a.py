"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    # Min pair distance using vectorized approach
    min_pair = np.inf
    for i in range(n):
        diffs = points[i+1:] - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            min_pair = min(min_pair, np.min(dists))
    
    return min(min_pair / 2.0, min_boundary)

def _ring_packing(ring_radii, ring_counts, ring_offsets):
    """Generate points from ring parameters."""
    points = []
    for r, count, offset in zip(ring_radii, ring_counts, ring_offsets):
        if count == 0:
            continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.append(np.column_stack([x, y]))
    if points:
        return np.vstack(points)
    return np.array([])

def _optimize_rings(ring_radii_init, ring_counts, max_iter=100, deadline=None):
    """Optimize ring radii and offsets to maximize minimum distance."""
    n_rings = len(ring_radii_init)
    total_points = sum(ring_counts)
    
    # Parameters: [r_0, r_1, ..., r_{n-1}, offset_0, offset_1, ..., offset_{n-1}]
    x0 = np.concatenate([ring_radii_init, np.zeros(n_rings)])
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return np.inf
        radii = x[:n_rings]
        offsets = x[n_rings:]
        
        # Ensure radii are positive and ordered
        if np.any(radii < 0.01) or np.any(np.diff(radii) < 0.001):
            return np.inf
        
        points = _ring_packing(radii, ring_counts, offsets)
        if len(points) == 0:
            return np.inf
        
        # Score: negative of min(min_pair/2, min_boundary)
        score = _compute_score(points)
        return -score
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.success:
        radii = result.x[:n_rings]
        offsets = result.x[n_rings:]
        points = _ring_packing(radii, ring_counts, offsets)
        return points, _compute_score(points)
    
    return None, -np.inf

def _hex_lattice_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < N:
        return None
    
    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]

def _local_optimize(points, max_iter=200, lr=0.0005, deadline=None):
    """Local optimization using gradient-free approach to maximize minimum distance."""
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion (vectorized)
        for i in range(n):
            diffs = points[i] - points[i+1:]
            dists = np.linalg.norm(diffs, axis=1)
            # Avoid division by zero
            mask = dists > 1e-10
            if np.any(mask):
                forces_near = (0.15 - dists[mask]) / dists[mask] * diffs[mask]
                forces[i] += np.sum(forces_near, axis=0)
                # Add equal and opposite forces to the other points
                for j in range(i+1, n):
                    if dists[j-i-1] < 0.15:
                        forces[j] -= forces_near[j-i-1]
        
        # Attraction to center (to keep points inside)
        radii = np.linalg.norm(points, axis=1)
        mask_outer = radii > 0.95
        if np.any(mask_outer):
            forces[mask_outer] -= 0.1 * points[mask_outer] / radii[mask_outer, None]
        
        # Update positions
        points = points + lr * forces
        
        # Ensure points stay inside unit circle
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] = points[mask_out] * (1.0 / radii[mask_out, None])
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Adaptive learning rate
        lr *= 0.999
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Radial ring decomposition
    # Determine number of rings and points per ring
    # For N=300, try various ring configurations
    
    ring_configs = [
        # (num_rings, points_per_ring_list)
        (10, [1] + [30]*9),  # 1 center + 9 rings of 30
        (11, [1] + [29]*10),  # 1 center + 10 rings of 29
        (12, [1] + [27]*11),  # 1 center + 11 rings of 27
        (8, [1] + [38]*7),    # 1 center + 7 rings of 38
        (9, [1] + [34]*8),    # 1 center + 8 rings of 34
        (15, [1] + [20]*14),  # 1 center + 14 rings of 20
        (13, [1] + [23]*12),  # 1 center + 12 rings of 23
    ]
    
    for num_rings, ring_counts in ring_configs:
        if time.time() > deadline - 30:
            break
        
        # Initial radii: evenly spaced from small to near boundary
        r_max = 1.0 - 0.05  # Leave margin for circle radius
        ring_radii_init = np.linspace(0.05, r_max, num_rings)
        
        # Optimize ring configuration
        points, score = _optimize_rings(ring_radii_init, ring_counts, 
                                       max_iter=80, deadline=deadline - 25)
        
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 2: Hexagonal lattice (fallback and comparison)
    if time.time() < deadline - 30:
        for spacing_factor in [0.97, 0.99, 1.0, 1.02]:
            spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
            points = _hex_lattice_init(spacing)
            
            if points is not None:
                score = _compute_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
    
    # Local optimization on best result
    if best_points is not None and time.time() < deadline - 15:
        optimized = _local_optimize(best_points, max_iter=150, lr=0.0004, 
                                    deadline=deadline - 10)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Final validation
    if best_points is None or len(best_points) != N:
        # Emergency fallback: simple radial rings
        n_rings = 10
        ring_counts = [1] + [29]*9
        ring_radii = np.linspace(0.05, 0.95, n_rings)
        ring_offsets = np.zeros(n_rings)
        best_points = _ring_packing(ring_radii, ring_counts, ring_offsets)
        
        if len(best_points) < N:
            # Pad with additional points
            angles = np.linspace(0, 2*np.pi, N - len(best_points), endpoint=False)
            r = 0.5
            extra = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
            best_points = np.vstack([best_points, extra])
    
    return best_points[:N]
# EVOLVE-BLOCK-END
