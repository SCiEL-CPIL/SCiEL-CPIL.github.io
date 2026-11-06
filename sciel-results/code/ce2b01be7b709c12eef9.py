"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math


def construct():
    """
    Pack N equal circles in unit disk using concentric ring structure with optimization.
    """
    # Step 1: Initialize with concentric rings
    # Estimate number of rings and circles per ring
    # For N=300, typical optimal has ~10-15 rings
    # Start with a reasonable guess
    
    # Determine ring structure: assign circles to rings
    # Use a heuristic: outer rings have more circles
    ring_counts = _estimate_ring_counts(N)
    num_rings = len(ring_counts)
    
    # Initial ring radii: spread from center to near boundary
    # Leave room for circle radius (unknown initially, estimate ~0.05)
    r_est = 0.05
    max_r = 1.0 - r_est
    
    # Initial radii: roughly evenly spaced in area
    total_area = math.pi * max_r**2
    area_per_circle = total_area / N
    # Ring i has circumference 2*pi*r_i, so circles per ring ~ 2*pi*r_i / (2*r_est) = pi*r_i/r_est
    # We'll just use a reasonable initial spread
    initial_radii = np.linspace(0.0, max_r, num_rings + 1)[1:]
    
    # Initial angles: evenly spaced per ring
    angles = []
    for i, count in enumerate(ring_counts):
        if count > 0:
            angles.append(np.arange(count) * 2 * np.pi / count + i * 0.1)  # slight offset
        else:
            angles.append(np.array([]))
    
    # Build initial positions
    positions = _build_positions_from_rings(initial_radii, angles, ring_counts)
    
    # Step 2: Optimize using differential evolution on ring parameters
    # Parameters: ring radii (num_rings) + angular offsets (num_rings)
    # Actually, let's optimize the full set of positions but with a smart initialization
    
    # Better approach: optimize ring radii and per-ring angular shifts
    def objective(params):
        """Minimize negative of the minimum circle radius (i.e., maximize it)"""
        ring_radii = params[:num_rings]
        ring_offsets = params[num_rings:]
        
        # Ensure ring radii are sorted and within bounds
        ring_radii = np.clip(ring_radii, 0.0, 1.0 - 0.01)
        
        # Build positions
        pos = _build_positions_from_rings(ring_radii, angles, ring_counts)
        
        # Compute the achievable radius
        r = _compute_min_radius(pos)
        
        # If invalid (negative or too small), return large penalty
        if r <= 0 or not np.isfinite(r):
            return 1e10
        
        return -r
    
    # Bounds for optimization
    bounds = []
    for i in range(num_rings):
        bounds.append((0.0, 1.0 - 0.01))  # ring radii
        bounds.append((0.0, 2 * np.pi))  # angular offsets
    
    # Run differential evolution
    x0 = np.concatenate([initial_radii, np.zeros(num_rings)])
    
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=50,
        tol=1e-8,
        popsize=15,
        polish=True
    )
    
    # Extract optimized parameters
    opt_ring_radii = result.x[:num_rings]
    opt_ring_offsets = result.x[num_rings:]
    
    # Apply angular offsets
    angles_opt = []
    for i, count in enumerate(ring_counts):
        if count > 0:
            angles_opt.append(angles[i] + opt_ring_offsets[i])
        else:
            angles_opt.append(np.array([]))
    
    # Build final positions
    positions = _build_positions_from_rings(opt_ring_radii, angles_opt, ring_counts)
    
    # Step 3: Local refinement - push circles apart and into container
    positions = _local_refinement(positions, max_iter=200)
    
    # Ensure valid: clip to unit disk
    radii = np.hypot(positions[:, 0], positions[:, 1])
    mask = radii <= 1.0
    if np.sum(mask) < N:
        # Fallback: use initial positions
        positions = _build_positions_from_rings(initial_radii, angles, ring_counts)
        radii = np.hypot(positions[:, 0], positions[:, 1])
        positions = positions[radii <= 1.0]
        if len(positions) < N:
            # Last resort: hexagonal grid
            return _hexagonal_fallback(N)
    
    return positions[:N]


def _estimate_ring_counts(N):
    """
    Estimate how many circles to put in each ring.
    Outer rings should have more circles.
    """
    # Try different numbers of rings
    best_counts = None
    best_score = -1
    
    for num_rings in range(8, 20):
        # Distribute circles: outer rings get more
        # Use a power law: count_i ~ (i+1)^p
        # Total should be N
        for p in [1.0, 1.5, 2.0]:
            weights = np.array([(i + 1) ** p for i in range(num_rings)])
            weights = weights / weights.sum()
            counts = np.round(weights * N).astype(int)
            
            # Adjust to sum to N
            diff = N - counts.sum()
            if diff > 0:
                # Add to outer rings
                for i in range(num_rings - 1, -1, -1):
                    if diff <= 0:
                        break
                    counts[i] += 1
                    diff -= 1
            elif diff < 0:
                # Remove from inner rings
                for i in range(0, num_rings):
                    if diff >= 0:
                        break
                    if counts[i] > 1:
                        counts[i] -= 1
                        diff += 1
            
            # Score: prefer balanced distribution
            score = -np.var(counts)
            if score > best_score:
                best_score = score
                best_counts = counts.tolist()
    
    if best_counts is None:
        # Fallback: simple distribution
        base = N // 12
        remainder = N % 12
        best_counts = [base + (1 if i < remainder else 0) for i in range(12)]
    
    return best_counts


def _build_positions_from_rings(ring_radii, angles_list, ring_counts):
    """Build 2D positions from ring radii and angles."""
    positions = []
    for i, (r, angles) in enumerate(zip(ring_radii, angles_list)):
        if ring_counts[i] > 0 and len(angles) > 0:
            for angle in angles:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                positions.append([x, y])
    return np.array(positions)


def _compute_min_radius(positions):
    """Compute the maximum common radius for the given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Minimum distance to container boundary
    dist_to_boundary = 1.0 - np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = np.min(dist_to_boundary)
    
    if min_boundary < 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    # For efficiency, use a grid-based approach for large N
    if n > 100:
        min_dist = _min_pairwise_distance_grid(positions)
    else:
        min_dist = _min_pairwise_distance_bruteforce(positions)
    
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)


def _min_pairwise_distance_bruteforce(positions):
    """Brute force minimum pairwise distance."""
    n = len(positions)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    return min_dist


def _min_pairwise_distance_grid(positions):
    """Grid-based minimum pairwise distance for larger N."""
    n = len(positions)
    if n < 2:
        return np.inf
    
    # Use a simple grid to reduce comparisons
    # Estimate cell size
    r_est = 0.05
    cell_size = 2 * r_est
    
    # Create grid
    grid = {}
    for i, pos in enumerate(positions):
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    
    min_dist = np.inf
    for i in range(n):
        pos = positions[i]
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        
        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in grid:
                    for j in grid[key]:
                        if j > i:
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_dist:
                                min_dist = d
    
    return min_dist


def _local_refinement(positions, max_iter=200):
    """
    Local refinement: iteratively push overlapping circles apart and 
    pull circles outside the container back in.
    """
    positions = positions.copy()
    n = len(positions)
    
    for iteration in range(max_iter):
        # Compute current minimum radius
        r = _compute_min_radius(positions)
        if r <= 0:
            break
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion for close circles
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = positions[i] - positions[j]
                d = np.linalg.norm(d_vec)
                if d < 2 * r * 1.1:  # Only interact if close
                    if d > 1e-10:
                        # Repulsive force
                        force_mag = (2 * r - d) / d
                        forces[i] += force_mag * d_vec
                        forces[j] -= force_mag * d_vec
        
        # Container constraint: push inward if too close to boundary
        radii = np.hypot(positions[:, 0], positions[:, 1])
        for i in range(n):
            if radii[i] > 1.0 - r:
                # Push toward center
                pos = positions[i]
                if radii[i] > 1e-10:
                    force_mag = (radii[i] - (1.0 - r)) / radii[i]
                    forces[i] -= force_mag * pos
        
        # Update positions with small step
        step_size = 0.01 * r
        positions += step_size * forces
        
        # Ensure positions stay within reasonable bounds
        radii = np.hypot(positions[:, 0], positions[:, 1])
        mask = radii > 1.5
        if np.any(mask):
            positions[mask] = positions[mask] * (1.0 / radii[mask])[:, None]
        
        # Check for improvement
        new_r = _compute_min_radius(positions)
        if new_r < r * 0.99 or not np.isfinite(new_r):
            # Revert
            positions -= step_size * forces
            break
    
    return positions


def _hexagonal_fallback(N):
    """Fallback: hexagonal lattice cut to unit disk."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98
# EVOLVE-BLOCK-END
