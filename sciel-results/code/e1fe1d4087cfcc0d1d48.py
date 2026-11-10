"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance using cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _fibonacci_lattice():
    """Create a Fibonacci lattice (sunflower pattern) initialization."""
    # Generate N points using the golden angle
    n = N
    theta = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    i = np.arange(n)
    r = np.sqrt(i / n) * 0.5  # scale to fit in radius 0.5
    angles = i * theta
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    centres = np.column_stack((x, y))
    # Clip to ensure inside square
    centres = np.clip(centres, -0.49, 0.49)
    return centres


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                
                # Create the grid
                centres = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres.copy()
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _force_based_optimization(centres, deadline):
    """Repulsive force-based simulation with rigid constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Initial parameters
    step_size = 0.01
    no_improve_count = 0
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Find nearest neighbors using KDTree
        tree = KDTree(best_centres)
        dists, indices = tree.query(best_centres, k=5)
        
        # Compute repulsive forces from close neighbors
        forces = np.zeros_like(best_centres)
        for i in range(N):
            for k in range(1, 5):
                j = indices[i, k]
                d = dists[i, k]
                if d < 0.1:  # Only consider close neighbors
                    # Repulsive force inversely proportional to distance squared
                    direction = best_centres[i] - best_centres[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        force_mag = 1.0 / (d * d + 1e-6)
                        forces[i] += force_mag * direction / norm
        
        # Add wall repulsion forces
        for i in range(N):
            if best_centres[i, 0] < -0.4:
                forces[i, 0] += 1.0 / (0.5 + best_centres[i, 0])
            elif best_centres[i, 0] > 0.4:
                forces[i, 0] -= 1.0 / (0.5 - best_centres[i, 0])
            if best_centres[i, 1] < -0.4:
                forces[i, 1] += 1.0 / (0.5 + best_centres[i, 1])
            elif best_centres[i, 1] > 0.4:
                forces[i, 1] -= 1.0 / (0.5 - best_centres[i, 1])
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms < 1e-10] = 1.0
        forces = forces / force_norms
        
        # Try multiple step sizes
        improved = False
        for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
            if time.time() > deadline:
                break
            
            new_centres = best_centres + step_size * scale * forces
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            step_size *= 0.98
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-7 or no_improve_count > 100:
                break
    
    return best_centres, best_r


def _local_optimization(centres, deadline):
    """Local optimization using multi-point simultaneous updates with adaptive step sizing."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Initial step size based on current radius
    step_size = max(0.005, best_r * 0.3)
    no_improve_count = 0
    
    for it in range(10000):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Find nearest neighbor for each point using KDTree
        tree = KDTree(best_centres)
        dists, indices = tree.query(best_centres, k=2)
        min_pair_dists = dists[:, 1]
        nearest_indices = indices[:, 1]
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        
        # Select top K most constrained points to move simultaneously
        K = min(5, N)
        constrained_indices = np.argsort(constraints)[:K]
        
        # Compute displacement directions for each selected point
        displacements = np.zeros_like(best_centres)
        for idx in constrained_indices:
            if min_pair_dists[idx] / 2.0 <= wall_dists[idx]:
                # Push away from nearest neighbor
                direction = best_centres[idx] - best_centres[nearest_indices[idx]]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction /= norm
            else:
                # Push away from the closest wall
                dx = np.array([
                    0.5 - best_centres[idx, 0],
                    0.5 + best_centres[idx, 0],
                    0.5 - best_centres[idx, 1],
                    0.5 + best_centres[idx, 1]
                ])
                wall_idx = np.argmin(dx)
                direction = np.zeros(2)
                if wall_idx == 0:
                    direction[0] = -1.0
                elif wall_idx == 1:
                    direction[0] = 1.0
                elif wall_idx == 2:
                    direction[1] = -1.0
                else:
                    direction[1] = 1.0
            
            displacements[idx] = direction
        
        # Try multiple step sizes
        improved = False
        for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
            if time.time() > deadline:
                break
            
            delta = step_size * scale * displacements
            new_centres = best_centres + delta
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            # Quick estimate: check distances from moved points to all others
            new_r_est = _compute_radius(new_centres)
            if new_r_est <= best_r + 1e-12:
                continue
            
            # Full radius computation
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            step_size *= 0.98
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-7 or no_improve_count > 100:
                break
    
    return best_centres, best_r


def _polish_pass(centres, deadline):
    """High-precision polish: alternate between single-point and multi-point fine adjustments."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Start with very small step size
    step_size = max(0.001, best_r * 0.1)
    no_improve_count = 0
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Find nearest neighbor for each point using KDTree
        tree = KDTree(best_centres)
        dists, indices = tree.query(best_centres, k=2)
        min_pair_dists = dists[:, 1]
        nearest_indices = indices[:, 1]
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        
        # Alternate between single-point (even iterations) and multi-point (odd)
        if it % 2 == 0:
            # Single point: most constrained
            idx = np.argmin(constraints)
            constrained_indices = [idx]
        else:
            # Multi-point: top 3 most constrained
            constrained_indices = np.argsort(constraints)[:3]
        
        # Compute displacement directions
        displacements = np.zeros_like(best_centres)
        for idx in constrained_indices:
            if min_pair_dists[idx] / 2.0 <= wall_dists[idx]:
                direction = best_centres[idx] - best_centres[nearest_indices[idx]]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction /= norm
            else:
                dx = np.array([
                    0.5 - best_centres[idx, 0],
                    0.5 + best_centres[idx, 0],
                    0.5 - best_centres[idx, 1],
                    0.5 + best_centres[idx, 1]
                ])
                wall_idx = np.argmin(dx)
                direction = np.zeros(2)
                if wall_idx == 0:
                    direction[0] = -1.0
                elif wall_idx == 1:
                    direction[0] = 1.0
                elif wall_idx == 2:
                    direction[1] = -1.0
                else:
                    direction[1] = 1.0
            
            displacements[idx] = direction
        
        # Try multiple small step sizes
        improved = False
        for scale in [1.0, 0.5, 0.25, 0.1, 2.0]:
            if time.time() > deadline:
                break
            
            delta = step_size * scale * displacements
            new_centres = best_centres + delta
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-14:
                best_r = new_r
                best_centres = new_centres
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size *= 0.85
            if step_size < 1e-10 or no_improve_count > 50:
                break
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3  # Leave 3s safety margin
    
    best_centres = None
    best_r = -1
    
    # Try multiple initializations for diversity
    initializations = []
    
    # 1. Fibonacci lattice
    fib_centres = _fibonacci_lattice()
    initializations.append(("fibonacci", fib_centres))
    
    # 2. Hexagonal packing
    hex_centres = _hexagonal_packing()
    initializations.append(("hexagonal", hex_centres))
    
    # 3. Random perturbations of hexagonal
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        perturbed = hex_centres.copy()
        perturbation = np.random.uniform(-0.005, 0.005, size=perturbed.shape)
        perturbed += perturbation
        perturbed = np.clip(perturbed, -0.499, 0.499)
        initializations.append((f"hex_perturb_{seed}", perturbed))
    
    for name, centres in initializations:
        if time.time() > deadline:
            break
        
        current_r = _compute_radius(centres)
        
        # Force-based optimization first
        if time.time() < deadline - 20:
            centres, current_r = _force_based_optimization(centres, deadline)
        
        # Local optimization
        if time.time() < deadline - 10:
            centres, current_r = _local_optimization(centres, deadline)
        
        # High-precision polish pass
        if time.time() < deadline - 5:
            centres, current_r = _polish_pass(centres, deadline)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
    
    # Final check
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback to a valid grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
