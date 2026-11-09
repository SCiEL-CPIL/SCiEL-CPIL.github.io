"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Pairwise distances using vectorized computation
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _vectorized_force_directed(centres, time_limit=100.0):
    """
    Fully vectorized force-directed optimization.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    step = 0.003
    min_step = 1e-8
    patience = 0
    max_patience = 200
    
    while time.time() - start_time < time_limit:
        # Compute all pairwise distances (vectorized)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces vectorized
        # Repulsive force: for each pair, F = (1/d^3) * (ci - cj) for d < threshold
        threshold = 0.35
        mask = (dist < threshold) & (dist > 1e-10)
        
        # Force magnitude: 1/d^2, direction: (ci-cj)/d => total: (ci-cj)/d^3
        inv_d3 = np.zeros_like(dist)
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        
        # forces[i] = sum_j (ci - cj) / d^3 for all j != i where d < threshold
        # diff[i,j] = ci - cj
        # force on i from j: diff[i,j] * inv_d3[i,j]
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces: push away from walls
        for k in range(3):
            # If c[i,k] > 0.45, push left (negative force)
            # If c[i,k] < -0.45, push right (positive force)
            above = c[:, k] > 0.45
            below = c[:, k] < -0.45
            forces[above, k] -= (c[above, k] - 0.45) * 50.0
            forces[below, k] += (-0.45 - c[below, k]) * 50.0
        
        # Weak centering force
        forces -= 0.005 * c
        
        # Normalize forces for stable stepping
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        # Apply forces
        c += step * forces
        
        # Clip to valid range
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check improvement
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
            step = min(step * 1.02, 0.01)
        else:
            patience += 1
            step = max(step * 0.99, min_step)
        
        # Random perturbation if stuck
        if patience > max_patience:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
            c += rng.normal(0, 0.008, size=c.shape)
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            patience = 0
            step = 0.003
    
    return best_centres, best_radius


def _make_fcc_packing():
    """
    Create an FCC-inspired packing for 50 spheres.
    """
    # FCC basis
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate FCC lattice points in a 3x3x4 supercell
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    # Scale to fit in [-0.5, 0.5]^3
                    # Extent in x: (nx-1) + 0.5 = 2.5, so scale = 1.0/3.0
                    # Actually: max coord = (nx-1)+0.5 = 2.5, min = 0
                    # Center: subtract 1.25, divide by 2.5
                    p = (p - 1.25) / 2.5
                    points.append(p)
    
    points = np.array(points)
    
    # We have 144 points, need 50
    # Select 50 with maximum minimum distance using greedy algorithm
    selected = [0]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_random_packing(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _make_grid_packing():
    """Create a grid-based initial packing."""
    # 4x4x4 grid = 64 points, select 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 with best spacing
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    
    # Add small perturbation
    rng = np.random.RandomState(123)
    centres += rng.normal(0, 0.002, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _polish(centres, time_limit=10.0):
    """
    High-precision polish pass with very small steps.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    step = 0.0005
    min_step = 1e-10
    patience = 0
    max_patience = 500
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        threshold = 0.2
        mask = (dist < threshold) & (dist > 1e-10)
        
        inv_d3 = np.zeros_like(dist)
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        for k in range(3):
            above = c[:, k] > 0.48
            below = c[:, k] < -0.48
            forces[above, k] -= (c[above, k] - 0.48) * 100.0
            forces[below, k] += (-0.48 - c[below, k]) * 100.0
        
        forces -= 0.001 * c
        
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        c += step * forces
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
        else:
            patience += 1
            step = max(step * 0.98, min_step)
        
        if patience > max_patience:
            break
    
    return best_centres, best_radius


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: FCC-inspired packing
    if time.time() < deadline - 30:
        try:
            init = _make_fcc_packing()
            centres, radius = _vectorized_force_directed(init, time_limit=min(45, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 2: Grid-based packing
    if time.time() < deadline - 25 and best_centres is not None:
        try:
            init = _make_grid_packing()
            centres, radius = _vectorized_force_directed(init, time_limit=min(30, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with short optimizations
    if time.time() < deadline - 15:
        for seed in [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]:
            if time.time() > deadline - 8:
                break
            try:
                init = _make_random_packing(seed)
                time_left = deadline - time.time() - 3
                if time_left < 5:
                    break
                centres, radius = _vectorized_force_directed(init, time_limit=min(12, time_left))
                if radius > best_radius:
                    best_radius = radius
                    best_centres = centres
            except Exception:
                continue
    
    # Final polish pass
    if best_centres is not None and time.time() < deadline - 2:
        try:
            polished, pr = _polish(best_centres, time_limit=min(8, deadline - time.time() - 1))
            if pr > best_radius:
                best_radius = pr
                best_centres = polished
        except Exception:
            pass
    
    # If no valid result, fallback to simple grid
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
