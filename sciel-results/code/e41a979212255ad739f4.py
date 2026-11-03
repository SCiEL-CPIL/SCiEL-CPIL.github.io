"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the valid radius for a set of points."""
    n = len(pts)
    # Boundary constraint
    r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    r_min_boundary = np.min(r_boundary)
    # Pairwise constraint
    if n > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_min_pair = np.min(dist) / 2.0
    else:
        r_min_pair = r_min_boundary
    return min(r_min_boundary, r_min_pair)


def _hexagonal_init(n):
    """Create a hexagonal close-packed initial configuration."""
    best_pts = None
    best_r = -1
    
    for n_cols in range(5, 12):
        rows_needed = int(np.ceil(n / n_cols))
        spacing = 1.0 / (n_cols - 0.5)
        row_height = spacing * np.sqrt(3) / 2
        total_h = (rows_needed - 1) * row_height
        scale = 1.0 / max(n_cols - 0.5, total_h / spacing)
        
        pts = []
        for row in range(rows_needed):
            y = -0.5 + row * row_height * scale
            offset = (row % 2) * 0.5 * spacing * scale
            for col in range(n_cols):
                x = -0.5 + (col + 0.5) * spacing * scale + offset
                if len(pts) < n:
                    x = np.clip(x, -0.5, 0.5)
                    y = np.clip(y, -0.5, 0.5)
                    pts.append([x, y])
        
        if len(pts) == n:
            pts = np.array(pts)
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n]
    
    return best_pts


def _random_init(n):
    """Random initialization within the square."""
    pts = np.random.uniform(-0.45, 0.45, size=(n, 2))
    return pts


def _local_refine(pts, max_time=30.0):
    """
    Coordinate-wise local refinement: iteratively try to expand the minimum radius
    by moving points along the direction that increases the minimum constraint.
    Uses a greedy approach: find the binding constraint, move the involved points
    to increase it.
    """
    start_time = time.time()
    pts = pts.copy().astype(float)
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # We'll use a simple iterative improvement:
    # 1. Find the current minimum radius and which constraints are binding
    # 2. Try small perturbations to improve
    
    step = 0.01
    no_improve_count = 0
    
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        r = _compute_radius(pts)
        
        # Find binding constraints
        # Boundary
        r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        # Pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum pairwise distance
        min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_dist_idx[0], min_dist_idx[1]
        if i == j:
            continue
        min_pair_dist = dist[i, j]
        r_pair = min_pair_dist / 2.0
        
        min_boundary_idx = np.argmin(r_boundary)
        r_bnd = r_boundary[min_boundary_idx]
        
        if r_pair < r_bnd:
            # Pairwise constraint is binding
            # Move i and j apart
            d = pts[i] - pts[j]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            d_unit = d / d_norm
            # Move them apart
            pts[i] += d_unit * step
            pts[j] -= d_unit * step
        else:
            # Boundary constraint is binding
            # Move the point toward center
            p = pts[min_boundary_idx]
            # Find which axis is binding
            for k in range(2):
                if p[k] > 0:
                    pts[min_boundary_idx, k] -= step
                else:
                    pts[min_boundary_idx, k] += step
        
        # Keep within bounds
        pts = np.clip(pts, -0.499, 0.499)
        
        new_r = _compute_radius(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                # Try a random perturbation
                pts = best_pts + np.random.randn(n, 2) * 0.005
                pts = np.clip(pts, -0.499, 0.499)
                no_improve_count = 0
                # Decrease step size
                step *= 0.95
        
    return best_pts, best_r


def _differential_evolution(pts, max_time=60.0):
    """
    Simple differential evolution to explore the landscape.
    """
    start_time = time.time()
    n = len(pts)
    best_pts = pts.copy().astype(float)
    best_r = _compute_radius(best_pts)
    
    # Population
    pop_size = 20
    population = [best_pts.copy()]
    for _ in range(pop_size - 1):
        pop = best_pts + np.random.randn(n, 2) * 0.05
        pop = np.clip(pop, -0.499, 0.499)
        population.append(pop)
    
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    
    gen = 0
    while time.time() - start_time < max_time:
        gen += 1
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = [j for j in range(pop_size) if j != i]
            a = np.random.choice(a)
            b = np.random.choice([j for j in range(pop_size) if j != i and j != a])
            c = np.random.choice([j for j in range(pop_size) if j != i and j != a and j != b])
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            
            # Crossover
            trial = population[i].copy()
            for idx in range(n * 2):
                if np.random.rand() < CR:
                    trial.flat[idx] = mutant.flat[idx]
            
            trial = np.clip(trial, -0.499, 0.499)
            
            # Evaluate
            trial_r = _compute_radius(trial)
            if trial_r > _compute_radius(population[i]):
                population[i] = trial
            
            # Update best
            if trial_r > best_r:
                best_r = trial_r
                best_pts = trial.copy()
        
        # Local refine the best
        if time.time() - start_time < max_time - 5:
            refined, refined_r = _local_refine(best_pts, max_time=min(10, max_time - (time.time() - start_time)))
            if refined_r > best_r:
                best_r = refined_r
                best_pts = refined
                population[0] = best_pts.copy()
        
        if time.time() - start_time > max_time - 10:
            break
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Multiple initialization strategies
    best_pts = None
    best_r = -1
    
    # 1. Hexagonal init + local refine
    if time.time() < deadline - 30:
        pts = _hexagonal_init(N)
        pts, r = _local_refine(pts, max_time=min(15, deadline - time.time() - 30))
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # 2. Random init + local refine
    if time.time() < deadline - 30:
        pts = _random_init(N)
        pts, r = _local_refine(pts, max_time=min(15, deadline - time.time() - 30))
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # 3. Differential evolution from best
    if time.time() < deadline - 10 and best_pts is not None:
        pts, r = _differential_evolution(best_pts, max_time=min(60, deadline - time.time() - 10))
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final local refine
    if time.time() < deadline - 5 and best_pts is not None:
        pts, r = _local_refine(best_pts, max_time=min(10, deadline - time.time() - 5))
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Ensure validity
    if best_pts is None:
        best_pts = _hexagonal_init(N)
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
