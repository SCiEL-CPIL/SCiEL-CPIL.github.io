"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with hexagonal lattice.
2. Minimize a smooth repulsive energy (with wall terms) via L-BFGS-B with analytic gradient.
3. Refine via local search maximizing the exact minimum distance.
"""

N = 200
import numpy as np
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.06, 0.08, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is not None:
        return best_c

    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _simulated_annealing(c0, time_limit=60.0):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    c = c0.copy()
    current_r = _compute_radius(c)
    best_r = current_r
    best_c = c.copy()
    
    start_time = time.time()
    
    # Initial temperature
    T = 0.001
    # Cooling rate
    alpha = 0.995
    
    # Step size
    step_size = 0.02
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random particle
        i = rng.integers(0, N)
        
        # Propose a move
        # Random direction
        theta = rng.uniform(0, 2 * np.pi)
        move = np.array([np.cos(theta), np.sin(theta)]) * step_size
        
        new_ci = c[i] + move
        
        # Check bounds
        if not (-0.499 < new_ci[0] < 0.499 and -0.499 < new_ci[1] < 0.499):
            T *= alpha
            continue
            
        c[i] = new_ci
        new_r = _compute_radius(c)
        
        delta = new_r - current_r
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            c[i] = new_ci - move # Revert
            
        T *= alpha
        
    return best_c, best_r

def _local_polish(c, max_iters=2000, time_limit=20.0):
    """
    Local search to maximize minimum distance.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        min_val = np.min(dist_matrix)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_val = np.min(bdists)
        b_idx = np.argmin(bdists)

        if min_val / 2.0 < min_boundary_val:
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm
                step = 1e-4
                new_ci = c[i] + unit_vec * step
                new_cj = c[j] - unit_vec * step

                if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                    c[i] = new_ci
                    c[j] = new_cj
                    new_r = _compute_radius(c)
                    if new_r > current_r:
                        current_r = new_r
                    else:
                        c[i] = c[i] - unit_vec * step
                        c[j] = c[j] + unit_vec * step
        else:
            x, y = c[b_idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5

            min_d = min(d_left, d_right, d_down, d_up)
            step = 1e-4
            if min_d == d_left:
                new_pt = c[b_idx] + np.array([step, 0])
            elif min_d == d_right:
                new_pt = c[b_idx] + np.array([-step, 0])
            elif min_d == d_down:
                new_pt = c[b_idx] + np.array([0, step])
            else:
                new_pt = c[b_idx] + np.array([0, -step])

            if np.all(np.abs(new_pt) < 0.5):
                c[b_idx] = new_pt
                new_r = _compute_radius(c)
                if new_r > current_r:
                    current_r = new_r
                else:
                    c[b_idx] = c[b_idx] - np.sign(new_pt - c[b_idx]) * step

    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145

    c0 = _initialize_hexagonal()
    r0 = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r0

    # Run Simulated Annealing
    if time.time() < deadline - 10:
        c_sa, r_sa = _simulated_annealing(best_c, time_limit=min(60.0, deadline - time.time() - 10))
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()

    # Run Local Polish
    if time.time() < deadline - 5:
        c_polished, r_polished = _local_polish(best_c, max_iters=2000, time_limit=min(30.0, deadline - time.time() - 5))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
