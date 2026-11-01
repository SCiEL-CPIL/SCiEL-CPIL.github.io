"""
Strategy: Discrete Perturbation Simulated Annealing (SA) on Critical Pairs.

Instead of continuous force-directed dynamics, this approach uses a
combinatorial-style search:
1. Initialize with a hexagonal packing.
2. In each step, identify the "critical set" of circles that are involved in
   the minimum distance constraints (either pair-pair or pair-wall).
3. Apply a random perturbation ONLY to these critical circles.
4. Use a Simulated Annealing acceptance criterion based on the change in the
   minimum radius (score).
5. Cool down the temperature over time.

This targets the bottlenecks directly, allowing the system to "shake" the
tightest parts of the packing to find a looser configuration, which is often
more effective than global forces for hard-sphere packing.
"""

import numpy as np
import time

N = 50

def _score(cents):
    """
    Compute the packing radius (score) and identify critical indices.
    Returns:
        r: float, the minimum radius.
        critical_pairs: list of tuple (i, j) for pair constraints, or (i, -1) for wall.
    """
    n = len(cents)
    # Wall distances
    # dist to left: 0.5 - x, right: 0.5 + x, bottom: 0.5 - y, top: 0.5 + y
    # min wall dist for circle i is min(0.5 - |x_i|, 0.5 - |y_i|)
    wall_dists = np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1]))
    
    # Pairwise distances
    # diff[i, j] = cents[i] - cents[j]
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # The radius is half the min distance
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    # Identify critical constraints
    critical_indices = set()
    
    # Check pair constraints
    # We want pairs where dist is close to the min_pair_dist
    # To be efficient, we can just take the global min pair
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    # Note: argmin on full matrix includes diagonal (inf) and symmetric. 
    # np.argmin returns first occurrence. 
    i, j = min_pair_idx
    if dists[i, j] < np.inf:
        critical_indices.add(i)
        critical_indices.add(j)
        
    # Check wall constraints
    # Find indices where wall_dist is close to min_wall_dist
    # Tolerance for "critical"
    tol = 1e-9
    wall_crit_mask = wall_dists <= (min_wall_dist + tol)
    for i in np.where(wall_crit_mask)[0]:
        critical_indices.add(i)
        
    return r, list(critical_indices)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0
    
    # Try multiple row configurations
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        # Estimate r to fit
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r_actual, _ = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r, _ = _score(best_cents)
    
    return best_cents, best_r

def _simulated_annealing(cents, time_limit=100.0):
    """
    Optimize using Simulated Annealing with critical-pair perturbations.
    """
    start_time = time.time()
    best_cents = cents.copy()
    best_r, _ = _score(best_cents)
    current_cents = best_cents.copy()
    current_r, _ = _score(current_cents)
    
    n = len(cents)
    rng = np.random.default_rng(42)
    
    # SA Parameters
    # Temperature should be in units of radius difference
    T = 0.001 
    T_min = 1e-6
    cooling_rate = 0.9995 # Slow cooling
    perturbation_scale = 0.01 # Base scale for random moves
    
    while time.time() - start_time < time_limit:
        # Identify critical circles
        _, critical_indices = _score(current_cents)
        
        if not critical_indices:
            # Fallback: pick random circles if no critical found (shouldn't happen)
            critical_indices = [rng.integers(0, n)]
            
        # Perturb only critical circles
        new_cents = current_cents.copy()
        # Random Gaussian noise
        noise = rng.normal(0, perturbation_scale, size=(len(critical_indices), 2))
        for idx, i in enumerate(critical_indices):
            new_cents[i] += noise[idx]
            
        # Keep within bounds (soft clamp for validity check, but SA handles rejection)
        # Actually, if we go outside, the score drops to 0 or negative? 
        # The score function returns min(dist). If outside, dist is negative? 
        # No, 0.5 - |x| becomes negative. 
        # So score will drop significantly, leading to rejection. 
        # This is fine.
        
        new_r, _ = _score(new_cents)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            # Improvement
            current_cents = new_cents
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_cents = current_cents.copy()
        else:
            # Worse: accept with probability exp(delta_r / T)
            if T > 0:
                accept_prob = np.exp(delta_r / T)
                if rng.random() < accept_prob:
                    current_cents = new_cents
                    current_r = new_r
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            # Reheat slightly or stop? 
            # For a fixed time budget, let's just keep going with minimal T
            # Or restart T? 
            # Let's try a "restart" strategy: if T is too low, reset T and keep best.
            # This allows escaping deep local minima.
            T = 0.0005 
            # Optionally, we could also perturb best_cents to restart from there?
            # No, keep current state but allow larger jumps via higher T.
            
        # Adaptive perturbation scale?
        # If we are stuck (no improvement in last X steps), increase scale?
        # Simple heuristic: scale increases as T decreases? No, usually scale decreases.
        # Let's keep scale constant for now, T controls exploration depth.
        
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Step 1: Create hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Simulated Annealing optimization
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        # Run SA for the remaining time minus a small buffer
        cents, r1 = _simulated_annealing(best_cents, time_limit=min(remaining - 2, 110))
        if r1 > best_r:
            best_r = r1
            best_cents = cents
            
    # Final polish: Try to fix any tiny violations if we are very close to boundary
    # The score function handles validity, but let's ensure we return a valid array
    # If any point is outside, clamp it. This might lower score but ensures validity.
    # However, the score function returns a low score for invalid points, so SA should avoid them.
    # Just in case:
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
