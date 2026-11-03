"""Pack 50 equal circles in a unit square using Simulated Annealing."""
import numpy as np
import time
import random

N = 50
DEADLINE = 148.0  # Safety margin before 150s kill

def _radius(centers):
    """Compute the exact max common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraints: distance to nearest wall
    # Centers are in [-0.5, 0.5]
    wall_dists = np.minimum(np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
                            np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5))
    min_wall = np.min(wall_dists)
    
    # Pairwise constraints
    # Compute pairwise distances efficiently
    # diff[i, j, k] = centers[i, k] - centers[j, k]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    return min(min_wall, min_pair_dist / 2.0)

def _energy(centers, r_target):
    """
    Calculate the energy (total violation) for a given configuration and target radius.
    Lower is better. 0 means feasible.
    """
    # Wall violations
    # A circle of radius r is inside if center is in [r-0.5, 0.5-r]
    # Violation is how far outside this box the center is.
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Left wall: x < r - 0.5 -> violation = (r - 0.5) - x
    v_x_left = np.maximum(0, (r_target - 0.5) - x)
    # Right wall: x > 0.5 - r -> violation = x - (0.5 - r)
    v_x_right = np.maximum(0, x - (0.5 - r_target))
    # Bottom wall: y < r - 0.5 -> violation = (r - 0.5) - y
    v_y_bot = np.maximum(0, (r_target - 0.5) - y)
    # Top wall: y > 0.5 - r -> violation = y - (0.5 - r)
    v_y_top = np.maximum(0, y - (0.5 - r_target))
    
    wall_violation = np.sum(v_x_left**2) + np.sum(v_x_right**2) + \
                     np.sum(v_y_bot**2) + np.sum(v_y_top**2)
    
    # Pairwise violations
    # Distance must be >= 2*r. Violation = max(0, 2*r - dist)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    pair_violations = np.maximum(0, 2 * r_target - dist)
    pair_violation = np.sum(pair_violations ** 2)
    
    return wall_violation + pair_violation

def _sa_optimize(centers, r_target, time_limit, start_temp=0.01, end_temp=1e-6, cooling_rate=0.999):
    """
    Perform Simulated Annealing to find a feasible configuration for r_target.
    Returns the best centers found.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_energy = _energy(current_centers, r_target)
    best_centers = current_centers.copy()
    best_energy = current_energy
    
    temp = start_temp
    
    # Step size for moves
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        new_centers = current_centers.copy()
        # Randomly pick a few points to move
        num_moves = random.randint(1, 3)
        indices = random.sample(range(N), num_moves)
        for idx in indices:
            # Gaussian move
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            
        # Ensure we stay roughly within bounds to prevent huge energies
        # Soft clipping
        new_centers = np.clip(new_centers, -0.6, 0.6)
        
        new_energy = _energy(new_centers, r_target)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_centers = new_centers
            current_energy = new_energy
            if new_energy < best_energy:
                best_centers = new_centers.copy()
                best_energy = new_energy
        else:
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if temp > 0:
                prob = np.exp(-delta_e / temp)
                if random.random() < prob:
                    current_centers = new_centers
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_centers = new_centers.copy()
                        best_energy = new_energy
        
        # Cool down
        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp
            
        # Occasionally increase step size if stuck? 
        # For simplicity, keep fixed or slightly decrease.
        # Let's decrease step size slowly to allow fine tuning.
        # step_size *= 0.9999 -- might be too slow. 
        # Let's keep step size constant for now, relying on SA to explore.
        
    return best_centers

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def construct():
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal grid (usually a good start for dense packing)
    # Try to fit 50 circles. 
    # Hex grid spacing s. Area per circle ~ s^2 * sqrt(3)/2.
    # Total area 1. 50 * s^2 * sqrt(3)/2 <= 1 => s <= sqrt(2/(50*sqrt(3))) ~ 0.152
    # Radius ~ s/2 ~ 0.076. Known best is 0.0713.
    # Let's generate a hex grid with spacing 0.15 and center it.
    for sp in [0.14, 0.15, 0.16, 0.17]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        # Center the grid
        c_hex = c_hex - np.mean(c_hex, axis=0)
        # Scale to fit inside [-0.5, 0.5] with some margin
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 2. Rectangular grid 7x8 (56 points, take 50)
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    inits.append(grid[:N].copy())
    
    # 3. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Strategy:
    # 1. For each init, run SA to find a good local optimum for a specific r.
    # 2. We will try to maximize r. 
    #    Approach: Start with a low r, run SA to get feasible, then increase r slightly and run SA again.
    #    This is "Annealing with Radius Expansion".
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 2:
            break
            
        # Allocate time for this init
        # Let's do a "ramp up" strategy for each init.
        # Start with r = 0.05, go up to 0.08 in steps of 0.001.
        # For each r, run SA for a short time to find a feasible config.
        
        current_centers = init.copy()
        
        # Estimate a good starting r based on the init
        r_current = _radius(current_centers)
        # If r_current is small, start there. If large, start a bit lower to ensure feasibility.
        r_start = max(0.01, r_current * 0.9)
        
        # We want to go up to ~0.075
        r_target = r_start
        max_r_target = 0.075
        
        # Time allocation per r step
        # Total time for this init: let's say 10s if we have enough time.
        time_per_init = min(15.0, time_left * 0.6)
        
        # Number of r steps: (0.075 - r_start) / 0.001 ~ 25 steps.
        # Time per step: time_per_init / 25
        time_per_step = time_per_init / 30.0
        
        while r_target < max_r_target and time.time() - start_time < DEADLINE - 5:
            # Run SA for time_per_step
            new_centers = _sa_optimize(current_centers, r_target, time_per_step, 
                                       start_temp=0.005, end_temp=1e-5, cooling_rate=0.9995)
            
            # Check if we achieved feasibility (energy ~ 0)
            # Actually, SA minimizes energy. If energy is very small, it's good.
            # But we need to check the actual radius.
            r_achieved = _radius(new_centers)
            
            if r_achieved >= r_target - 1e-5:
                # We successfully packed at r_target. Update current state and increase r.
                current_centers = new_centers
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_centers = new_centers.copy()
                
                # Increase r_target
                r_target += 0.0005 # Small increment to be safe
            else:
                # Failed to reach r_target. 
                # Maybe reduce increment or stop?
                # Let's try a slightly smaller increment next time, or just stop if stuck.
                # If we can't even reach r_start, we might have a bad init.
                if r_target <= r_start + 0.001:
                    break
                # Try smaller step
                r_target += 0.0002
                
        # If we have time left, do a final polish with SA at the current best r
        if time.time() - start_time < DEADLINE - 5 and best_centers is not None:
            time_left = DEADLINE - 5 - (time.time() - start_time)
            if time_left > 1.0:
                # Polish the best result found so far
                # Try to squeeze out a bit more radius
                r_polish = best_r + 0.0001
                polished = _sa_optimize(best_centers, r_polish, time_left, 
                                        start_temp=0.001, end_temp=1e-6, cooling_rate=0.9999)
                r_polished = _radius(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished
    
    # Fallback: if no valid centers found, return a simple grid
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

# Ensure we don't crash if called multiple times or if state is weird
if __name__ == "__main__":
    # Simple self-test
    c = construct()
    r = _radius(c)
    print(f"Radius: {r}")
