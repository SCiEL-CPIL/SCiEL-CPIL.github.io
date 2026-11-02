"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = N
    time_budget = 115.0
    start_time = time.time()
    
    # 1. Initialization: Hexagonal Lattice
    # Approximate optimal radius for hex packing in unit circle
    # r_hex = sqrt(1 / (N * 2/sqrt(3))) approx
    # Let's generate a hex grid
    points = np.zeros((n, 2))
    # Hex grid spacing 'a' is related to radius r by a = 2*r / sqrt(3) ? No, distance is 2r.
    # In hex lattice, nearest neighbor distance is 'a'. So a = 2r.
    # Let's pick an initial r guess
    r_init = 0.05
    a = 2 * r_init
    dx = a
    dy = a * np.sqrt(3) / 2
    
    # Generate points in a large enough box
    max_coord = 1.2
    xs = np.arange(-max_coord, max_coord, dx)
    ys = np.arange(-max_coord, max_coord, dy)
    
    hex_points = []
    for i, x in enumerate(xs):
        for y in ys:
            # Offset every other row
            y_offset = dy / 2 if i % 2 == 1 else 0
            hex_points.append([x, y + y_offset])
            
    hex_points = np.array(hex_points)
    
    # Keep only points inside the unit circle (with some margin)
    dists = np.linalg.norm(hex_points, axis=1)
    mask = dists < 1.0
    hex_points = hex_points[mask]
    
    # If we have too many, subsample. If too few, we need to adjust r_init or logic.
    # For N=300, r=0.05, a=0.1. 
    # Area of hex cell = a^2 * sqrt(3)/2 = 0.01 * 0.866 = 0.00866.
    # Area of circle = pi. Num points approx pi / 0.00866 = 363.
    # So we should have around 300-360 points.
    
    if len(hex_points) > n:
        # Randomly select n points
        indices = np.random.choice(len(hex_points), n, replace=False)
        points = hex_points[indices]
    elif len(hex_points) < n:
        # Add random points if we don't have enough (should not happen with r_init=0.05)
        # Or reduce r_init and regenerate. For simplicity, just add randoms.
        points = np.vstack([hex_points, np.random.uniform(-1, 1, (n - len(hex_points), 2))])
        
    # Shuffle to break symmetry
    np.random.shuffle(points)

    # 2. Spatial Hashing Grid
    # Cell size should be slightly larger than 2*r to ensure neighbors are in adjacent cells
    # We will update grid as points move.
    
    def get_grid_neighbors(grid, point, cell_size):
        # Returns indices of points in the same and 8 neighboring cells
        cx = int((point[0] + 1.0) // cell_size)
        cy = int((point[1] + 1.0) // cell_size)
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in grid:
                    neighbors.extend(grid[key])
        return neighbors

    def build_grid(points, cell_size):
        grid = {}
        for i, p in enumerate(points):
            cx = int((p[0] + 1.0) // cell_size)
            cy = int((p[1] + 1.0) // cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        return grid

    def calc_energy_and_forces(points, r, grid, cell_size):
        # Energy = sum of squared overlaps
        # Forces = gradient of energy
        energy = 0.0
        forces = np.zeros_like(points)
        
        # We only need to check pairs. 
        # To avoid double counting and O(N^2), we iterate through grid cells.
        # For efficiency in Python, we will just iterate i, find neighbors j > i.
        
        # Rebuild grid is expensive. We assume points don't move too far between checks?
        # No, in SA we move one point at a time. We can update the grid incrementally or just use it for neighbor lookup.
        # Since we move one point, we can just check its neighbors in the current grid.
        # However, the grid becomes stale. 
        # Strategy: Rebuild grid every K steps or use a static grid for the initial r and update as r increases?
        # Better: Just use the grid to find *candidate* neighbors. Verify distance.
        
        # Let's simplify: For each point i, find neighbors from grid, check distance to j.
        # Since we move one point at a time, the grid is slightly inaccurate but acceptable for SA.
        
        for i in range(n):
            p_i = points[i]
            # Get candidates from grid
            # Note: Grid is built for the *current* positions. 
            # If we move point i, its cell might change. 
            # We will rebuild the grid periodically or just accept the error.
            # For speed, we will rebuild the grid every 50 steps.
            
            cx = int((p_i[0] + 1.0) // cell_size)
            cy = int((p_i[1] + 1.0) // cell_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if j <= i: # Only check j > i to avoid double count
                                continue
                            p_j = points[j]
                            diff = p_i - p_j
                            dist_sq = np.dot(diff, diff)
                            dist = np.sqrt(dist_sq)
                            min_dist = 2 * r
                            
                            if dist < min_dist:
                                # Overlap
                                overlap = min_dist - dist
                                energy += overlap * overlap
                                # Force on i: push away from j
                                # Gradient of (min_dist - dist)^2 w.r.t p_i
                                # d/dp_i ( (min_dist - dist)^2 ) = 2 (min_dist - dist) * (-1) * d(dist)/d(p_i)
                                # d(dist)/d(p_i) = (p_i - p_j) / dist
                                # So force = -2 (min_dist - dist) * (p_i - p_j) / dist
                                # Wait, we want to minimize energy. 
                                # Energy increases as dist decreases. 
                                # Gradient points in direction of increasing energy. 
                                # We want to move in direction of -gradient.
                                # grad_E = 2 (min_dist - dist) * (p_j - p_i) / dist  (Vector pointing from i to j? No.)
                                # Let's derive carefully.
                                # E = (D - d)^2. d = ||p_i - p_j||.
                                # dE/dp_i = 2(D-d) * (-1) * d/dp_i ||p_i - p_j||
                                # d/dp_i ||p_i - p_j|| = (p_i - p_j) / d
                                # dE/dp_i = -2(D-d) * (p_i - p_j) / d = 2(D-d) * (p_j - p_i) / d
                                # This vector points from i to j. 
                                # So gradient points towards j. 
                                # To minimize E, we move opposite to gradient: away from j.
                                # Force = -grad_E = -2(D-d) * (p_j - p_i) / d = 2(D-d) * (p_i - p_j) / d
                                if dist > 1e-6:
                                    f = 2 * overlap * diff / dist
                                    forces[i] += f
                                    forces[j] -= f
                            
                            # Boundary check
                            # E_bound = max(0, ||p_i|| - (1-r))^2
                            # Only if outside
                            r_max = 1.0 - r
                            norm_i = np.linalg.norm(p_i)
                            if norm_i > r_max:
                                viol = norm_i - r_max
                                energy += viol * viol
                                # grad_E_bound = 2 * viol * p_i / norm_i
                                # Force = -grad = -2 * viol * p_i / norm_i
                                if norm_i > 1e-6:
                                    forces[i] -= 2 * viol * p_i / norm_i
                                    
        return energy, forces

    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.0001
    cooling_rate = 0.9999
    
    # We will try to pack for a specific radius r_target
    # Start with a radius slightly smaller than theoretical max
    # Theoretical max ~ 0.055. Best known 0.0531.
    # Let's try to reach 0.053.
    
    best_points = points.copy()
    best_radius = 0.0
    
    # We will run SA for a range of radii, increasing slowly.
    # Or just run SA with a fixed r that we hope is achievable.
    # If we pick r=0.053, and it fails, we get a bad score.
    # Better: Start with r=0.04, run SA, if success, increase r.
    
    current_r = 0.04
    max_r_try = 0.054
    
    steps_per_r = 5000
    grid_rebuild_interval = 100
    
    while current_r < max_r_try and time.time() - start_time < time_budget:
        T = T_start
        cell_size = 2 * current_r * 1.1 # Slightly larger than diameter
        
        # Rebuild grid initially
        grid = build_grid(points, cell_size)
        
        current_energy = 0.0
        # Calculate initial energy
        e, _ = calc_energy_and_forces(points, current_r, grid, cell_size)
        current_energy = e
        
        for step in range(steps_per_r):
            if time.time() - start_time > time_budget:
                break
            
            # Pick random point
            i = np.random.randint(n)
            
            # Propose move
            # Step size depends on T
            step_size = 0.01 * (T / T_start + 0.1)
            delta = np.random.normal(0, step_size, 2)
            old_p = points[i].copy()
            new_p = old_p + delta
            
            # Keep inside container roughly? SA can handle boundary forces, 
            # but large jumps out are bad. Let's clamp.
            new_p = np.clip(new_p, -1.0, 1.0)
            
            # Calculate energy change
            # Instead of full recalc, we can approximate, but full recalc is safer.
            # Optimization: Only recalc energy for neighbors of i.
            # For simplicity and correctness, we recalc full energy. 
            # With spatial hashing, this is fast enough? 
            # N=300. 300 * neighbors (~10-20) = 3000-6000 ops. 
            # 5000 steps * 3000 ops = 15M ops. In Python, this might be slow.
            
            # Let's optimize: 
            # 1. Store current energy.
            # 2. Remove i from grid.
            # 3. Move i.
            # 4. Add i to grid.
            # 5. Recalc energy for i and its neighbors.
            
            # This is complex to implement correctly in a short time.
            # Alternative: Just use the full calc function but it's O(N * K).
            # Let's try to make the full calc faster by using NumPy where possible?
            # No, the grid logic is Python loops.
            
            # Let's reduce steps_per_r if it's too slow.
            # Or increase time_budget usage.
            
            # Let's assume the full calc is the bottleneck.
            # We will do a simplified energy calc for the proposal:
            # Only check neighbors of i.
            
            # Get old neighbors
            cx_old = int((old_p[0] + 1.0) // cell_size)
            cy_old = int((old_p[1] + 1.0) // cell_size)
            old_neighbors = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx_old + dx, cy_old + dy)
                    if key in grid:
                        old_neighbors.update(grid[key])
            old_neighbors.discard(i)
            
            # Get new neighbors
            cx_new = int((new_p[0] + 1.0) // cell_size)
            cy_new = int((new_p[1] + 1.0) // cell_size)
            new_neighbors = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx_new + dx, cy_new + dy)
                    if key in grid:
                        new_neighbors.update(grid[key])
            new_neighbors.discard(i)
            
            # Union of neighbors
            affected = old_neighbors.union(new_neighbors)
            
            # Calculate old energy contribution of i and affected
            e_old = 0.0
            for j in affected:
                # Pair i-j
                diff = old_p - points[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * current_r:
                    e_old += (2 * current_r - dist) ** 2
                # Boundary i
                norm_i = np.linalg.norm(old_p)
                if norm_i > 1 - current_r:
                    e_old += (norm_i - (1 - current_r)) ** 2
            
            # Calculate new energy contribution
            e_new = 0.0
            for j in affected:
                diff = new_p - points[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * current_r:
                    e_new += (2 * current_r - dist) ** 2
                # Boundary i
                norm_i = np.linalg.norm(new_p)
                if norm_i > 1 - current_r:
                    e_new += (norm_i - (1 - current_r)) ** 2
            
            delta_e = e_new - e_old
            
            # Acceptance
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                # Accept move
                points[i] = new_p
                current_energy += delta_e
                
                # Update grid
                # Remove i from old cell
                if (cx_old, cy_old) in grid:
                    if i in grid[(cx_old, cy_old)]:
                        grid[(cx_old, cy_old)].remove(i)
                        if not grid[(cx_old, cy_old)]:
                            del grid[(cx_old, cy_old)]
                
                # Add i to new cell
                key_new = (cx_new, cy_new)
                if key_new not in grid:
                    grid[key_new] = []
                grid[key_new].append(i)
                
                # Rebuild grid periodically to fix drift
                if step % grid_rebuild_interval == 0:
                    grid = build_grid(points, cell_size)
                    # Recalc energy to be safe
                    current_energy, _ = calc_energy_and_forces(points, current_r, grid, cell_size)
            
            # Cool down
            T *= cooling_rate
            
            # Track best
            if current_energy < 1e-6:
                 # We found a valid packing for this r
                 # Check actual radius
                 # For now, assume valid
                 pass
        
        # After SA for this r, check if valid
        # Recalc energy to be sure
        grid = build_grid(points, 2 * current_r * 1.1)
        final_e, _ = calc_energy_and_forces(points, current_r, grid, 2 * current_r * 1.1)
        
        if final_e < 1e-5:
            best_radius = current_r
            best_points = points.copy()
            # Increase r
            current_r += 0.001
        else:
            # Failed, maybe retry with smaller step or break
            # If we fail, we can try to repair by reducing r?
            # For simplicity, break if we can't increase
            # Or just keep the last best
            break
            
    # Final verification and score calculation
    # Calculate the actual radius of best_points
    def calc_actual_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    # If best_radius is 0, it means we never succeeded. 
    # In that case, use the initial points or the last points.
    # Let's calculate the radius of the last points to be safe.
    r_final = calc_actual_radius(points)
    if r_final > best_radius:
        best_radius = r_final
        best_points = points.copy()
        
    return best_points
# EVOLVE-BLOCK-END
