import numpy as np
import time
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill
    
    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals of outer hexagon (centered at 0,0, rotation 0)
    # Faces are perpendicular to vectors at angles 0, 60, ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def get_hex_verts(x, y, angle_deg):
        """Get vertices of hexagon at (x, y) with given rotation."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts
    
    def calc_side(centers, angles):
        """Calculate required outer hexagon side length."""
        max_apothem = 0.0
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            # Project all vertices onto all normals
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3)
    
    def check_non_overlap_exact(centers, angles, tol=1e-9):
        """Check if hexagons are non-overlapping using exact polygon intersection."""
        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])
                
                # Check if any vertex of i is inside j or vice versa
                def point_in_hexagon(point, verts):
                    # Check if point is inside hexagon using winding number
                    inside = True
                    for k in range(6):
                        v1 = verts[k]
                        v2 = verts[(k + 1) % 6]
                        # Check if point is on the wrong side of edge
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        if np.dot(normal, point - v1) < -tol:
                            inside = False
                            break
                    return inside
                
                # Check vertices of i against j
                for v in verts_i:
                    if point_in_hexagon(v, verts_j):
                        return False
                # Check vertices of j against i
                for v in verts_j:
                    if point_in_hexagon(v, verts_i):
                        return False
        return True
    
    def calc_overlap_penalty(centers, angles, penalty_scale=100.0):
        """Calculate smooth penalty for overlaps using distance-based heuristic."""
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    # Smooth penalty: quadratic for small violations
                    violation = min_dist - dist
                    penalty += penalty_scale * violation * violation
        return penalty

    # Generate candidate configurations using hexagonal lattice
    d = np.sqrt(3)  # Distance between centers of adjacent hexagons in lattice

    # Strategy: Use hexagonal lattice coordinates
    # In a hexagonal lattice, centers are at positions:
    # (i * d + j * d/2, j * 1.5) for integers i, j
    # But we need to be careful about the exact geometry

    # Let's use axial coordinates for hexagonal grid
    # Position in Cartesian: x = d * (i + j/2), y = 1.5 * j
    # Actually, for unit hexagons (side=1), the distance between centers
    # of adjacent hexagons sharing an edge is sqrt(3)

    # Let me use a simpler approach: generate candidate positions on a grid
    # and try different subsets

    # Hexagonal lattice positions within a reasonable range
    # Centers at (x, y) where x = a * sqrt(3) + b * sqrt(3)/2, y = b * 1.5
    # for integers a, b

    candidate_positions = []
    for b in range(-3, 4):  # y coordinate index
        for a in range(-3, 4):  # x coordinate index
            x = a * d + b * d / 2
            y = b * 1.5
            candidate_positions.append((x, y))

    # Convert to numpy array
    candidate_positions = np.array(candidate_positions)
    n_candidates = len(candidate_positions)

    # Precompute pairwise distances
    # dist_matrix[i, j] = distance between candidate i and candidate j
    dist_matrix = np.zeros((n_candidates, n_candidates))
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            dx = candidate_positions[i, 0] - candidate_positions[j, 0]
            dy = candidate_positions[i, 1] - candidate_positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Find all valid subsets of N positions that are non-overlapping
    # Use backtracking with pruning

    best_side = float('inf')
    best_centers = None
    best_angles = None

    # For efficiency, use a greedy approach with local search
    # First, try to find a good configuration using simulated annealing

    def objective(centers, angles):
        """Objective function: side length + penalty for overlaps."""
        side = calc_side(centers, angles)
        
        # Penalty for overlaps
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return side + penalty

    # Simulated annealing
    def simulated_annealing(initial_centers, initial_angles, max_iters=30000, start_temp=1.0, end_temp=0.001):
        """Run simulated annealing to optimize the packing."""
        current_centers = initial_centers.copy()
        current_angles = initial_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj
        
        # Track best valid (non-overlapping) solution
        best_valid_side = float('inf')
        best_valid_centers = None
        best_valid_angles = None
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Generate perturbation - mix of small and medium steps
            if np.random.random() < 0.7:
                # Small step
                idx = np.random.randint(N)
                dx = np.random.normal(0, 0.15)
                dy = np.random.normal(0, 0.15)
                dangle = np.random.normal(0, 3.0)
            else:
                # Medium step - move two hexagons
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    idx2 = (idx1 + 1) % N
                dx1 = np.random.normal(0, 0.3)
                dy1 = np.random.normal(0, 0.3)
                dx2 = np.random.normal(0, 0.3)
                dy2 = np.random.normal(0, 0.3)
                dangle1 = np.random.normal(0, 5.0)
                dangle2 = np.random.normal(0, 5.0)
                
                new_centers = current_centers.copy()
                new_angles = current_angles.copy()
                new_centers[idx1, 0] += dx1
                new_centers[idx1, 1] += dy1
                new_centers[idx2, 0] += dx2
                new_centers[idx2, 1] += dy2
                new_angles[idx1] += dangle1
                new_angles[idx2] += dangle2
            if np.random.random() >= 0.7:
                pass  # Already handled above
            else:
                new_centers = current_centers.copy()
                new_angles = current_angles.copy()
                new_centers[idx, 0] += dx
                new_centers[idx, 1] += dy
                new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
                
                # Check if this is a valid (non-overlapping) configuration
                if check_non_overlap_exact(current_centers, current_angles):
                    side = calc_side(current_centers, current_angles)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_centers = current_centers.copy()
                        best_valid_angles = current_angles.copy()
        
        # Return the best valid solution if found, otherwise best objective
        if best_valid_centers is not None:
            return best_valid_centers, best_valid_angles, best_valid_side
        else:
            return best_centers, best_angles, best_obj

    # Generate multiple initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Config 1: Hexagonal cluster centered at origin
    # 7 hexagons in a flower pattern + 4 more
    centers1 = []
    # Center
    centers1.append((0, 0))
    # 6 surrounding
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    # 4 more: extend in some directions
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        # Place at distance 2*d in these directions
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    
    while len(centers1) < N:
        # Add more in a ring
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.5 * d
        centers1.append((r * np.cos(angle), r * np.sin(angle)))
    
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))
    
    # Config 2: Linear arrangement
    centers2 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))
    
    # Config 3: Two rows
    centers3 = []
    for i in range(N):
        if i < 6:
            centers3.append((i * d - 2.5 * d, 0))
        else:
            centers3.append(((i - 6) * d - 1.5 * d + d / 2, 1.5))
    centers3 = np.array(centers3)
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))
    
    # Config 4: Random scatter
    centers4 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles4 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers4, angles4))
    
    # Config 5: Tighter hexagonal with some rotations
    centers5 = []
    # Try a 3-4-3 or 2-3-2-3-1 pattern
    # Row 0: 2 hexagons
    centers5.append((-d / 2, 0))
    centers5.append((d / 2, 0))
    # Row 1: 3 hexagons
    centers5.append((-d, 1.5))
    centers5.append((0, 1.5))
    centers5.append((d, 1.5))
    # Row 2: 2 hexagons
    centers5.append((-d / 2, 3.0))
    centers5.append((d / 2, 3.0))
    # Row 3: 3 hexagons
    centers5.append((-d, -1.5))
    centers5.append((0, -1.5))
    centers5.append((d, -1.5))
    # Row 4: 1 hexagon
    centers5.append((0, -3.0))
    
    centers5 = np.array(centers5[:N])
    angles5 = np.zeros(N)
    initial_configs.append((centers5, angles5))
    
    # Run simulated annealing from each initial configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        try:
            result_centers, result_angles, result_obj = simulated_annealing(
                init_centers, init_angles, max_iters=25000, start_temp=0.8, end_temp=0.001
            )
            
            # Check if the result is valid (no overlaps)
            if check_non_overlap_exact(result_centers, result_angles):
                side = calc_side(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass
    
    # If we haven't found a good solution, try more aggressive search
    if best_side > 4.0 or best_centers is None:
        # Try a different approach: start from a known good packing and perturb
        # Known: 7 hexagons in a flower fit in a hexagon of side 3
        # Let's try to extend this
        
        # Use a grid of candidate positions and do a combinatorial search
        # for the best subset of N positions
        
        # Sort candidate positions by distance from origin
        dists = np.linalg.norm(candidate_positions, axis=1)
        sorted_indices = np.argsort(dists)
        
        # Try subsets of the closest positions
        for start_idx in range(0, min(50, n_candidates - N + 1), 2):
            if time.time() > deadline - 5:
                break
            
            # Take N positions from a window of candidates
            window_start = sorted_indices[start_idx]
            window_end = min(window_start + N + 10, n_candidates)
            candidate_subset = sorted_indices[window_start:window_end]
            
            if len(candidate_subset) < N:
                continue
            
            # Try all combinations of N from this subset (if small enough)
            if len(candidate_subset) <= 15:
                for combo in itertools.combinations(range(len(candidate_subset)), N):
                    indices = candidate_subset[list(combo)]
                    centers = candidate_positions[indices]
                    angles = np.zeros(N)
                    
                    if not check_non_overlap_exact(centers, angles):
                        continue
                    
                    side = calc_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
            
            else:
                # Use greedy: pick N positions that minimize the bounding hexagon
                # Start with the closest N
                selected = list(candidate_subset[:N])
                best_subset_obj = objective(candidate_positions[selected], np.zeros(N))
                
                # Try swapping
                for _ in range(150):
                    if time.time() > deadline - 5:
                        break
                    
                    # Random swap
                    i = np.random.randint(N)
                    j = np.random.randint(len(candidate_subset))
                    if candidate_subset[j] in selected:
                        continue
                    
                    new_selected = selected.copy()
                    new_selected[i] = candidate_subset[j]
                    new_obj = objective(candidate_positions[new_selected], np.zeros(N))
                    
                    if new_obj < best_subset_obj:
                        selected = new_selected
                        best_subset_obj = new_obj
                
                centers = candidate_positions[selected]
                angles = np.zeros(N)
                if check_non_overlap_exact(centers, angles):
                    side = calc_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
    
    # Final refinement: local search on the best solution
    if best_centers is not None and time.time() < deadline - 8:
        # Do a fine-tuning local search with exact overlap checking
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_side = calc_side(current_centers, current_angles)
        
        # Verify current solution is valid
        if not check_non_overlap_exact(current_centers, current_angles):
            # Try to fix overlaps with local search
            for it in range(10000):
                if time.time() > deadline - 5:
                    break
                
                idx = np.random.randint(N)
                dx = np.random.normal(0, 0.1)
                dy = np.random.normal(0, 0.1)
                dangle = np.random.normal(0, 2.0)
                
                new_centers = current_centers.copy()
                new_angles = current_angles.copy()
                new_centers[idx, 0] += dx
                new_centers[idx, 1] += dy
                new_angles[idx] += dangle
                
                if check_non_overlap_exact(new_centers, new_angles):
                    new_side = calc_side(new_centers, new_angles)
                    if new_side < current_side:
                        current_centers = new_centers
                        current_angles = new_angles
                        current_side = new_side
        
        # Now do fine-tuning to reduce side length
        step_size = 0.03
        for it in range(8000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 8)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            # Check if still valid
            if check_non_overlap_exact(new_centers, new_angles):
                new_side = calc_side(new_centers, new_angles)
                if new_side < current_side:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_side = new_side
            
            # Decay step size
            step_size *= 0.9995
        
        if current_side < best_side:
            best_centers = current_centers
            best_angles = current_angles
            best_side = current_side
    
    # Construct output
    if best_centers is None:
        # Fallback: widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
