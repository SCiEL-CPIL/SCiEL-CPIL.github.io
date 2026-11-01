def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Constants
    N = 12
    # Unit hexagon vertices at angle 0
    base_verts = np.array([
        [np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)
    ])
    
    def get_hex_verts(cx, cy, angle_deg):
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return base_verts @ rot.T + np.array([cx, cy])

    def compute_minimal_outer_side(centers, angles_deg):
        # Collect all vertices
        all_verts = []
        for i in range(N):
            verts = get_hex_verts(centers[i, 0], centers[i, 1], angles_deg[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        def calc_side(theta):
            # Normals to the sides of the outer hexagon
            # Outer hexagon is regular, so normals are at theta + i*pi/3
            angles = theta + np.arange(0, np.pi, np.pi/3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            # Project all vertices onto these normals
            dots = all_verts @ normals.T
            widths = dots.max(axis=0) - dots.min(axis=0)
            max_width = np.max(widths)
            # For a regular hexagon of side s, the width (distance between opposite sides) is s * sqrt(3)
            # So s = max_width / sqrt(3)
            return max_width / np.sqrt(3)

        # Coarse search for optimal orientation
        thetas_coarse = np.linspace(0, np.pi / 3, 180)
        sides = np.array([calc_side(th) for th in thetas_coarse])
        best_idx = np.argmin(sides)
        
        # Refine
        from scipy.optimize import minimize_scalar
        lo = thetas_coarse[max(0, best_idx - 2)]
        hi = thetas_coarse[min(len(thetas_coarse) - 1, best_idx + 2)]
        res = minimize_scalar(calc_side, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
        
        theta = res.x
        angles = theta + np.arange(0, np.pi, np.pi/3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        # Solve for center (cx, cy) such that cx*nx + cy*ny = mid for all 3 axes
        # We have 3 equations, 2 unknowns. Use least squares.
        A = normals.T
        b = mids
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return res.fun, c, theta

    def get_overlap_penalty(centers, angles_deg, threshold=0.0):
        """
        Calculate penalty based on minimum distance between any two hexagons.
        Returns 0 if disjoint, positive value if overlapping or too close.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Distance between centers
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                
                # Approximate minimum distance required for disjoint hexagons
                # For two unit hexagons, the minimum distance between centers
                # depends on relative rotation. 
                # Max distance from center to boundary is 1 (vertex).
                # Min distance from center to boundary is sqrt(3)/2 (side midpoint).
                # A safe lower bound for disjointness is sqrt(3)/2 + sqrt(3)/2 = sqrt(3) ~ 1.732
                # But if rotated 30 deg, one vertex points to the other's side.
                # To be robust, we use a slightly larger threshold or check vertices.
                # For speed in SA, we use a conservative threshold.
                # If d < 1.7, they definitely overlap or are very close.
                # Let's use a soft penalty that kicks in if d < 1.8.
                # However, exact disjointness is hard to check fast.
                # We will rely on the final verification.
                # For SA, we want to guide them apart.
                # Let's use a threshold of 1.75.
                if d < 1.75:
                    penalty += (1.75 - d) ** 2
        return penalty

    # Simulated Annealing
    # Initial state: random positions in a box
    # Box size: Let's estimate. 12 hexagons area = 12 * 3 * sqrt(3) / 2 = 18 * 1.732 = 31.17
    # Outer hexagon area = 3 * sqrt(3) / 2 * S^2
    # If S=3, Area = 3 * 1.732 / 2 * 9 = 23.38. Too small.
    # If S=4, Area = 3 * 1.732 / 2 * 16 = 41.56.
    # So S is likely between 3 and 4. Let's say 3.5.
    # Bounding box for centers should be within the outer hexagon.
    # Let's start with a box of size 6x6 centered at 0.
    
    def random_state():
        centers = np.random.uniform(-3, 3, size=(N, 2))
        # Angles: 0 or 30 degrees are the main symmetries, but let's allow any.
        # Actually, for packing, 0 and 30 are most efficient.
        # Let's initialize with 0 degrees for all, and let SA change them if needed?
        # Or randomize between 0 and 30.
        angles = np.random.choice([0.0, 30.0], size=N)
        return centers, angles

    best_centers, best_angles = random_state()
    best_side, best_center, best_theta = compute_minimal_outer_side(best_centers, best_angles)
    best_penalty = get_overlap_penalty(best_centers, best_angles)
    
    current_centers, current_angles = best_centers.copy(), best_angles.copy()
    current_penalty = best_penalty
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    max_steps = 10000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        new_centers = current_centers.copy()
        new_angles = current_angles.copy()
        
        # Move one hexagon
        idx = np.random.randint(N)
        move_size = 0.5 * (1.0 - step / max_steps) + 0.1
        new_centers[idx, 0] += np.random.normal(0, move_size)
        new_centers[idx, 1] += np.random.normal(0, move_size)
        
        # Rotate one hexagon (with low probability)
        if np.random.rand() < 0.1:
            rot_idx = np.random.randint(N)
            new_angles[rot_idx] = np.random.uniform(0, 180)
            
        # Evaluate
        new_penalty = get_overlap_penalty(new_centers, new_angles)
        
        # Calculate cost: side length + large penalty for overlap
        # We want to minimize side length, but avoid overlap.
        # If penalty > 0, we assume it's invalid for now, but SA can accept it to escape local minima.
        # However, for the "side" calculation, we need a valid config.
        # We will only compute the expensive side length if the penalty is low enough
        # or if we are accepting a worse penalty to explore.
        
        # Heuristic: If new_penalty is 0, compute side. Else, use a proxy cost.
        # Proxy cost: penalty * 100 + estimated_side
        # Estimated side can be approx by bounding box of centers + 2.
        # But let's just use penalty as the primary driver for invalid states.
        
        if new_penalty == 0.0:
            # Compute actual side length (expensive)
            # To save time, we might skip this every step?
            # No, we need to know if it's better.
            # But compute_minimal_outer_side is ~1ms. 10000 steps * 1ms = 10s. OK.
            new_side, _, _ = compute_minimal_outer_side(new_centers, new_angles)
            new_cost = new_side
            # If it's valid and better, accept
            if new_cost < best_side:
                best_side = new_cost
                best_centers = new_centers.copy()
                best_angles = new_angles.copy()
                
            # Acceptance criteria for SA
            # If current is valid, we want to minimize side.
            # If current is invalid, we want to minimize penalty.
            
            # Let's define a unified cost function for SA acceptance
            # Cost = side_length if valid, else 100 + penalty
            if current_penalty == 0.0:
                current_cost = compute_minimal_outer_side(current_centers, current_angles)[0]
            else:
                current_cost = 100.0 + current_penalty
                
            if new_cost < current_cost:
                accept = True
            else:
                delta = new_cost - current_cost
                if np.random.rand() < np.exp(-delta / T):
                    accept = True
                else:
                    accept = False
                    
            if accept:
                current_centers = new_centers
                current_angles = new_angles
                current_penalty = new_penalty
        else:
            # Invalid state
            # Compare penalties
            if current_penalty == 0.0:
                current_cost = compute_minimal_outer_side(current_centers, current_angles)[0]
            else:
                current_cost = 100.0 + current_penalty
                
            new_cost = 100.0 + new_penalty
            
            if new_cost < current_cost:
                accept = True
            else:
                delta = new_cost - current_cost
                if np.random.rand() < np.exp(-delta / T):
                    accept = True
                else:
                    accept = False
                    
            if accept:
                current_centers = new_centers
                current_angles = new_angles
                current_penalty = new_penalty

        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final verification and refinement
    # Check if best is valid
    if get_overlap_penalty(best_centers, best_angles) > 0:
        # If best is invalid, try to fix it or use a known valid config
        # Fallback to a simple grid-like arrangement
        # 2 rows of 6?
        # Hexagon side 1.
        # Width of 1 hex: 2. Height: sqrt(3).
        # Stacking:
        # Row 1: y=0, x = -2.5, -1.5, -0.5, 0.5, 1.5, 2.5
        # Row 2: y=sqrt(3), x = -2.0, -1.0, 0.0, 1.0, 2.0, 3.0? No, offset by 0.5
        # Let's try a honeycomb pattern.
        centers = []
        angles = []
        # 4 rows of 3?
        # Or 3 rows of 4?
        # Let's just place them in a circle or grid and let the outer hex fit them.
        # A simple valid config: 12 hexagons in a 2x6 grid?
        # Spacing: x: 1.8, y: 1.6
        for i in range(6):
            for j in range(2):
                x = (i - 2.5) * 1.8
                y = (j - 0.5) * 1.6
                centers.append([x, y])
                angles.append(0.0)
        best_centers = np.array(centers)
        best_angles = np.array(angles)
        
        # Re-verify
        if get_overlap_penalty(best_centers, best_angles) > 0:
             # Even grid might overlap if spacing is tight.
             # Increase spacing.
             for i in range(6):
                 for j in range(2):
                     x = (i - 2.5) * 2.0
                     y = (j - 0.5) * 1.8
                     centers[j*6+i] = [x, y]
             best_centers = np.array(centers)
             
    # Compute final outer hexagon
    final_side, final_center, final_theta = compute_minimal_outer_side(best_centers, best_angles)
    
    inner = [[best_centers[i, 0], best_centers[i, 1], best_angles[i]] for i in range(N)]
    outer_center = tuple(final_center)
    outer_side = final_side
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
