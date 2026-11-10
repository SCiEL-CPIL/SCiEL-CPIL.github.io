import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590  # 10s margin

    n = 11
    # Initial guess: Compact cluster
    # 1 center, 6 around, 4 more
    # Using a hexagonal grid arrangement as start
    r = 1.0 # side length
    # Coordinates for a tight packing of 11 hexagons
    # Center at 0,0
    # 6 neighbors at distance 2*sqrt(3)/2 * 1? No, distance between centers of touching hexagons is 2*apothem = sqrt(3).
    # Apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    
    # Let's use a known good initial configuration
    # 1 at (0,0)
    # 6 at distance sqrt(3) at angles 0, 60, ...
    # 4 more at distance 2*sqrt(3)?
    
    init_pos = []
    init_pos.append([0, 0])
    for i in range(6):
        angle = np.deg2rad(60 * i)
        init_pos.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    # 4 more, maybe at 2*sqrt(3) at 30, 90, 150, 210?
    for i in [1, 3, 5, 7]: # 30, 90, 150, 210
        angle = np.deg2rad(60 * i)
        init_pos.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        
    # Ensure we have 11
    while len(init_pos) < 11:
        init_pos.append([0, 0])
        
    x0 = np.array(init_pos).flatten()

    def get_vertices(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        verts = []
        for p in pos:
            # Hexagon with side 1, angle 0
            # Vertices at angles 0, 60, ...
            angles = np.deg2rad(np.arange(0, 360, 60))
            vs = p + np.column_stack([np.cos(angles), np.sin(angles)])
            verts.append(vs)
        return np.vstack(verts)

    def calc_min_side(pos_flat):
        verts = get_vertices(pos_flat)
        # Check orientation 0
        # Normals at 30, 90, 150, 210, 270, 330
        angles = np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_width = 0
        for n in normals:
            proj = verts @ n
            w = np.max(proj) - np.min(proj)
            if w > max_width:
                max_width = w
        s0 = max_width / np.sqrt(3)
        
        # Check orientation 30 (normals at 0, 60, ...)
        angles = np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_width = 0
        for n in normals:
            proj = verts @ n
            w = np.max(proj) - np.min(proj)
            if w > max_width:
                max_width = w
        s30 = max_width / np.sqrt(3)
        
        return min(s0, s30)

    def check_overlap_gap(pos1, pos2):
        # pos1, pos2 are (2,) arrays
        # Get vertices
        angles = np.deg2rad(np.arange(0, 360, 60))
        v1 = pos1 + np.column_stack([np.cos(angles), np.sin(angles)])
        v2 = pos2 + np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Normals for v1
        n1 = np.column_stack([np.cos(angles + np.pi/6), np.sin(angles + np.pi/6)])
        # Normals for v2 (same)
        n2 = n1
        
        all_normals = np.vstack([n1, n2])
        
        min_gap = np.inf
        for n in all_normals:
            p1 = v1 @ n
            p2 = v2 @ n
            gap = np.min(p2) - np.max(p1)
            if gap < 0:
                # Check other direction
                gap2 = np.min(p1) - np.max(p2)
                gap = max(gap, gap2) # If both negative, they overlap significantly?
                # Actually, if they overlap, one of these will be negative.
                # The gap is the minimum distance.
                # If min(p2) < max(p1) and min(p1) < max(p2), they overlap in this projection.
                # The "gap" is 0 if they touch, negative if overlap?
                # Let's define gap as the minimum distance between the two sets of projections.
                # If intervals [min1, max1] and [min2, max2] overlap, gap is 0?
                # For penalty, we want to know how much they overlap.
                # Overlap amount = max(0, max1 - min2) + max(0, max2 - min1) ? No.
                # Simplest: if they overlap, return a negative value indicating overlap depth.
                # Depth = min(max1, max2) - max(min1, min2)
                # If depth > 0, they overlap.
                depth = min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2))
                if depth > 0:
                    # They overlap in this direction
                    # The actual geometric overlap might be less, but this is a lower bound on separation
                    # We want to penalize this.
                    # Return -depth
                    return -depth
            else:
                # Separated
                if gap < min_gap:
                    min_gap = gap
        # If we get here, no overlap found in any normal?
        # This should not happen if they are convex and we checked all normals.
        # If they don't overlap, min_gap is the minimum distance.
        return min_gap

    def penalty(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                gap = check_overlap_gap(pos[i], pos[j])
                if gap < 0:
                    pen += gap * gap
        return pen

    def objective(pos_flat):
        if time.time() > deadline:
            return np.inf
        side = calc_min_side(pos_flat)
        pen = penalty(pos_flat)
        # We want to minimize side, but penalize overlaps heavily
        # If pen is large, the configuration is invalid.
        # We can use a large multiplier.
        return side + 10.0 * pen

    best_result = None
    best_val = np.inf
    
    # Run optimization
    # L-BFGS-B is good
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
    
    if res.fun < best_val:
        best_val = res.fun
        best_result = res.x
        
    # Try a few random restarts if time permits
    restarts = 5
    for _ in range(restarts):
        if time.time() > deadline - 60:
            break
        # Perturb initial guess
        x0_perturbed = x0 + np.random.normal(0, 0.5, size=x0.shape)
        res = minimize(objective, x0_perturbed, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x

    if best_result is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    pos = best_result.reshape(-1, 2)
    side = calc_min_side(best_result)
    
    # Center the inner hexagons at the origin for the outer hexagon
    # The outer hexagon should be centered at the centroid of the inner hexagons
    centroid = np.mean(pos, axis=0)
    pos_centered = pos - centroid
    
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = pos_centered[:, 0]
    inner_hex_data[:, 1] = pos_centered[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = side

    return inner_hex_data, outer_hex_data, outer_hex_side_length

