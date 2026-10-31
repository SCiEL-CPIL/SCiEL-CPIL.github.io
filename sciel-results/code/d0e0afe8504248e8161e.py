import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3.0)
    
    # Axial to Cartesian for flat-top hexagons
    # Neighbors at distance sqrt(3) apart
    def axial_to_xy(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate polyhexes of size N
    def canonical(cells):
        min_cell = min(cells, key=lambda c: (c[0], c[1]))
        translated = [(c[0] - min_cell[0], c[1] - min_cell[1]) for c in cells]
        return tuple(sorted(translated))
    
    seen_canonical = set()
    all_shapes = []
    
    def grow(cells):
        if time.time() > deadline - 10:
            return
        if len(cells) == N:
            canon = canonical(cells)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                all_shapes.append(canon)
            return
        
        cell_set = set(cells)
        for c in cells:
            q, r = c
            for dq, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                nc = (q+dq, r+dr)
                if nc not in cell_set:
                    grow(cells + [nc])
                    if time.time() > deadline - 10:
                        return
    
    grow([(0,0)])
    
    # Fallback shapes if generation was too slow
    if len(all_shapes) < 1:
        # 3-4-3 plus one
        cells = [(q, -1) for q in range(-1, 2)] + [(q, 0) for q in range(-2, 2)] + [(q, 1) for q in range(-1, 2)] + [(-2, 1)]
        all_shapes = [canonical(cells)]
    
    def compute_min_hex(points_xy, n_rotations=144):
        """Compute minimum enclosing regular hexagon side length and optimal rotation."""
        pts = np.array(points_xy, dtype=np.float64)
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        
        best_side = float('inf')
        best_rot = 0.0
        
        for i in range(n_rotations):
            theta = 2.0 * np.pi * i / n_rotations
            # 6 normal directions of the hexagon
            angles = theta + np.arange(6) * np.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
            
            # Max projection in each normal direction
            projections = pts_c @ normals.T  # (n_pts, 6)
            max_proj = projections.max(axis=0)  # (6,)
            min_proj = projections.min(axis=0)  # (6,)
            
            # For a hexagon centered at centroid, we need the hexagon to contain all points
            # The constraint: for each normal n_i, max_j dot(p_j, n_i) <= apothem
            # and min_j dot(p_j, n_i) >= -apothem
            # So apothem >= max(max_proj, -min_proj)
            required_apothem = np.maximum(max_proj, -min_proj).max()
            if required_apothem < 0:
                required_apothem = 0.0
            side = required_apothem * 2.0 / sqrt3
            if side < best_side:
                best_side = side
                best_rot = theta
        
        return best_side, centroid, best_rot
    
    best_result = None
    best_side = float('inf')
    
    for shape in all_shapes:
        if time.time() > deadline - 5:
            break
        
        cells_xy = [axial_to_xy(q, r) for q, r in shape]
        side, center, rot = compute_min_hex(cells_xy)
        
        if side < best_side:
            best_side = side
            best_result = (cells_xy, center, rot, side)
    
    if best_result is None:
        # Fallback
        cells_xy = [axial_to_xy(q, r) for q, r in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(1,1),(-2,0),(0,2)]]
        center = np.array([0.0, 0.0])
        best_result = (cells_xy, center, 0.0, 10.0)
    
    cells_xy, center, rot, side = best_result
    
    # Inner hexagons: (x, y, angle_degrees)
    inner = [[x, y, 0.0] for x, y in cells_xy]
    
    # Outer: center (x,y), side length, rotation (degrees)
    outer_center = (float(center[0]), float(center[1]))
    outer_side = float(side)
    outer_rotation = float(np.degrees(rot))
    
    return inner, outer_center, outer_side, outer_rotation

# EVOLVE-BLOCK-END
