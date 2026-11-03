"""Hexagon packing: lattice-shape enumeration + angle-only optimization."""
N = 12
import numpy as np
import time
from itertools import product

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.radians(angle_deg)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    # unit hexagon vertices (flat-top reference at angle 0)
    base = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
                     [-1.0, 0.0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    # rotate and scale
    R = np.array([[cos, -sin], [sin, cos]])
    verts = (base @ R.T) * side
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def _outer_side_from_points(points, outer_angle_deg):
    """Given an array of points (M,2), compute the minimum side length of a
    regular hexagon centered at origin with given rotation that contains all points."""
    # Outer hex face normals: 6 directions
    theta = np.radians(outer_angle_deg)
    # Face normals at theta + pi/6 + k*pi/3
    normals = np.array([[np.cos(theta + np.pi/6 + k*np.pi/3),
                         np.sin(theta + np.pi/6 + k*np.pi/3)] for k in range(6)])
    # For each point, max projection onto each normal
    # points: (M,2), normals: (6,2)
    projections = points @ normals.T  # (M, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    max_inradius = np.max(np.abs(max_proj))
    return 2.0 * max_inradius / np.sqrt(3)

def _compute_outer_side(positions, angles_deg, outer_angle_deg):
    """Compute outer hex side length for given inner positions and angles.
    positions: (12,2), angles_deg: (12,)
    """
    # All vertices of all 12 hexagons: (72, 2)
    angle_rad = np.radians(angles_deg)
    cos = np.cos(angle_rad)[:, None]
    sin = np.sin(angle_rad)[:, None]
    # base hex vertices (6,2)
    base = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
                     [-1.0, 0.0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    # Rotate each: (12, 6, 2)
    R00 = cos
    R01 = -sin
    R10 = sin
    R11 = cos
    # base: (6,2) -> broadcast to (12,6,2)
    bx = base[None, :, 0]  # (1,6)
    by = base[None, :, 1]  # (1,6)
    vx = R00 * bx + R01 * by  # (12,6)
    vy = R10 * bx + R11 * by  # (12,6)
    # Add positions
    vx += positions[:, 0:1]  # (12,1)
    vy += positions[:, 1:2]  # (12,1)
    # Flatten to (72,2)
    all_pts = np.column_stack([vx.ravel(), vy.ravel()])
    return _outer_side_from_points(all_pts, outer_angle_deg)

def _generate_shapes(n_cells=12, max_radius=4):
    """Generate all connected n-cell shapes on hex grid (free polyhexes up to translation).
    Returns list of lists of (q,r) axial coordinates, normalized so min q,r = (0,0)."""
    def neighbors(q, r):
        return [(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)]
    
    def normalize(cells):
        cells = sorted(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        return tuple((c[0]-min_q, c[1]-min_r) for c in cells)
    
    seen = set()
    shapes = []
    
    # BFS: start from single cell, grow by adding adjacent cells
    # Use a queue of (frozen_set, normalized_tuple)
    queue = [frozenset({(0,0)})]
    seen.add(normalize([(0,0)]))
    
    while queue:
        current = queue.pop(0)
        if len(current) == n_cells:
            norm = normalize(list(current))
            if norm not in seen:
                seen.add(norm)
                shapes.append(list(norm))
            continue
        if len(current) > n_cells:
            continue
        # Try adding each neighbor
        for cell in current:
            for nb in neighbors(*cell):
                if nb not in current:
                    new_set = frozenset(current | {nb})
                    norm = normalize(list(new_set))
                    if norm not in seen:
                        seen.add(norm)
                        queue.append(new_set)
    
    return shapes

def _shape_bounding_hex_side(cells, angles_deg, outer_angle_deg):
    """Compute outer side for a lattice shape with given angles.
    cells: list of (q,r), angles_deg: (12,)
    """
    sqrt3 = np.sqrt(3)
    positions = np.array([[1.5*q, sqrt3*(r + q/2.0)] for q, r in cells])
    # Center the shape
    positions -= positions.mean(axis=0)
    return _compute_outer_side(positions, angles_deg, outer_angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Generate all connected 12-cell shapes
    shapes = _generate_shapes(12)
    
    best_side = np.inf
    best_data = None
    
    # For each shape, optimize angles and outer rotation
    for shape_idx, cells in enumerate(shapes):
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Coarse search - try symmetric angle assignments
        # All angles = 0 (lattice-aligned)
        for outer_a in [0, 15, 30, 45, 60, 75, 90]:
            angles = np.zeros(12)
            s = _shape_bounding_hex_side(cells, angles, outer_a)
            if s < best_side:
                best_side = s
                best_data = (cells, angles.copy(), outer_a)
        
        # Try angles = 30 (pointy-top)
        for outer_a in [0, 15, 30, 45, 60, 75, 90]:
            angles = np.full(12, 30.0)
            s = _shape_bounding_hex_side(cells, angles, outer_a)
            if s < best_side:
                best_side = s
                best_data = (cells, angles.copy(), outer_a)
        
        if time.time() > deadline - 5:
            break
        
        # Phase 2: Local optimization of angles + outer angle
        if best_data is not None and best_data[0] is cells:
            x0 = np.concatenate([best_data[1], [best_data[2]]])
        else:
            x0 = np.concatenate([np.zeros(12), [0.0]])
        
        def obj(x):
            angles = x[:12]
            outer_a = x[12]
            return _shape_bounding_hex_side(cells, angles, outer_a)
        
        # Nelder-Mead on 13 params
        from scipy.optimize import minimize
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.001,
                                   'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_data = (cells, res.x[:12].copy(), res.x[12])
        except Exception:
            pass
        
        # Phase 3: Refine with finer grid if time allows
        if time.time() > deadline - 3:
            break
        
        if best_data is not None and best_data[0] is cells:
            # Fine grid around best
            base_angles = best_data[1].copy()
            base_outer = best_data[2]
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(12):
                    for da in [-5, -2, 2, 5]:
                        trial = base_angles.copy()
                        trial[i] += da
                        s = _shape_bounding_hex_side(cells, trial, base_outer)
                        if s < best_side - 1e-6:
                            best_side = s
                            base_angles = trial
                            improved = True
                for da in [-5, -2, 2, 5]:
                    s = _shape_bounding_hex_side(cells, base_angles, base_outer + da)
                    if s < best_side - 1e-6:
                        best_side = s
                        base_outer += da
                        improved = True
                if improved:
                    best_data = (cells, base_angles.copy(), base_outer)
    
    # Build result from best_data
    if best_data is None:
        # Fallback: simple hex cluster
        sqrt3 = np.sqrt(3)
        cells = [(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1),(-1,-1),(0,-1),(1,-1),(0,2),(1,2)]
        angles = np.zeros(12)
        outer_a = 0.0
        positions = np.array([[1.5*q, sqrt3*(r + q/2.0)] for q, r in cells])
        positions -= positions.mean(axis=0)
        best_side = _compute_outer_side(positions, angles, outer_a)
        best_data = (cells, angles, outer_a)
    
    cells, angles, outer_a = best_data
    sqrt3 = np.sqrt(3)
    positions = np.array([[1.5*q, sqrt3*(r + q/2.0)] for q, r in cells])
    positions -= positions.mean(axis=0)
    
    inner = [[float(positions[i,0]), float(positions[i,1]), float(angles[i])] for i in range(12)]
    return inner, (0.0, 0.0), float(best_side), float(outer_a)
