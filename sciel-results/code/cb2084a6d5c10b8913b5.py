"""Packing 12 unit regular hexagons in a minimal regular hexagon."""
import numpy as np
import time
from itertools import product

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _hex_in_hexagon(inner_verts, outer_center, outer_side, outer_angle):
    """Check if all inner hex vertices are inside the outer hexagon."""
    oc, os_, oa = outer_center, outer_side, outer_angle
    # Outer hexagon vertices
    outer_verts = _hex_vertices(oc[0], oc[1], os_, oa)
    # For each inner vertex, check if inside outer hexagon (convex polygon)
    for vx, vy in inner_verts:
        # Check if point is inside convex polygon using cross products
        inside = True
        n = len(outer_verts)
        for i in range(n):
            x1, y1 = outer_verts[i]
            x2, y2 = outer_verts[(i + 1) % n]
            # Cross product to check same side
            cross = (x2 - x1) * (vy - y1) - (y2 - y1) * (vx - x1)
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def _hexagons_disjoint(v1, v2):
    """Check if two hexagons (with same side length 1) are disjoint (no overlap).
    Two unit regular hexagons are disjoint if their centers are far enough apart
    considering their orientations. We use a conservative check: if the distance
    between centers is greater than 2*side, they're definitely disjoint.
    For tighter check, we'd need full polygon intersection."""
    cx1, cy1 = v1[0], v1[1]
    cx2, cy2 = v2[0], v2[1]
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    # Two unit hexagons have inradius = sqrt(3)/2, circumradius = 1
    # Minimum distance for disjointness depends on orientation
    # Conservative: if dist > 2.0, definitely disjoint
    # If dist < sqrt(3) - small, might overlap
    return dist > 2.0 - 1e-6

def _max_dist_to_center(verts, center):
    """Maximum distance from center to any vertex."""
    return np.max(np.sqrt((verts[:, 0] - center[0])**2 + (verts[:, 1] - center[1])**2))

def _compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    For a regular hexagon, the side length is determined by the maximum 'radius' 
    in the hexagonal norm."""
    # For a regular hexagon with side s and angle, a point (x,y) relative to center
    # is inside if max of 6 directional distances <= s * sqrt(3)/2 (apothem)
    # Actually, for regular hexagon with side s, the apothem (inradius) = s * sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes.
    
    # Let's compute the required side length by checking all inner vertices
    all_verts = []
    for pos, ang in zip(inner_positions, inner_angles):
        v = _hex_vertices(pos[0], pos[1], 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # For outer hexagon centered at outer_center with angle outer_angle,
    # find minimum side s such that all points are inside.
    # The distance from center to each side (apothem) must be >= s*sqrt(3)/2
    # For a point p, the distance to each of the 6 sides must be >= 0
    # The required apothem is max over all points and all 6 directions of the projection
    
    oa = np.radians(outer_angle)
    # The 6 outward normals of the outer hexagon
    normals = []
    for i in range(6):
        theta = oa + i * np.pi / 3.0 + np.pi / 6.0  # normals are at 30 deg offset
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    # For each point, compute max projection along each normal
    # The apothem needed is max over all points of max over normals of dot(p-center, normal)
    max_proj = 0.0
    for p in all_verts:
        rel = p - np.array(outer_center)
        for n in normals:
            proj = np.dot(rel, n)
            if proj > max_proj:
                max_proj = proj
    
    # apothem = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    s = 2.0 * max_proj / np.sqrt(3.0)
    return s

def _is_valid_configuration(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    """Check if configuration is valid: all inner hexagons inside outer, and disjoint."""
    # Check all inner hexagons are inside outer
    for pos, ang in zip(inner_positions, inner_angles):
        v = _hex_vertices(pos[0], pos[1], 1.0, ang)
        if not _hex_in_hexagon(v, outer_center, outer_side, outer_angle):
            return False
    
    # Check pairwise disjointness (conservative)
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            if not _hexagons_disjoint(inner_positions[i], inner_positions[j]):
                return False
    
    return True

def _honeycomb_positions(n):
    """Generate honeycomb-like positions for n unit hexagons.
    In a honeycomb, hexagons are arranged in rows with offset.
    For hexagonal packing, the distance between centers in adjacent rows is sqrt(3)."""
    positions = []
    # Arrange in a roughly hexagonal cluster
    # For 12 hexagons: center + 6 in first ring + 5 in second ring (partial)
    # Or: 3 rows of 4, or 2 rows of 6, etc.
    
    # Try a 3-4-3-2 arrangement or similar
    # Let's try: center at origin, then rings
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexes at distance sqrt(3) in 6 directions
    # Ring 2: 12 hexes at distance 2*sqrt(3) - but we only need 5 more
    
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 hexes
    for i in range(6):
        theta = i * np.pi / 3.0
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        positions.append((x, y))
    
    # Ring 2: need 5 more
    for i in range(5):
        theta = i * np.pi / 3.0 + np.pi / 6.0  # offset by 30 deg
        x = 2 * np.sqrt(3) * np.cos(theta)
        y = 2 * np.sqrt(3) * np.sin(theta)
        positions.append((x, y))
    
    return positions[:n]

def _row_positions(n, cols):
    """Generate row-based positions."""
    positions = []
    row_height = np.sqrt(3)  # distance between rows in hexagonal packing
    for row in range(n // cols + 1):
        for col in range(cols):
            if len(positions) >= n:
                break
            x = col * 2.0 + (row % 2) * 1.0  # offset alternate rows
            y = row * row_height
            positions.append((x, y))
        if len(positions) >= n:
            break
    return positions[:n]

def _optimize_configuration(initial_positions, initial_angles, max_time=5.0):
    """Optimize a configuration to minimize outer hexagon side length."""
    start_time = time.time()
    n = len(initial_positions)
    
    # Parameters: positions (2n) + angles (n) + outer_center (2) + outer_angle (1)
    # Total: 3n + 3 = 39 parameters for n=12
    
    params = np.array([
        *[p[0] for p in initial_positions],
        *[p[1] for p in initial_positions],
        *[a for a in initial_angles],
        0.0,  # outer center x
        0.0,  # outer center y
        0.0   # outer angle
    ])
    
    best_side = float('inf')
    best_params = params.copy()
    
    # Simple gradient-free optimization: coordinate descent with random perturbations
    step_size = 0.5
    n_iter = 0
    
    while time.time() - start_time < max_time and n_iter < 2000:
        n_iter += 1
        
        # Evaluate current
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        angles = [params[2*n + i] for i in range(n)]
        oc = (params[2*n + n], params[2*n + n + 1])
        oa = params[2*n + n + 2]
        
        side = _compute_outer_side(pos, angles, oc, oa)
        
        # Check validity
        if _is_valid_configuration(pos, angles, oc, side, oa):
            if side < best_side:
                best_side = side
                best_params = params.copy()
        
        # Try small perturbations
        improved = False
        for dim in range(len(params)):
            for delta in [step_size, -step_size]:
                trial = params.copy()
                trial[dim] += delta
                pos_t = [(trial[2*i], trial[2*i+1]) for i in range(n)]
                angles_t = [trial[2*n + i] for i in range(n)]
                oc_t = (trial[2*n + n], trial[2*n + n + 1])
                oa_t = trial[2*n + n + 2]
                
                side_t = _compute_outer_side(pos_t, angles_t, oc_t, oa_t)
                if _is_valid_configuration(pos_t, angles_t, oc_t, side_t, oa_t):
                    if side_t < best_side - 1e-6:
                        params = trial
                        best_side = side_t
                        best_params = params.copy()
                        improved = True
                        break
            if improved:
                break
        
        # Reduce step size
        if n_iter % 50 == 0:
            step_size *= 0.8
    
    return best_side, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_side = 40.0
    best_result = None
    
    # Try multiple initial topologies
    topologies = []
    
    # Honeycomb-like
    topologies.append(('honeycomb', _honeycomb_positions(N), [0.0]*N))
    
    # Row-based: 4x3
    topologies.append(('rows4x3', _row_positions(N, 4), [0.0]*N))
    
    # Row-based: 6x2
    topologies.append(('rows6x2', _row_positions(N, 6), [0.0]*N))
    
    # Row-based: 3x4
    topologies.append(('rows3x4', _row_positions(N, 3), [0.0]*N))
    
    # Grid-like: 4x3 with hexagonal offset
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            positions.append((x, y))
    topologies.append(('grid4x3', positions[:N], [0.0]*N))
    
    # 3x4 grid
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            positions.append((x, y))
    topologies.append(('grid3x4', positions[:N], [0.0]*N))
    
    # Linear arrangement (baseline)
    positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
    topologies.append(('linear', positions, [0.0]*N))
    
    for name, positions, angles in topologies:
        if time.time() > deadline:
            break
        
        # Optimize this topology
        side, params = _optimize_configuration(positions, angles, max_time=15.0)
        
        if side < best_side:
            best_side = side
            n = len(positions)
            pos = [(params[2*i], params[2*i+1]) for i in range(n)]
            angs = [params[2*n + i] for i in range(n)]
            oc = (params[2*n + n], params[2*n + n + 1])
            oa = params[2*n + n + 2]
            best_result = (pos, angs, oc, oa)
    
    if best_result is None:
        # Fallback: return seed-like result
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    pos, angs, oc, oa = best_result
    inner = [[pos[i][0], pos[i][1], angs[i]] for i in range(N)]
    return inner, oc, best_side, oa

# EVOLVE-BLOCK-END
