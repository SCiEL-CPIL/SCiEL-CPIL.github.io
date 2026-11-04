"""
Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.
Strategy:
1. Initialize with a honeycomb-like pattern.
2. Use Nelder-Mead optimization with a penalty-based objective.
3. Use Separating Axis Theorem (SAT) for accurate collision detection and depth.
4. Multi-start with perturbations to escape local minima.
5. Polish the best result found.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # 5s safety margin below 150s
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300])
    base = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in base_angles])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    verts = base @ rot.T * side + np.array([cx, cy])
    return verts

def _get_axes(verts):
    """Get edge normals (axes) for a convex polygon."""
    axes = []
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        # Normal perpendicular to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            axes.append(normal / norm)
    return axes

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return min and max."""
    projections = np.dot(verts, axis)
    return np.min(projections), np.max(projections)

def _min_overlap_depth(v1, v2):
    """
    Calculate the minimum translation distance to separate two convex polygons.
    Returns (depth, direction) where direction is a unit vector from v1 to v2 
    along the axis of minimum overlap. If separated, returns (0, None).
    """
    axes1 = _get_axes(v1)
    axes2 = _get_axes(v2)
    all_axes = axes1 + axes2
    
    min_overlap = float('inf')
    min_axis = None
    
    for axis in all_axes:
        min1, max1 = _project_polygon(v1, axis)
        min2, max2 = _project_polygon(v2, axis)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            return 0.0, None  # Separated
        
        # Calculate overlap depth
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis
    
    if min_overlap == 0.0 or min_axis is None:
        return 0.0, None
        
    # Determine direction: from v1 to v2
    center1 = np.mean(v1, axis=0)
    center2 = np.mean(v2, axis=0)
    if np.dot(center2 - center1, min_axis) < 0:
        min_axis = -min_axis
        
    return min_overlap, min_axis

def _distance_to_outer(v1, outer_verts):
    """
    Calculate how far the polygon v1 is outside the outer polygon.
    Returns a penalty value (0 if inside, >0 if outside).
    For simplicity, we check if all vertices of v1 are inside outer_verts.
    If not, we estimate the penetration depth.
    """
    # Check if all vertices of v1 are inside outer_verts
    # Using winding number or cross product for convex polygon
    n_outer = len(outer_verts)
    for v in v1:
        inside = True
        for i in range(n_outer):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % n_outer]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            # Estimate penetration: distance to nearest edge
            min_dist = float('inf')
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                # Point to line segment distance
                dp = p2 - p1
                t = np.dot(v - p1, dp) / np.dot(dp, dp)
                t = max(0, min(1, t))
                proj = p1 + t * dp
                dist = np.linalg.norm(v - proj)
                min_dist = min(min_dist, dist)
            return min_dist
    return 0.0

def _objective(params):
    """
    Objective: Minimize outer side length + penalties.
    params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, S, outer_angle]
    """
    n = N
    pos = params[0:2*n].reshape(n, 2)
    angles = params[2*n:3*n]
    cx, cy = params[3*n:3*n+2]
    S = params[3*n+2]
    outer_angle = params[3*n+3]
    
    if S < 2.0: S = 2.0
    
    # Build inner hexagons
    inner_hexes = []
    for i in range(n):
        v = _hex_vertices(pos[i][0], pos[i][1], 1.0, angles[i])
        inner_hexes.append(v)
        
    # Build outer hexagon
    outer_verts = _hex_vertices(cx, cy, S, outer_angle)
    
    penalty = 0.0
    
    # Containment penalty
    for v in inner_hexes:
        pen = _distance_to_outer(v, outer_verts)
        if pen > 0:
            penalty += 100.0 * pen
            
    # Collision penalty
    for i in range(n):
        for j in range(i+1, n):
            # Quick check
            dist_c = np.linalg.norm(inner_hexes[i].mean(axis=0) - inner_hexes[j].mean(axis=0))
            if dist_c > 4.0:
                continue
            depth, _ = _min_overlap_depth(inner_hexes[i], inner_hexes[j])
            if depth > 0:
                penalty += 100.0 * depth
                
    return S + penalty

def _is_valid(inner_hexes, center, side, angle_deg):
    """Check if configuration is strictly valid."""
    outer_verts = _hex_vertices(center[0], center[1], side, angle_deg)
    
    # Containment
    for v in inner_hexes:
        if _distance_to_outer(v, outer_verts) > 1e-6:
            return False
            
    # Collision
    for i in range(len(inner_hexes)):
        for j in range(i+1, len(inner_hexes)):
            depth, _ = _min_overlap_depth(inner_hexes[i], inner_hexes[j])
            if depth > 1e-6:
                return False
    return True

def _initial_config_honeycomb():
    """
    Generate a honeycomb-like initial configuration.
    """
    positions = []
    angles = [0.0] * N
    
    # Honeycomb lattice vectors for unit hexagons
    # Distance between centers in honeycomb is sqrt(3) for edge-to-edge, but we need disjoint.
    # Let's use a grid with spacing slightly larger than min distance.
    # Min distance between centers of disjoint unit hexagons is sqrt(3) (if aligned) or 2 (if rotated 30 deg).
    # Let's use spacing 2.2 to be safe.
    
    dx = 2.2
    dy = 2.2 * np.sqrt(3) / 2 # ~1.9
    
    # Center the cluster
    # 4 rows, 3 cols roughly
    rows = 4
    cols = 3
    
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= N:
                break
            x = (c - 1) * dx + (0.5 * dx if r % 2 else 0)
            y = (r - 1.5) * dy
            positions.append((x, y))
            
    # Fill remaining if any
    while len(positions) < N:
        positions.append((0.0, 0.0))
        
    inner = [[x, y, 0.0] for x, y in positions[:N]]
    
    # Estimate outer side
    all_pts = []
    for hx, hy, ha in inner:
        verts = _hex_vertices(hx, hy, 1.0, ha)
        all_pts.extend(verts)
    all_pts = np.array(all_pts)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    # Approximate side length for hexagon containing this
    # For a hexagon with side S, width is 2S, height is sqrt(3)S (if pointy top)
    # Or width sqrt(3)S, height 2S (if flat top).
    # Let's use a generous estimate
    side = max(width, height) / np.sqrt(3) + 1.0
    center = ((min_x + max_x)/2, (min_y + max_y)/2)
    
    return inner, center, side, 0.0

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_inner = None
    best_center = None
    best_side = float('inf')
    best_angle = 0.0
    
    # Initial config
    inner, center, side, angle_deg = _initial_config_honeycomb()
    
    # Prepare initial parameters
    x0_base = np.array([[inner[i][0], inner[i][1]] for i in range(N)]).flatten()
    angles0 = np.array([inner[i][2] for i in range(N)])
    x0_base = np.concatenate([x0_base, angles0, [center[0], center[1], side, angle_deg]])
    
    # Multi-start optimization
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for k in range(num_restarts):
        if _remaining() < 20.0:
            break
            
        # Perturb initial guess
        if k == 0:
            x0 = x0_base.copy()
        else:
            x0 = x0_base + rng.normal(0, 0.5, size=x0_base.shape)
            # Keep side length reasonable
            x0[3*N+2] = max(x0[3*N+2], 3.0)
            
        try:
            # Use Nelder-Mead for robustness
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12})
            
            if result.success or result.fun < best_side:
                params = result.x
                n = N
                pos = params[0:2*n].reshape(n, 2)
                angles = params[2*n:3*n]
                cx, cy = params[3*n:3*n+2]
                s = max(params[3*n+2], 2.0)
                outer_angle = params[3*n+3]
                
                inner_new = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
                
                # Validate
                if _is_valid(inner_new, (cx, cy), s, outer_angle):
                    if s < best_side:
                        best_inner = inner_new
                        best_center = (cx, cy)
                        best_side = s
                        best_angle = outer_angle
                        
        except Exception as e:
            # print(f"Optimization failed: {e}")
            pass

    # Polish the best result found
    if best_inner is not None and _remaining() > 10.0:
        x0_best = np.array([[best_inner[i][0], best_inner[i][1]] for i in range(N)]).flatten()
        angles_best = np.array([best_inner[i][2] for i in range(N)])
        x0_best = np.concatenate([x0_best, angles_best, [best_center[0], best_center[1], best_side, best_angle]])
        
        try:
            result = minimize(_objective, x0_best, method='Nelder-Mead',
                             options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-14})
            
            params = result.x
            n = N
            pos = params[0:2*n].reshape(n, 2)
            angles = params[2*n:3*n]
            cx, cy = params[3*n:3*n+2]
            s = max(params[3*n+2], 2.0)
            outer_angle = params[3*n+3]
            
            inner_new = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
            
            if _is_valid(inner_new, (cx, cy), s, outer_angle):
                if s < best_side:
                    best_inner = inner_new
                    best_center = (cx, cy)
                    best_side = s
                    best_angle = outer_angle
                    
        except Exception:
            pass

    # Fallback if no valid solution found (should not happen)
    if best_inner is None:
        return inner, center, side, angle_deg

    return best_inner, best_center, best_side, best_angle

# Entry point wrapper to match original signature
def construct():
    return construct_impl()

def construct_impl():
    return construct()
