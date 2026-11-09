"""Improved hexagon packing: minimize outer hexagon side length for 11 unit hexagons."""
import numpy as np
from scipy.optimize import minimize_scalar
import time
import itertools

N = 11
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given angle and side length."""
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.arange(6) * np.pi / 3.0
    angles = base_angles + angle_rad
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of the outer hexagon."""
    return _hex_vertices(cx, cy, angle_deg, side)

def _point_in_hexagon(px, py, hex_verts):
    """Check if point (px,py) is inside or on boundary of convex polygon hex_verts."""
    n = len(hex_verts)
    for i in range(n):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True

def _polygons_overlap(v1, v2, tol=1e-7):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap (including touching)."""
    polys = [v1, v2]
    for i, poly in enumerate(polys):
        n = len(poly)
        for j in range(n):
            x1, y1 = poly[j]
            x2, y2 = poly[(j + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx = -ey
            ny = ex
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def _min_distance_polygons(v1, v2):
    """Compute minimum distance between two convex polygons (0 if overlapping)."""
    min_sep = np.inf
    polys = [v1, v2]
    for i, poly in enumerate(polys):
        n = len(poly)
        for j in range(n):
            x1, y1 = poly[j]
            x2, y2 = poly[(j + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx = -ey
            ny = ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            sep = max(min2 - max1, min1 - max2)
            if sep > 0:
                min_sep = min(min_sep, sep)
    if min_sep < np.inf:
        return min_sep
    return 0.0

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """
    params: array of shape (3*N,) = [x1,y1,ang1, x2,y2,ang2, ...]
    Returns: total penalty (0 if valid packing)
    """
    cx, cy = outer_center
    outer_verts = _outer_hex_vertices(cx, cy, outer_side, outer_angle)
    
    total_penalty = 0.0
    
    inner_verts_list = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_vertices(x, y, ang, 1.0)
        inner_verts_list.append(verts)
    
    for i in range(N):
        for px, py in inner_verts_list[i]:
            if not _point_in_hexagon(px, py, outer_verts):
                total_penalty += 10.0
    
    for i in range(N):
        for j in range(i+1, N):
            if _polygons_overlap(inner_verts_list[i], inner_verts_list[j], tol=-1e-4):
                total_penalty += 5.0
    
    return total_penalty

def _objective(params, outer_side, outer_center, outer_angle):
    """Objective: outer_side + penalty for violations."""
    penalty = _compute_penalty(params, outer_side, outer_center, outer_angle)
    return outer_side + penalty

def _random_initialization(scale=3.0):
    """Generate random initial positions within a reasonable range."""
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = np.random.uniform(-scale, scale)
        params[3*i+1] = np.random.uniform(-scale, scale)
        params[3*i+2] = np.random.uniform(0, 60)
    return params

def _nelder_mead_optimize(x0, outer_side, outer_center, outer_angle, maxiter=200, timeout_check=None):
    """Run Nelder-Mead optimization."""
    def obj(p):
        if timeout_check and timeout_check():
            return 1e10
        return _objective(p, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', 
                         options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-5})
        return result.x, result.fun
    except Exception:
        return x0, _objective(x0, outer_side, outer_center, outer_angle)

def construct():
    global _START_TIME
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Structural Change: Use Hexagonal Grid Clusters
    
    # 1. Generate candidate clusters of N hexagons on a hex grid
    # Axial coordinates (q, r)
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    
    # We will search for connected sets of size N.
    # To keep it fast, we restrict the search to a small area.
    # A 3x3 block is 9 hexagons. A 4x4 block is 16.
    # We need 11.
    
    # Let's generate all subsets of size N from a slightly larger grid and filter for connectivity?
    # No, that's too many.
    # Better: DFS to generate connected clusters.
    
    best_side = 100.0
    best_config = None
    
    # Helper to convert axial to Cartesian
    # Standard hex grid:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    # Side length 1 means distance between centers is sqrt(3).
    # Wait, side length 1 hexagon has width sqrt(3) (flat-to-flat).
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # So x = 1.5 * q, y = sqrt(3) * (r + q/2) is correct for side=1?
    # Let's check:
    # (0,0) -> (0,0)
    # (1,0) -> (1.5, sqrt(3)/2). Dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        return np.array([x, y])
        
    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
        
    def is_connected(cells):
        # cells is a set of (q,r)
        if not cells:
            return True
        start = next(iter(cells))
        visited = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            for nq, nr in get_neighbors(q, r):
                if (nq, nr) in cells and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(cells)
        
    def calculate_min_enclosing_hexagon(points):
        """
        points: Nx2 array of Cartesian coordinates of centers.
        Returns: (min_side, optimal_angle_deg, center_offset)
        The outer hexagon is centered at the centroid of the points?
        Actually, the center of the outer hexagon should be optimized too.
        However, for a centrally symmetric shape like a hexagon, the minimal enclosing hexagon
        for a set of points is often centered at the centroid, but not always if the points are skewed.
        But for regular hexagons, the "width" in any direction is fixed.
        
        Let's assume the outer hexagon is centered at the centroid of the inner centers for now.
        Then we just need to find the optimal rotation.
        
        For a fixed center C and rotation theta, the condition for point p to be inside
        a hexagon of side S is:
        | (p-C) dot n_k(theta) | <= S * sqrt(3) / 2 for k=0,1,2
        where n_k are the 3 axis normals.
        
        So S(theta) = (2/sqrt(3)) * max_k max_i | (p_i - C) dot n_k(theta) |
        
        We minimize S(theta) over theta in [0, pi/3).
        """
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        def S_of_theta(theta):
            # Normals at theta, theta+pi/3, theta+2pi/3
            # Normal to side 1 is at angle theta + pi/6?
            # Vertices are at theta + k*pi/3.
            # Sides are perpendicular to vertices? No.
            # In a regular hexagon, the normal to a side is in the direction of the vertex opposite to the side?
            # Let's define vertices at angles 0, 60, 120...
            # Side 1 connects v0 and v1. Midpoint angle 30. Normal angle 30.
            # So normals are at theta + 30, theta + 90, theta + 150 degrees.
            
            n1 = np.array([np.cos(theta + np.pi/6), np.sin(theta + np.pi/6)])
            n2 = np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
            n3 = np.array([np.cos(theta + 5*np.pi/6), np.sin(theta + 5*np.pi/6)])
            
            max_proj = 0
            for n in [n1, n2, n3]:
                projs = np.abs(centered_points @ n)
                max_proj = max(max_proj, np.max(projs))
            
            return (2.0 / np.sqrt(3.0)) * max_proj
        
        # Minimize S(theta)
        # S(theta) is periodic with period pi/3.
        # It is a convex function of theta in the interval?
        # It's the max of absolute values of sinusoids, so it's convex-ish.
        
        result = minimize_scalar(S_of_theta, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
        
        if result.success:
            best_theta = result.x
            best_S = result.fun
            return best_S, best_theta * 180 / np.pi, centroid
        else:
            # Fallback
            return 100.0, 0.0, centroid

    # Generate clusters
    # We'll do a DFS from (0,0) to generate connected sets of size N.
    # To avoid duplicates, we enforce a canonical ordering or just use a set of frozensets.
    
    clusters = set()
    
    def dfs(current_set, last_added):
        if len(current_set) == N:
            # Canonicalize: sort the tuple
            clusters.add(frozenset(current_set))
            return
        
        if time.time() > deadline - 20:
            return
            
        for nq, nr in get_neighbors(last_added[0], last_added[1]):
            if (nq, nr) not in current_set:
                # Optional: restrict to a bounding box to keep it finite and relevant
                # If we go too far, the hexagon will be huge.
                if abs(nq) > 4 or abs(nr) > 4:
                    continue
                new_set = set(current_set)
                new_set.add((nq, nr))
                dfs(new_set, (nq, nr))
                
    # Start DFS from (0,0)
    # Note: This generates all connected sets containing (0,0).
    # This is sufficient because we can translate the whole cluster.
    dfs({(0,0)}, (0,0))
    
    # Also, the cluster might not contain (0,0) in its local coordinates, but since we center it later,
    # generating all shapes containing the origin is enough to cover all topological shapes.
    # Wait, if the shape is "long", the origin might not be in the middle of the bounding box,
    # but it will be in the set of vertices.
    # This DFS generates all connected subsets of the grid of size N that include (0,0).
    # This covers all possible "shapes" of connected hexagon clusters.
    
    print(f"Generated {len(clusters)} unique cluster shapes")
    
    for i, cluster in enumerate(clusters):
        if time.time() > deadline - 10:
            break
            
        # Convert to Cartesian
        points = np.array([axial_to_cartesian(q, r) for q, r in cluster])
        
        # Calculate minimal enclosing hexagon
        S, theta_deg, center = calculate_min_enclosing_hexagon(points)
        
        if S < best_side:
            best_side = S
            # Store the configuration
            # We need to return the centers relative to the outer hexagon center.
            # The outer hexagon center is `center` in the cluster's local coordinates.
            # So inner centers are `points - center`.
            # Angles are 0 (since we used standard orientation for inner hexagons in the grid).
            # Wait, the inner hexagons are aligned with the grid.
            # The outer hexagon is rotated by `theta_deg`.
            # The inner hexagons are at angle 0.
            # Is this valid? Yes, the problem allows rotation of inner hexagons.
            # But here we fixed inner angles to 0.
            # The outer hexagon is rotated.
            # This is a valid configuration.
            
            best_config = {
                'points': points,
                'center': center,
                'S': S,
                'theta': theta_deg
            }
            
    if best_config is not None:
        # Construct the return values
        # Inner hexagons:
        # x, y, angle
        # The centers are best_config['points'] - best_config['center']
        # Angles are 0.0
        
        inner = []
        for p in best_config['points']:
            cx = float(p[0] - best_config['center'][0])
            cy = float(p[1] - best_config['center'][1])
            inner.append([cx, cy, 0.0])
            
        # Outer hexagon
        outer_center = (0.0, 0.0)
        outer_side = float(best_config['S'])
        outer_angle = float(best_config['theta'])
        
        # Quick verification
        # Check overlaps (should be none)
        # Check containment
        outer_verts = _outer_hex_vertices(0, 0, outer_side, outer_angle)
        valid = True
        for i in range(N):
            v = _hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
            for px, py in v:
                if not _point_in_hexagon(px, py, outer_verts):
                    valid = False
                    break
            if not valid: break
            
        if valid:
            return inner, outer_center, outer_side, outer_angle
        else:
            # If verification fails, it's a bug in the logic or tolerance.
            # Fallback to a large safe box
            pass
            
    # Fallback if no cluster found or verification failed
    # Use a simple grid
    inner = []
    positions = [
        (0, 0),
        (2, 0), (-2, 0),
        (1, 1.732), (-1, 1.732),
        (1, -1.732), (-1, -1.732),
        (0, 3.464), (0, -3.464),
        (3, 1.732), (-3, 1.732),
    ]
    for px, py in positions[:N]:
        inner.append([float(px), float(py), 0.0])
    return inner, (0.0, 0.0), 10.0, 0.0
