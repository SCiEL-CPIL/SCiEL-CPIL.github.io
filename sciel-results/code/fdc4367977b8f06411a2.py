<<<<<<< SEARCH
import numpy as np

def construct():
    # Use a very small epsilon to ensure disjointness while minimizing side length
    eps = 1e-9
    
    # Centers
    centers = []
    # Origin
    centers.append((0.0, 0.0))
    
    # 6 neighbors at dist sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        
    # 4 more at dist 3 + 2*eps
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    # Max X extent comes from ring 2 hexagons at 30 and 150 degrees?
    # Ring 2 at 30 deg: Center X = (3+2eps)*cos(30) = (3+2eps)*sqrt(3)/2.
    # Rightmost point of that hex: Center X + sqrt(3)/2.
    # Total Max X = (3+2eps)*sqrt(3)/2 + sqrt(3)/2 = (4+2eps)*sqrt(3)/2 = (2+eps)*sqrt(3).
    
    # Max Y extent comes from ring 2 hexagon at 90 deg.
    # Center Y = 3+2eps.
    # Topmost point: Vertex at 90 deg?
    # v1: (cos90, sin90) = (0, 1).
    # So max y offset is 1.
    # Total Max Y = 3+2eps + 1 = 4+2eps.
    
    # Outer hexagon orientation: 30 degrees (flat top/bottom, vertical sides).
    # Width = sqrt(3) * S.
    # Height = 2 * S.
    
    # Constraints:
    # sqrt(3) * S >= 2 * Max X
    # 2 * S >= 2 * Max Y  => S >= Max Y
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    # Add a tiny margin to side to ensure containment
    side += 1e-9
    
    return inner, (0.0, 0.0), side, 30.0
=======
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg):
    """Get vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append((cx + np.cos(a), cy + np.sin(a)))
    return np.array(verts)

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of outer regular hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _polygon_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    # Get all edges
    edges = []
    for v in (v1, v2):
        for i in range(len(v)):
            j = (i + 1) % len(v)
            edges.append(v[j] - v[i])
    
    # Check projection on each edge's normal
    for e in edges:
        normal = np.array([-e[1], e[0]])
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False  # Separated
    
    return True  # Overlapping

def _min_distance(v1, v2):
    """Minimum distance between two convex polygons (0 if overlapping)."""
    if _polygon_overlap(v1, v2):
        return 0.0
    
    # Find minimum distance between any edge pair
    min_dist = float('inf')
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i+1) % len(v1)]
        for j in range(len(v2)):
            p3 = v2[j]
            p4 = v2[(j+1) % len(v2)]
            # Segment-segment distance
            d1 = p2 - p1
            d2 = p4 - p3
            r = p1 - p3
            a = d1 @ d1
            b = d1 @ d2
            c = d2 @ d2
            d = d1 @ r
            e = d2 @ r
            denom = a * c - b * b
            if denom < 1e-15:
                s = 0.0
                t = e / c if c > 1e-15 else 0.0
            else:
                s = (b * e - c * d) / denom
                t = (a * e - b * d) / denom
            s = max(0, min(1, s))
            t = max(0, min(1, t))
            closest1 = p1 + s * d1
            closest2 = p3 + t * d2
            dist = np.linalg.norm(closest1 - closest2)
            min_dist = min(min_dist, dist)
    
    return min_dist

def _compute_side(inner_positions, outer_angle_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    
    # For each inner hexagon, compute the required outer side
    # A regular hexagon with side S and rotation phi has support function:
    # h(phi, theta) = S * max_k cos(theta - phi - k*pi/3)
    # For containment: for all inner vertices, they must be inside outer hex
    
    # Simpler: for a given outer side S, check if all inner vertices are inside
    # Use bisection on S
    
    def is_contained(S):
        outer_verts = _outer_hex_vertices(0, 0, S, outer_angle_deg)
        # Check each inner hexagon vertex
        for cx, cy, ang in inner_positions:
            inner_verts = _hex_vertices(cx, cy, ang)
            for vx, vy in inner_verts:
                # Point in polygon test
                inside = False
                n = len(outer_verts)
                for i in range(n):
                    x1, y1 = outer_verts[i]
                    x2, y2 = outer_verts[(i+1) % n]
                    if ((y1 > vy) != (y2 > vy)):
                        x_intersect = (x2 - x1) * (vy - y1) / (y2 - y1) + x1
                        if vx < x_intersect:
                            inside = not inside
                if not inside:
                    return False
        return True
    
    # Bisection for S
    lo, hi = 3.0, 15.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if is_contained(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def _overlap_penalty(inner_positions):
    """Penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner_positions[i]
            cx2, cy2, a2 = inner_positions[j]
            v1 = _hex_vertices(cx1, cy1, a1)
            v2 = _hex_vertices(cx2, cy2, a2)
            dist = _min_distance(v1, v2)
            if dist < 1e-6:
                penalty += (1e-6 - dist) ** 2
    return penalty

def _objective(params):
    """Objective: minimize outer side + overlap penalty."""
    n_inner = 11
    # params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, outer_angle]
    inner_positions = []
    for i in range(n_inner):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_positions.append((x, y, a))
    
    outer_angle = params[33]
    
    side = _compute_side(inner_positions, outer_angle)
    overlap = _overlap_penalty(inner_positions)
    
    return side + 100 * overlap

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s deadline with 20s margin
    
    # Initial guess: the current solution
    init_positions = []
    init_positions.append((0.0, 0.0, 30.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        init_positions.append((x, y, 30.0))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        init_positions.append((x, y, 30.0))
    
    # Build initial params
    init_params = []
    for x, y, a in init_positions:
        init_params.extend([x, y, a])
    init_params.append(30.0)  # outer angle
    
    best_params = np.array(init_params, dtype=float)
    best_score = _objective(best_params)
    
    # Try optimization with different methods
    for method in ['L-BFGS-B', 'SLSQP']:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                best_params,
                method=method,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
            
            if time.time() > deadline - 20:
                break
                
        except Exception:
            continue
    
    # Extract best solution
    inner = []
    for i in range(11):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    outer_angle = best_params[33]
    side = _compute_side(inner, outer_angle)
    
    # Add small margin
    side += 1e-9
    
    # If optimization failed or made things worse, use fallback
    if side > 10.0:
        # Use original solution
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        for k in range(6):
            angle = np.deg2rad(k * 60)
            x = (np.sqrt(3) + eps) * np.cos(angle)
            y = (np.sqrt(3) + eps) * np.sin(angle)
            centers.append((x, y))
        for k in range(4):
            angle = np.deg2rad(30 + k * 60)
            x = (3.0 + 2*eps) * np.cos(angle)
            y = (3.0 + 2*eps) * np.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        side = 4 + 2*eps + 1e-9
        outer_angle = 30.0
    
    return inner, (0.0, 0.0), side, outer_angle
>>>>>>> REPLACE
