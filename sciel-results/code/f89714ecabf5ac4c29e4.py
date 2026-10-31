import math
import numpy as np
from scipy.optimize import minimize
import time

N = 11  # number of unit hexagons

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle (degrees) and side length."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def polygon_area(verts):
    """Shoelace formula for polygon area."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def polygons_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    # Get all edges' normal vectors
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            edge = verts[(i+1) % n] - verts[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
        return axes
    
    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    
    for axis in axes1 + axes2:
        # Project both polygons onto this axis
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def point_in_hexagon(pt, cx, cy, angle_deg, side=1.0):
    """Check if point pt is inside a regular hexagon."""
    # Transform to hexagon's local coordinate system
    angle = math.radians(angle_deg)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    dx = pt[0] - cx
    dy = pt[1] - cy
    local_x = cos_a * dx - sin_a * dy
    local_y = sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side s, the distance from center to edge is s*sqrt(3)/2
    # The hexagon can be described as |x| <= s, |y| <= s*sqrt(3)/2, and |x| + |y|/sqrt(3) <= s
    # Actually, for a unit hexagon with flat-top orientation:
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # In local coords, the hexagon is defined by:
    # |local_x| <= side
    # |local_y| <= side * sqrt(3)/2
    # |local_x| + |local_y|/sqrt(3) <= side
    
    s = side
    if abs(local_x) > s + 1e-10:
        return False
    if abs(local_y) > s * math.sqrt(3) / 2 + 1e-10:
        return False
    if abs(local_x) + abs(local_y) / math.sqrt(3) > s + 1e-10:
        return False
    return True

def outer_hex_contains_inner(inner_verts, outer_cx, outer_cy, outer_angle, outer_side):
    """Check if all vertices of inner hex are inside outer hexagon."""
    for pt in inner_verts:
        if not point_in_hexagon(pt, outer_cx, outer_cy, outer_angle, outer_side):
            return False
    return True

def smooth_penalty_for_overlap(v1, v2, beta=100.0):
    """Smooth approximation of overlap detection using minimum distance between polygons."""
    # Compute minimum distance between two polygons
    min_dist = float('inf')
    n1 = len(v1)
    n2 = len(v2)
    
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j+1) % n2]
            # Distance from point to segment
            def point_segment_dist(pt, a, b):
                ab = b - a
                t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                return np.linalg.norm(pt - closest)
            
            d1 = point_segment_dist(p1, q1, q2)
            d2 = point_segment_dist(q1, p1, p2)
            min_dist = min(min_dist, d1, d2)
    
    # If min_dist < 0, they overlap (but our computation gives >= 0)
    # Use smooth penalty: exp(-beta * min_dist) when close
    return math.exp(-beta * min_dist) if min_dist < 0.1 else 0.0

def objective(params, n_inner, fixed_angles=None):
    """
    Objective function: minimize outer side length + penalties for overlaps and containment.
    params: [outer_cx, outer_cy, outer_side, outer_angle, 
             inner_1_x, inner_1_y, inner_1_angle,
             inner_2_x, inner_2_y, inner_2_angle, ...]
    """
    start_time = time.time()
    
    outer_cx = params[0]
    outer_cy = params[1]
    outer_side = params[2]
    outer_angle = params[3]
    
    # Ensure outer_side is positive
    outer_side = max(outer_side, 1.0)
    
    total_penalty = 0.0
    inner_verts_list = []
    
    for i in range(n_inner):
        base_idx = 4 + i * 3
        ix = params[base_idx]
        iy = params[base_idx + 1]
        iangle = params[base_idx + 2]
        
        inner_verts = hex_vertices(ix, iy, iangle, 1.0)
        inner_verts_list.append(inner_verts)
        
        # Check containment: all inner vertices must be in outer hex
        for pt in inner_verts:
            if not point_in_hexagon(pt, outer_cx, outer_cy, outer_angle, outer_side):
                # Penalty for being outside
                # Compute distance to boundary
                total_penalty += 10.0
    
    # Check pairwise overlaps
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                total_penalty += 5.0
    
    # Objective: outer_side + penalty
    return outer_side + total_penalty

def construct():
    """
    Construct placement of N unit hexagons inside a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for each inner hexagon
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation angle of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: place hexagons in a row
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial orientations for outer hexagon
    outer_angle_inits = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    for outer_angle_init in outer_angle_inits:
        if time.time() > deadline:
            break
        
        # Initialize inner hexagons in a hexagonal pattern
        # For N=11, use a 3-4-3 pattern or similar
        if N <= 1:
            inner_init = [[0.0, 0.0, 0.0]]
        elif N <= 7:
            # Central + 6 around
            inner_init = [[0.0, 0.0, 0.0]]
            for i in range(6):
                angle = i * 60.0
                inner_init.append([2.0 * math.cos(math.radians(angle)), 
                                  2.0 * math.sin(math.radians(angle)), 0.0])
            inner_init = inner_init[:N]
        else:
            # For larger N, use a row pattern
            inner_init = []
            # Arrange in rows
            row_lengths = []
            remaining = N
            row = 0
            while remaining > 0:
                if row == 0:
                    l = min(remaining, 3)
                elif row == 1:
                    l = min(remaining, 4)
                else:
                    l = min(remaining, 3)
                row_lengths.append(l)
                remaining -= l
                row += 1
            
            y_offset = 0.0
            for row_idx, row_len in enumerate(row_lengths):
                y = row_idx * math.sqrt(3)
                for col in range(row_len):
                    x = (col - (row_len - 1) / 2.0) * 2.0
                    inner_init.append([x, y, 0.0])
        
        # Build initial params
        params = [0.0, 0.0, 5.0, outer_angle_init]
        for ix, iy, iang in inner_init:
            params.extend([ix, iy, iang])
        
        # Optimize using Nelder-Mead
        result = minimize(objective, params, args=(N,), method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if time.time() > deadline:
            break
        
        score = result.fun
        if score < best_score:
            best_score = score
            # Extract result
            outer_cx = result.x[0]
            outer_cy = result.x[1]
            outer_side = result.x[2]
            outer_angle = result.x[3]
            
            inner = []
            for i in range(N):
                base_idx = 4 + i * 3
                inner.append([result.x[base_idx], result.x[base_idx + 1], result.x[base_idx + 2]])
            
            best_result = (inner, (outer_cx, outer_cy), outer_side, outer_angle)
    
    if best_result is None:
        # Fallback: return a simple valid configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Result: side={result[2]:.4f}, angle={result[3]:.2f}")
    print(f"Center: {result[1]}")
    print(f"Inner hexagons: {len(result[0])}")
