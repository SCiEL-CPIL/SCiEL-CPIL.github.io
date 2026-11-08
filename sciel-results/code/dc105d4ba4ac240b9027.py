"""Improved packing of 12 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s wall-clock

def _hex_vertices(cx, cy, angle_deg=0.0, side=1.0):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + x, cy + y))
    return np.array(verts)

def _hex_face_normals(angle_deg=0.0):
    """Return 6 outward face normals of a regular hexagon at given rotation."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + k * math.pi / 3.0 + math.pi / 6.0
        normals.append((math.cos(theta), math.sin(theta)))
    return np.array(normals)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx, dy = px - cx, py - cy
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    # Transform to hexagon local frame
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Hexagon with flat-top orientation: |x| <= side, |y| <= side*sqrt(3)/2
    # Actually for a regular hexagon with vertices at angles 0, 60, ..., the
    # inscribed radius is side * cos(30°) = side * sqrt(3)/2
    # The hexagon can be defined as: for each of 6 directions, |dot(p, n)| <= inradius
    inradius = side * math.sqrt(3) / 2.0
    for k in range(6):
        theta = a + k * math.pi / 3.0 + math.pi / 6.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = dx * nx + dy * ny
        if abs(proj) > inradius:
            return False
    return True

def _separation(i_verts, j_verts):
    """
    Compute minimum separation between two convex polygons (hexagons).
    Returns positive if disjoint, 0 if touching, negative if overlapping.
    Uses separating axis theorem.
    """
    # For two convex polygons, if they are separated, there exists an axis
    # (normal to an edge of either polygon) along which their projections don't overlap.
    # We want the minimum over all axes of the maximum gap.
    # Actually, we want to find if they overlap and by how much.
    
    # Collect all edge normals from both hexagons
    all_normals = []
    for verts in [i_verts, j_verts]:
        for k in range(6):
            p1 = verts[k]
            p2 = verts[(k + 1) % 6]
            edge = p2 - p1
            # Normal perpendicular to edge
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx * nx + ny * ny)
            if norm > 1e-12:
                all_normals.append((nx / norm, ny / norm))
    
    min_gap = float('inf')
    for nx, ny in all_normals:
        # Project both hexagons onto this axis
        proj_i = i_verts[:, 0] * nx + i_verts[:, 1] * ny
        proj_j = j_verts[:, 0] * nx + j_verts[:, 1] * ny
        min_i, max_i = proj_i.min(), proj_i.max()
        min_j, max_j = proj_j.min(), proj_j.max()
        
        # Gap between projections
        if max_i < min_j:
            gap = min_j - max_i
        elif max_j < min_i:
            gap = min_i - max_j
        else:
            gap = 0.0  # Overlap on this axis
        
        if gap < 0:
            gap = 0.0
        min_gap = min(min_gap, gap)
    
    return min_gap

def _min_enclosing_hexagon(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Find minimum side length of a regular hexagon (with given outer angle)
    that contains all inner hexagons.
    Uses optimization over the center of the outer hexagon.
    """
    def enclosing_side(scaled_params):
        """Given outer center (cx, cy), compute minimum side length needed."""
        cx, cy = scaled_params
        # For each inner hexagon, compute the maximum over its vertices of
        # the support function in each of the 6 outer hexagon directions.
        outer_normals = []
        oa = math.radians(outer_angle_deg)
        for k in range(6):
            theta = oa + k * math.pi / 3.0 + math.pi / 6.0
            outer_normals.append((math.cos(theta), math.sin(theta)))
        outer_normals = np.array(outer_normals)
        
        max_support = 0.0
        for idx in range(N):
            cx_i, cy_i, ang_i = inner_centers[idx][0], inner_centers[idx][1], inner_angles[idx]
            verts = _hex_vertices(cx_i, cy_i, ang_i, 1.0)
            # Compute support in each outer normal direction
            for nx, ny in outer_normals:
                proj = verts[:, 0] * nx + verts[:, 1] * ny
                s = proj.max() - (cx * nx + cy * ny)
                # Also need to consider the opposite direction
                s2 = (cx * nx + cy * ny) - proj.min()
                max_support = max(max_support, s, s2)
        
        # For a regular hexagon, the inradius is side * sqrt(3)/2
        # So side = max_support / (sqrt(3)/2) = 2 * max_support / sqrt(3)
        side = 2.0 * max_support / math.sqrt(3)
        return side

    # Initial guess: center of mass of inner hexagons
    centers_arr = np.array([[c[0], c[1]] for c in inner_centers])
    initial = (centers_arr[:, 0].mean(), centers_arr[:, 1].mean())
    
    result = minimize(enclosing_side, initial, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.success:
        cx, cy = result.x
        side = result.fun
    else:
        cx, cy = initial
        side = enclosing_side(initial)
    
    return cx, cy, side

def _check_validity(inner, outer_center, outer_side, outer_angle):
    """Verify all inner hexagons are disjoint and inside outer hexagon."""
    ocx, ocy = outer_center
    outer_verts = _hex_vertices(ocx, ocy, outer_angle, outer_side)
    
    # Check each inner hexagon is inside outer
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, ang, 1.0)
        inner_verts_list.append(verts)
        for vx, vy in verts:
            if not _point_in_hexagon(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    
    # Check pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            sep = _separation(inner_verts_list[i], inner_verts_list[j])
            if sep < -1e-7:
                return False
    
    return True

def construct():
    start_time = time.time()
    
    # Initial configuration: 12 hexagons in a honeycomb-like arrangement
    # 3 rows of 4, or 2 rows of 6, etc.
    # Let's try a few initial patterns
    
    def make_honeycomb_row():
        """4 hexagons in a row, 3 rows with offset."""
        result = []
        dx = 2.0  # horizontal spacing for touching hexagons
        dy = math.sqrt(3)  # vertical spacing
        for row in range(3):
            for col in range(4):
                x = col * dx + (0.5 * dx if row % 2 == 1 else 0.0)
                y = row * dy
                result.append([x, y, 0.0])
        # Center the arrangement
        xs = [p[0] for p in result]
        ys = [p[1] for p in result]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        for p in result:
            p[0] -= cx
            p[1] -= cy
        return result

    def make_grid_3x4():
        """3 columns, 4 rows."""
        result = []
        dx = 2.0
        dy = math.sqrt(3)
        for row in range(4):
            for col in range(3):
                x = col * dx + (0.5 * dx if row % 2 == 1 else 0.0)
                y = row * dy
                result.append([x, y, 0.0])
        xs = [p[0] for p in result]
        ys = [p[1] for p in result]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        for p in result:
            p[0] -= cx
            p[1] -= cy
        return result

    def make_ring():
        """1 in center, 6 around, 5 more."""
        result = []
        result.append([0.0, 0.0, 0.0])
        for k in range(6):
            theta = k * math.pi / 3.0
            x = 2.0 * math.cos(theta)
            y = 2.0 * math.sin(theta)
            result.append([x, y, 0.0])
        # 5 more around
        for k in range(5):
            theta = (k + 0.5) * 2 * math.pi / 5.0
            r = 3.5
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            result.append([x, y, 0.0])
        return result

    initial_patterns = [make_honeycomb_row(), make_grid_3x4(), make_ring()]
    
    best_score = float('inf')
    best_result = None
    
    for pattern_idx, inner in enumerate(initial_patterns):
        if time.time() > DEADLINE - 30:
            break
        
        # Optimize outer hexagon
        outer_center, outer_side, outer_angle = _min_enclosing_hexagon(
            inner, [p[2] for p in inner], 0.0)
        
        if outer_side < best_score:
            best_score = outer_side
            best_result = (inner, outer_center, outer_side, outer_angle)
        
        # Local refinement: optimize positions, angles, and outer center
        if time.time() > DEADLINE - 45:
            break
        
        # Define optimization variables:
        # For each inner hex: x, y, angle (3 vars * 12 = 36)
        # Outer: cx, cy, angle (3 vars)
        # Total: 39 vars
        
        def objective(params):
            if time.time() > DEADLINE - 5:
                return best_score  # Return best known
            
            # Unpack
            inner_pos = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                ang = params[i * 3 + 2]
                inner_pos.append([x, y, ang])
            ocx = params[N * 3]
            ocy = params[N * 3 + 1]
            oang = params[N * 3 + 2]
            
            # Compute enclosing side
            outer_normals = []
            oa = math.radians(oang)
            for k in range(6):
                theta = oa + k * math.pi / 3.0 + math.pi / 6.0
                outer_normals.append((math.cos(theta), math.sin(theta)))
            outer_normals = np.array(outer_normals)
            
            max_support = 0.0
            inner_verts_list = []
            for i in range(N):
                cx_i, cy_i, ang_i = inner_pos[i]
                verts = _hex_vertices(cx_i, cy_i, ang_i, 1.0)
                inner_verts_list.append(verts)
                for nx, ny in outer_normals:
                    proj = verts[:, 0] * nx + verts[:, 1] * ny
                    s = proj.max() - (ocx * nx + ocy * ny)
                    s2 = (ocx * nx + ocy * ny) - proj.min()
                    max_support = max(max_support, s, s2)
            
            side = 2.0 * max_support / math.sqrt(3)
            
            # Penalty for overlapping inner hexagons
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    sep = _separation(inner_verts_list[i], inner_verts_list[j])
                    if sep < 0.01:
                        penalty += (0.01 - sep) ** 2 * 1000.0
            
            return side + penalty
        
        # Initial parameters
        init_params = []
        for i in range(N):
            init_params.extend(inner[i])
        init_params.extend([outer_center[0], outer_center[1], outer_angle])
        
        result = minimize(objective, init_params, method='L-BFGS-B',
                         options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8})
        
        if time.time() > DEADLINE - 5:
            break
        
        if result.fun < best_score and result.fun < outer_side + 1:
            # Validate
            inner_new = []
            for i in range(N):
                inner_new.append([result.x[i * 3], result.x[i * 3 + 1], result.x[i * 3 + 2]])
            ocx_new = result.x[N * 3]
            ocy_new = result.x[N * 3 + 1]
            oang_new = result.x[N * 3 + 2]
            
            if _check_validity(inner_new, (ocx_new, ocy_new), result.fun, oang_new):
                best_score = result.fun
                best_result = (inner_new, (ocx_new, ocy_new), result.fun, oang_new)
    
    # Return best result
    if best_result is not None:
        return best_result
    else:
        # Fallback: return the first pattern with its enclosing hexagon
        inner = make_honeycomb_row()
        outer_center, outer_side, outer_angle = _min_enclosing_hexagon(
            inner, [p[2] for p in inner], 0.0)
        return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
