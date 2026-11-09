"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import random
import numpy as np

N = 11
DEADLINE = 145.0  # seconds, safety margin before 150s kill

# Precompute unit hexagon vertices (flat-top orientation, angle 0)
_UNIT_HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = side * _UNIT_HEX_VERTS @ R.T
    return verts + np.array([cx, cy])

def _hexagons_overlap(x1, y1, a1, x2, y2, a2, side1=1.0, side2=1.0, tol=1e-9):
    """
    Check if two regular hexagons overlap.
    Uses separating axis theorem with all 12 edge normals (6 from each hex).
    """
    verts1 = _hex_vertices(x1, y1, side1, a1)
    verts2 = _hex_vertices(x2, y2, side2, a2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
            normals.append(normal)
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
            return False
    return True

def _compute_outer_side(inner_positions):
    """
    Find the minimum outer regular hexagon side length and center that contains all inner hexagons.
    """
    all_verts = []
    for x, y, ang in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    best_side = 1e18
    best_center = (0, 0)
    best_outer_angle = 0
    
    # The outer hexagon can be rotated. We try a few angles.
    # Due to symmetry, we only need to check 0 to 60 degrees.
    for outer_angle in np.linspace(0, 60, 13):
        # Face normals for this outer angle
        normals = []
        for i in range(6):
            ang = math.radians(outer_angle + 30 + 60 * i)
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        # Estimate center as centroid
        centroid = np.mean(all_verts, axis=0)
        
        # Refine center slightly to minimize max projection
        # The side length s is determined by max_i (max_proj_i - n_i . C) / apothem
        # To minimize s, we want to minimize the max of these terms.
        # This is a convex optimization problem (minimize max of linear functions).
        # Simple gradient descent or just checking centroid + small offsets.
        
        best_local_side = 1e18
        best_local_center = centroid
        
        # Try centroid and a few offsets
        offsets = [(0,0), (0.1,0), (-0.1,0), (0,0.1), (0,-0.1), (0.1,0.1), (-0.1,-0.1), (0.1,-0.1), (-0.1,0.1)]
        for dx, dy in offsets:
            C = centroid + np.array([dx, dy])
            max_val = 0
            for n in normals:
                proj = all_verts @ n
                max_proj = np.max(proj)
                val = (max_proj - n @ C) / (math.sqrt(3) / 2)
                if val > max_val:
                    max_val = val
            
            if max_val < best_local_side:
                best_local_side = max_val
                best_local_center = C
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = (best_local_center[0], best_local_center[1])
            best_outer_angle = outer_angle
            
    return best_center, best_side, best_outer_angle

def _is_valid(inner_positions, outer_center, outer_side, outer_angle, tol=1e-8):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            x1, y1, a1 = inner_positions[i]
            x2, y2, a2 = inner_positions[j]
            if _hexagons_overlap(x1, y1, a1, x2, y2, a2):
                return False
    
    # Check containment
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    def point_in_convex_poly(px, py, poly):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
            if cross < -tol:
                return False
        return True

    for x, y, a in inner_positions:
        verts = _hex_vertices(x, y, 1.0, a)
        for v in verts:
            if not point_in_convex_poly(v[0], v[1], outer_verts):
                return False
    return True

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Correct initial configuration using hexagonal lattice
    # Hexagons of side 1, angle 0.
    # Neighbors of (0,0) are at:
    # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (0, -sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2)
    h = math.sqrt(3) / 2
    w = 1.5
    
    # 11 hexagons: 1 center, 6 neighbors, 4 second ring
    # Center: (0,0)
    # Ring 1:
    # (w, h), (w, -h), (0, 2*h), (0, -2*h), (-w, h), (-w, -h)
    # Ring 2 (select 4 to make 11 total? No, 1+6=7. Need 4 more.)
    # Let's pick 4 from the second ring to keep it compact.
    # Second ring positions (relative to center):
    # (3, 0), (3, 2h), (1.5, 3h), (-1.5, 3h), (-3, 2h), (-3, 0), (-3, -2h), (-1.5, -3h), (1.5, -3h)
    # Actually, let's just use a known compact packing for 11 hexagons if possible, 
    # or start with a valid cluster and optimize.
    
    positions = [
        [0, 0, 0],
        [w, h, 0],
        [w, -h, 0],
        [0, 2*h, 0],
        [0, -2*h, 0],
        [-w, h, 0],
        [-w, -h, 0],
        # 4 more:
        [3, 0, 0],
        [3, 2*h, 0],
        [-3, 0, 0],
        [-3, -2*h, 0],
    ]
    
    # Verify initial validity
    center, side, angle = _compute_outer_side(positions)
    if not _is_valid(positions, center, side, angle):
        # Fallback to a very sparse configuration if initial is invalid
        positions = [[i * 3.0, 0, 0] for i in range(11)]
        center, side, angle = _compute_outer_side(positions)
    
    best_inner = [row[:] for row in positions]
    best_center = center
    best_side = side
    best_angle = angle
    best_score = side
    
    # Local Search
    step_size = 0.5
    max_time = DEADLINE - 5
    
    while time.time() - start_time < max_time:
        # Pick a random hexagon
        idx = random.randint(0, N - 1)
        old = best_inner[idx][:]
        
        # Perturb
        new_pos = best_inner[idx][:]
        new_pos[0] += random.gauss(0, step_size)
        new_pos[1] += random.gauss(0, step_size)
        new_pos[2] += random.gauss(0, 15) # degrees
        
        # Temporarily set
        best_inner[idx] = new_pos
        
        center_new, side_new, angle_new = _compute_outer_side(best_inner)
        
        if _is_valid(best_inner, center_new, side_new, angle_new):
            if side_new < best_score:
                best_score = side_new
                best_center = center_new
                best_angle = angle_new
                # Keep change
                step_size = max(0.05, step_size * 0.98)
            else:
                # Revert
                best_inner[idx] = old
                step_size = max(0.05, step_size * 0.99)
        else:
            # Revert
            best_inner[idx] = old
            step_size = max(0.05, step_size * 0.99)
            
    return best_inner, best_center, best_side, best_angle
