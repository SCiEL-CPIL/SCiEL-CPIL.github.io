"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 115.0  # safety margin before 150s kill
    
    # Geometry helpers for regular hexagons
    # A regular hexagon with side length s has:
    # - Circumradius (center to vertex) = s
    # - Inradius (center to edge midpoint) = s * sqrt(3)/2
    # - Adjacent hexagons (sharing edge) have centers separated by 2 * inradius = s * sqrt(3)
    
    SQRT3 = np.sqrt(3)
    
    def hex_vertices(cx, cy, s, angle_deg):
        """Get 6 vertices of a regular hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.radians(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def point_in_hex(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside regular hexagon."""
        a = np.radians(angle_deg)
        dx, dy = px - cx, py - cy
        # Rotate point into hexagon's local frame
        rx = dx * np.cos(a) + dy * np.sin(a)
        ry = -dx * np.sin(a) + dy * np.cos(a)
        # Apothem (distance from center to each edge)
        apothem = s * SQRT3 / 2
        # The 6 edge normals in local frame are at angles 30, 90, 150, 210, 270, 330
        for k in range(6):
            edge_angle = np.radians(30 + 60 * k)
            nx, ny = np.cos(edge_angle), np.sin(edge_angle)
            dist = rx * nx + ry * ny
            if dist > apothem + 1e-10:
                return False
        return True
    
    def hexagons_overlap(i1, i2):
        """Check if two unit hexagons overlap. 
        i1, i2 = (x, y, angle_deg)
        For simplicity, we check if center distance < 2*inradius when aligned.
        For rotated hexagons, this is more complex. We'll use vertex-edge check.
        """
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        
        # Quick reject: center distance
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx*dx + dy*dy)
        # Minimum distance between centers of non-overlapping unit hexagons is SQRT3
        # (when they share an edge). If dist < SQRT3 - epsilon, they definitely overlap.
        if dist < SQRT3 - 0.01:
            return True
        # If dist > 2, they definitely don't overlap (max distance between any two points
        # in a unit hexagon is 2, so if centers are > 2+SQRT3 apart... actually this isn't tight)
        # Let's do a more careful check: check if any vertex of one is inside the other
        verts1 = hex_vertices(x1, y1, 1.0, a1)
        verts2 = hex_vertices(x2, y2, 1.0, a2)
        
        for (vx, vy) in verts1:
            if point_in_hex(vx, vy, x2, y2, 1.0, a2):
                return True
        for (vx, vy) in verts2:
            if point_in_hex(vx, vy, x1, y1, 1.0, a1):
                return True
        return False
    
    def all_non_overlapping(inner_list):
        """Check no two inner hexagons overlap."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                if hexagons_overlap(inner_list[i], inner_list[j]):
                    return False
        return True
    
    def all_inside_outer(inner_list, cx, cy, s, angle_deg):
        """Check all inner hexagon vertices are inside outer hexagon."""
        for (ix, iy, iangle) in inner_list:
            verts = hex_vertices(ix, iy, 1.0, iangle)
            for (vx, vy) in verts:
                if not point_in_hex(vx, vy, cx, cy, s, angle_deg):
                    return False
        return True
    
    def find_min_outer_side(inner_list, max_iter=50):
        """Find minimum outer hexagon side length for given inner arrangement."""
        if not all_non_overlapping(inner_list):
            return float('inf'), None
        
        centers = np.array([(x, y) for (x, y, a) in inner_list])
        centroid = centers.mean(axis=0)
        
        best_side = float('inf')
        best_params = None
        
        # Try different outer angles and center offsets
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # Try center near centroid with small offsets
            for dx in np.linspace(-1.5, 1.5, 7):
                for dy in np.linspace(-1.5, 1.5, 7):
                    cx = centroid[0] + dx
                    cy = centroid[1] + dy
                    
                    # Binary search on side length
                    lo, hi = 1.0, 15.0
                    for _ in range(max_iter):
                        mid = (lo + hi) / 2
                        if all_inside_outer(inner_list, cx, cy, mid, outer_angle):
                            hi = mid
                        else:
                            lo = mid
                        if hi - lo < 1e-5:
                            break
                    
                    if hi < best_side:
                        best_side = hi
                        best_params = (cx, cy, outer_angle)
        
        return best_side, best_params
    
    # Generate candidate arrangements
    candidates = []
    
    # Arrangement A: Compact hexagonal cluster (1 center + 6 ring + 4 outer)
    d = SQRT3  # distance between adjacent hexagon centers
    inner_A = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        inner_A.append([d * np.cos(angle), d * np.sin(angle), 0.0])
    # 4 more at the "corner" positions
    for ang in [30, 90, 210, 270]:
        angle = np.radians(ang)
        inner_A.append([2 * d * np.cos(angle), 2 * d * np.sin(angle), 0.0])
    candidates.append(inner_A)
    
    # Arrangement B: 2 staggered rows (6 + 5)
    inner_B = []
    for k in range(6):
        inner_B.append([d * (k - 2.5), 0.0, 0.0])
    for k in range(5):
        inner_B.append([d * (k - 2.0), d * SQRT3 / 2, 0.0])
    candidates.append(inner_B)
    
    # Arrangement C: 3 rows (4 + 3 + 4) staggered
    inner_C = []
    for k in range(4):
        inner_C.append([d * (k - 1.5), 0.0, 0.0])
    for k in range(3):
        inner_C.append([d * (k - 1.0), d * SQRT3 / 2, 0.0])
    for k in range(4):
        inner_C.append([d * (k - 1.5), d * SQRT3, 0.0])
    candidates.append(inner_C)
    
    # Arrangement D: Linear
    inner_D = [[d * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    candidates.append(inner_D)
    
    # Evaluate all candidates
    best_score = float('inf')
    best_result = None
    
    for idx, inner in enumerate(candidates):
        if time.time() - t_start > deadline:
            break
        side, params = find_min_outer_side(inner)
        if side < best_score and params is not None:
            best_score = side
            best_result = (inner, params[0], params[1], side, params[2])
    
    if best_result is None:
        # Fallback
        return inner_D, (0.0, 0.0), 40.0, 0.0
    
    inner, cx, cy, side, angle = best_result
    return inner, (cx, cy), side, angle

# EVOLVE-BLOCK-END
