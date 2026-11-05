"""Unit regular hexagons inside a regular hexagon - optimized packing."""
import math
import time
import numpy as np

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Return the 6 vertices of a regular hexagon (side s, center, rotation)."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def _point_in_hexagon(px, py, hx, hy, s, angle_deg):
    """Check if point (px,py) is inside hexagon with center (hx,hy), side s, rotation."""
    # Transform to hexagon's local frame
    a = math.radians(angle_deg)
    dx, dy = px - hx, py - hy
    # Rotate point by -a
    cx = dx * math.cos(-a) - dy * math.sin(-a)
    cy = dx * math.sin(-a) + dy * math.cos(-a)
    # Regular hexagon with "pointy top" has vertices at angles 0,60,120,180,240,300
    # The hexagon is the intersection of 6 half-planes: |dot(p, n_k)| <= s * sqrt(3)/2 * ... 
    # Actually for a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The 6 normals point at angles 30, 90, 150, 210, 270, 330 (perpendicular to edges)
    # A point is inside if for all 6 directions, the projection is within the apothem
    apothem = s * math.sqrt(3) / 2.0
    for k in range(6):
        theta = math.pi / 6 + k * math.pi / 3.0
        proj = cx * math.cos(theta) + cy * math.sin(theta)
        if abs(proj) > apothem:
            return False
    return True

def _hexagon_overlap(v1, v2, v3, v4):
    """Check if two hexagons (each given as 6 vertices) overlap using SAT.
    Returns True if they overlap (interiors intersect)."""
    hex1 = [(v1[i][0], v1[i][1]) for i in range(6)]
    hex2 = [(v2[i][0], v2[i][1]) for i in range(6)]
    # Combined edge normals
    polys = [hex1, hex2]
    for poly in polys:
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 6]
            # Edge vector
            ex, ey = x2 - x1, y2 - y1
            # Normal (perpendicular)
            nx, ny = -ey, ex
            nlen = math.sqrt(nx * nx + ny * ny)
            if nlen < 1e-12:
                continue
            nx /= nlen
            ny /= nlen
            # Project both polygons
            min1 = min(dot(p, nx, ny) for p in hex1)
            max1 = max(dot(p, nx, ny) for p in hex1)
            min2 = min(dot(p, nx, ny) for p in hex2)
            max2 = max(dot(p, nx, ny) for p in hex2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def dot(p, nx, ny):
    return p[0] * nx + p[1] * ny

def _min_enclosing_hexagon_side(centers, n_inner, inner_side=1.0):
    """Given centers and angles of inner hexagons, find minimum enclosing regular hexagon.
    We try rotations of the outer hexagon and for each, compute the required side length.
    The outer hexagon must contain all inner hexagons."""
    # For a given outer hexagon orientation (angle theta) and center (cx, cy),
    # the constraint is: for every point in every inner hexagon, it must be inside.
    # Equivalent: for each of the 6 face normals of the outer hexagon,
    # the max projection of all inner hexagon vertices onto that normal 
    # minus min projection must be <= side * sqrt(3).
    # Actually, the width of a set of points in direction n is max(p.n) - min(p.n).
    # For a regular hexagon with side L, the width in direction perpendicular to each 
    # face is L * sqrt(3). The width in the direction of a vertex is 2*L.
    
    # Better approach: the minimum enclosing regular hexagon with a given orientation
    # is determined by the support function. For orientation theta, the hexagon has
    # 6 face normals at theta + k*pi/3, k=0..5, and 6 vertex directions at theta + pi/6 + k*pi/3.
    # The side length L must satisfy: for each face normal direction, the width of the
    # set in that direction <= L * sqrt(3).
    # And for each vertex direction, the width <= 2*L.
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(n_inner):
        cx, cy, ang = centers[i]
        verts = _hex_vertices(cx, cy, inner_side, ang)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    best_side = float('inf')
    best_theta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Try a range of outer hexagon rotations
    for theta_deg in range(0, 180, 1):
        theta = math.radians(theta_deg)
        # Face normals at theta + k*pi/3
        # The width in direction n is max(all_verts dot n) - min(all_verts dot n)
        required = 0.0
        for k in range(6):
            angle = theta + k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            width = projs.max() - projs.min()
            # Width in face-normal direction must be <= L * sqrt(3)
            req = width / (math.sqrt(3))
            if req > required:
                required = req
        # Also check vertex directions
        for k in range(6):
            angle = theta + math.pi / 6 + k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            width = projs.max() - projs.min()
            # Width in vertex direction must be <= 2*L
            req = width / 2.0
            if req > required:
                required = req
        if required < best_side:
            best_side = required
            best_theta = theta
            # Center should be at the midpoint of the bounding box in the hexagon frame
            cx = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2.0
            cy = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2.0
            best_cx = cx
            best_cy = cy
    
    # Refine with finer search around best_theta
    if best_side < float('inf'):
        for theta_deg in np.linspace(best_theta - 2, best_theta + 2, 41):
            theta = math.radians(theta_deg)
            required = 0.0
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                width = projs.max() - projs.min()
                req = width / (math.sqrt(3))
                if req > required:
                    required = req
            for k in range(6):
                angle = theta + math.pi / 6 + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                width = projs.max() - projs.min()
                req = width / 2.0
                if req > required:
                    required = req
            if required < best_side:
                best_side = required
                best_theta = theta
                best_cx = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2.0
                best_cy = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2.0
    
    return best_cx, best_cy, best_side, math.degrees(best_theta)

def _check_disjoint(centers):
    """Check that all inner hexagons are pairwise disjoint (non-overlapping)."""
    verts_list = []
    for i in range(len(centers)):
        cx, cy, ang = centers[i]
        verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hexagon_overlap(verts_list[i], verts_list[j], None, None):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    safety_margin = 15.0
    
    # Start with a reasonable initial configuration
    # Place hexagons in a compact pattern
    # For N=11, try a central hexagon surrounded by 6, plus 4 more
    
    # Strategy: use a dense packing arrangement
    # Hexagonal close packing: rows offset by half
    
    def make_row_config():
        """Create a row-based configuration."""
        config = []
        # Center hexagon
        config.append((0.0, 0.0, 0.0))
        # 6 around it (hexagonal neighbors)
        d = 2.0  # distance between centers for non-overlapping unit hexagons
        for k in range(6):
            angle = k * math.pi / 3.0
            config.append((d * math.cos(angle), d * math.sin(angle), 0.0))
        # 4 more: extend outward
        # Place at the "corners" of the hexagonal arrangement
        d2 = 2.0 * math.sqrt(3)  # further out
        for k in range(6):
            angle = math.pi / 6 + k * math.pi / 3.0
            config.append((d2 * math.cos(angle), d2 * math.sin(angle), 0.0))
        return config[:N]
    
    initial_config = make_row_config()
    
    # Compute initial score
    init_cx, init_cy, init_side, init_angle = _min_enclosing_hexagon_side(initial_config, N)
    best_config = list(initial_config)
    best_score = init_side
    
    # Local search: perturb positions and angles
    import random
    random.seed(42)
    
    # Define candidate moves
    n_inner = N
    positions = [list(c) for c in best_config]  # [x, y, angle]
    
    def evaluate(config):
        """Evaluate a configuration, return (side_length, valid)."""
        if not _check_disjoint(config):
            return float('inf'), False
        cx, cy, side, angle = _min_enclosing_hexagon_side(config, n_inner)
        return side, True
    
    # Simulated annealing style local search
    current_config = [list(c) for c in best_config]
    current_score = best_score
    T = 5.0  # temperature
    cooling = 0.995
    
    while time.time() < deadline - safety_margin:
        # Pick a random inner hexagon to move
        idx = random.randint(0, n_inner - 1)
        
        # Random perturbation
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        da = random.gauss(0, 15.0)
        
        new_config = [list(c) for c in current_config]
        new_config[idx] = [new_config[idx][0] + dx, new_config[idx][1] + dy, new_config[idx][2] + da]
        
        new_score, valid = evaluate(new_config)
        
        if not valid:
            # Reject invalid configurations
            # But sometimes accept to escape local minima
            if random.random() < 0.1:
                current_config = new_config
                current_score = float('inf')
            continue
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_config = new_config
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_config = [list(c) for c in current_config]
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final evaluation of best config
    final_cx, final_cy, final_side, final_angle = _min_enclosing_hexagon_side(best_config, N)
    
    # Build inner list: (x, y, angle_degrees)
    inner = []
    for (x, y, ang) in best_config:
        inner.append([x, y, ang])
    
    return inner, (final_cx, final_cy), final_side, final_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
