import math
import time
import numpy as np
from itertools import combinations

N = 11
DEADLINE = time.time() + 140.0  # Safety margin from 150s limit

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, angle."""
    angles = [math.radians(angle_deg + 60 * i) for i in range(6)]
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def point_in_hexagon(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with given side and angle."""
    # Transform to hexagon's local frame
    dx = px - hx
    dy = py - hy
    theta = -math.radians(angle_deg)
    lx = dx * math.cos(theta) - dy * math.sin(theta)
    ly = dx * math.sin(theta) + dy * math.cos(theta)
    # For a regular hexagon with flat top (angle 0), the hexagon is symmetric
    # The hexagon contains points where the distance to each of 6 sides is <= side/sqrt(3)
    # Actually, for a regular hexagon with side length s, the distance from center to each side is s*sqrt(3)/2
    # Let me use a different approach: check if point is inside the hexagon defined by 6 lines
    # A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (flat-top) 
    # has equations: |x| <= s, |x/2 + sqrt(3)*y/2| <= s, |-x/2 + sqrt(3)*y/2| <= s
    # Actually let me just use the vertex winding or shapely-like check via half-plane test
    
    # For a regular hexagon with side s and rotation angle, the 6 sides are at distances s*sqrt(3)/2 from center
    # The normal directions are at angles angle_deg, angle_deg+60, ..., angle_deg+300
    # Point is inside if for each normal direction n_i, dot(point - center, n_i) <= s*sqrt(3)/2
    dist = side * math.sqrt(3) / 2
    for i in range(6):
        na = math.radians(angle_deg + 60 * i)
        nx = math.cos(na)
        ny = math.sin(na)
        if dx * nx + dy * ny > dist + 1e-9:
            return False
    return True

def hexagons_disjoint(h1x, h1y, h1a, h2x, h2y, h2a, side=1.0, tol=1e-9):
    """Check if two unit hexagons are disjoint (no interior overlap)."""
    # Get vertices of both hexagons
    v1 = hex_vertices(h1x, h1y, side, h1a)
    v2 = hex_vertices(h2x, h2y, side, h2a)
    # Check if any vertex of one is inside the other
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, h2x, h2y, side, h2a):
            return False
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, h1x, h1y, side, h1a):
            return False
    # Check for edge intersection - simplified: if centers are close, do a more careful check
    # For now, vertex-in-hexagon check is sufficient for most cases with regular hexagons
    # But to be safe, let's also check if edges intersect
    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def min_enclosing_hexagon_size(positions, angles):
    """Find minimal side length S such that all unit hexagons fit inside a regular hexagon of side S.
    Returns (S, outer_center, outer_angle)."""
    # For each candidate outer hexagon orientation, find the minimal S
    # The outer hexagon must contain all inner hexagons
    # A unit hexagon with center c and angle a is contained in outer hexagon (center O, side S, angle A)
    # iff all 6 vertices of the inner hexagon are inside the outer hexagon
    
    best_S = float('inf')
    best_O = (0.0, 0.0)
    best_A = 0.0
    
    # Try several outer hexagon angles
    for outer_angle in range(0, 30, 5):
        # For a given outer angle, we need to find the minimal S and center
        # The constraint is that each inner hexagon's vertices must be inside the outer hexagon
        # This is a complex optimization. Let's use a simpler approach:
        # Estimate S from the spread of inner hexagon centers plus some margin
        
        # Get all vertices of all inner hexagons
        all_points = []
        for i in range(len(positions)):
            x, y = positions[i]
            a = angles[i]
            vs = hex_vertices(x, y, 1.0, a)
            all_points.extend(vs)
        
        if not all_points:
            continue
        
        all_points = np.array(all_points)
        
        # The outer hexagon with center O, side S, angle A contains a point P iff
        # for all 6 normal directions n_k of the outer hexagon: (P - O) . n_k <= S * sqrt(3)/2
        # So S >= max over all points P and all normals n_k of (P - O) . n_k / (sqrt(3)/2)
        
        # For a fixed outer_angle, the normals are fixed. We need to choose O to minimize
        # max_P,k (P - O).n_k
        # This is equivalent to: S*sqrt(3)/2 >= max_k max_P (P.n_k - O.n_k)
        # And also S*sqrt(3)/2 >= max_k max_P (-P.n_k + O.n_k) [since we need both sides]
        # Actually the hexagon constraint is: |(P-O).n_k| <= S*sqrt(3)/2 for k=0..5
        # Wait no, it's (P-O).n_k <= S*sqrt(3)/2 for all k, where n_k are outward normals
        
        normals = []
        for k in range(6):
            na = math.radians(outer_angle + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        normals = np.array(normals)
        
        # For each normal n_k, we need: max_P (P.n_k - O.n_k) <= S*sqrt(3)/2
        # => O.n_k >= max_P P.n_k - S*sqrt(3)/2
        # And: max_P (-P.n_k + O.n_k) <= S*sqrt(3)/2 (for the opposite side, normal is -n_k)
        # But -n_k is already one of the normals (opposite direction)
        
        # So the constraint is: for each normal n_k, (P-O).n_k <= S*sqrt(3)/2 for all P
        # => O.n_k >= max_P(P.n_k) - S*sqrt(3)/2
        # And for the opposite normal -n_k: (-n_k).P - (-n_k).O <= S*sqrt(3)/2
        # => -P.n_k + O.n_k <= S*sqrt(3)/2
        # => O.n_k <= S*sqrt(3)/2 + min_P(P.n_k)
        
        # So for each k: max_P(P.n_k) - S*sqrt(3)/2 <= O.n_k <= S*sqrt(3)/2 + min_P(P.n_k)
        # => S*sqrt(3)/2 >= (max_P(P.n_k) - min_P(P.n_k)) / 2
        # => S >= (max_P(P.n_k) - min_P(P.n_k)) / sqrt(3)
        
        # And O.n_k can be anything in [max_P(P.n_k) - S*sqrt(3)/2, S*sqrt(3)/2 + min_P(P.n_k)]
        # The tightest S for this angle is: S = max_k (max_P(P.n_k) - min_P(P.n_k)) / sqrt(3)
        
        projected = all_points @ normals.T  # shape (num_points, 6)
        ranges = projected.max(axis=0) - projected.min(axis=0)  # shape (6,)
        S_candidate = ranges.max() / math.sqrt(3)
        
        if S_candidate < best_S:
            # Find a valid O
            # For each k, O.n_k must be in [proj_max_k - S*sqrt(3)/2, S*sqrt(3)/2 + proj_min_k]
            # The midpoint of these intervals gives a valid O if the intervals are non-empty
            # Let's just use the centroid of all points as O for simplicity
            O = all_points.mean(axis=0)
            
            # Verify
            valid = True
            for k in range(6):
                nx, ny = normals[k]
                dist = S_candidate * math.sqrt(3) / 2
                for P in all_points:
                    if (P[0] - O[0]) * nx + (P[1] - O[1]) * ny > dist + 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_S = S_candidate
                best_O = (O[0], O[1])
                best_A = float(outer_angle)
    
    return best_S, best_O, best_A

def generate_hex_lattice(n):
    """Generate n hexagons on a triangular/hex lattice."""
    positions = []
    angles = []
    
    # Generate hexagonal lattice points
    # Use axial coordinates
    lattice_points = []
    radius = 3  # search radius in axial coords
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            # Convert axial to cartesian (pointy-top hex lattice, side 1)
            # For unit hexagons that tile, the center spacing is sqrt(3) in the appropriate directions
            x = 1.5 * q
            y = math.sqrt(3) / 2 * q + math.sqrt(3) * r
            lattice_points.append((x, y, q, r))
    
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Select n closest points, but ensure they form a connected packing
    selected = []
    used = set()
    for p in lattice_points:
        if len(selected) >= n:
            break
        # Check if this point is adjacent to an already selected point (or is the first)
        if len(selected) == 0:
            selected.append(p)
            used.add((p[2], p[3]))
        else:
            # Check adjacency: in hex lattice, neighbors are at axial distance 1
            is_adjacent = False
            for s in selected:
                dq = p[2] - s[2]
                dr = p[3] - s[3]
                # Hex neighbors in axial coords
                if (abs(dq) <= 1 and abs(dr) <= 1 and abs(dq - dr) <= 1):
                    is_adjacent = True
                    break
            if is_adjacent:
                selected.append(p)
                used.add((p[2], p[3]))
    
    # If we didn't get n, just take the n closest
    if len(selected) < n:
        selected = lattice_points[:n]
    
    for x, y, q, r in selected:
        positions.append((x, y))
        angles.append(0.0)
    
    return positions, angles

def generate_row_packing(n):
    """Generate a row-based packing."""
    # 4-3-4 pattern for n=11
    rows = [4, 3, 4]
    positions = []
    angles = []
    dy = math.sqrt(3)
    for ri, count in enumerate(rows):
        y = (ri - 1) * dy
        for i in range(count):
            x = (i - (count - 1) / 2) * 2.0  # spacing of 2 for flat-top hexagons? 
            # Actually for pointy-top hexagons with side 1, horizontal spacing is sqrt(3), vertical is 1.5
            # Let me use pointy-top: x spacing = sqrt(3), y spacing = 1.5, offset rows
            x = (i - (count - 1) / 2) * math.sqrt(3)
            positions.append((x, y))
            angles.append(0.0)
    
    return positions, angles

def check_packing_valid(positions, angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    # Check disjointness
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not hexagons_disjoint(positions[i][0], positions[i][1], angles[i],
                                     positions[j][0], positions[j][1], angles[j]):
                return False
    
    # Check containment
    for i in range(len(positions)):
        x, y = positions[i]
        a = angles[i]
        vs = hex_vertices(x, y, 1.0, a)
        for vx, vy in vs:
            if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    return True

def local_optimize(positions, angles, outer_center, outer_side, outer_angle, max_iters=100):
    """Try to reduce outer_side by adjusting positions slightly."""
    pos = [list(p) for p in positions]
    ang = list(angles)
    ox, oy = outer_center
    S = outer_side
    A = outer_angle
    
    step = 0.01
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(len(pos)):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    new_x = pos[i][0] + di * step
                    new_y = pos[i][1] + dj * step
                    old_x, old_y = pos[i][0], pos[i][1]
                    pos[i] = [new_x, new_y]
                    
                    # Check disjointness
                    valid = True
                    for j in range(len(pos)):
                        if i == j:
                            continue
                        if not hexagons_disjoint(pos[i][0], pos[i][1], ang[i],
                                                 pos[j][0], pos[j][1], ang[j]):
                            valid = False
                            break
                    if valid:
                        # Check containment and get new S
                        new_S, new_O, new_A = min_enclosing_hexagon_size(pos, ang)
                        if new_S < S - 1e-8 and new_S < float('inf'):
                            S = new_S
                            ox, oy = new_O
                            A = new_A
                            improved = True
                        else:
                            pos[i] = [old_x, old_y]
                    else:
                        pos[i] = [old_x, old_y]
            if time.time() > DEADLINE:
                break
        if not improved and it > 20:
            step *= 0.5
            if step < 0.001:
                break
    
    return [tuple(p) for p in pos], ang, (ox, oy), S, A

def construct():
    start_time = time.time()
    
    # Generate several candidate packings
    candidates = []
    
    # 1. Hex lattice
    pos1, ang1 = generate_hex_lattice(N)
    candidates.append((pos1, ang1))
    
    # 2. Row packing 4-3-4
    pos2, ang2 = generate_row_packing(N)
    candidates.append((pos2, ang2))
    
    # 3. 3-4-3 row packing
    rows3 = [3, 4, 3]
    pos3 = []
    ang3 = []
    dy = math.sqrt(3)
    for ri, count in enumerate(rows3):
        y = (ri - 1) * dy
        for i in range(count):
            x = (i - (count - 1) / 2) * math.sqrt(3)
            pos3.append((x, y))
            ang3.append(0.0)
    candidates.append((pos3, ang3))
    
    # 4. 4-4-3 row packing
    rows4 = [4, 4, 3]
    pos4 = []
    ang4 = []
    for ri, count in enumerate(rows4):
        y = (ri - 1) * dy
        for i in range(count):
            x = (i - (count - 1) / 2) * math.sqrt(3)
            pos4.append((x, y))
            ang4.append(0.0)
    candidates.append((pos4, ang4))
    
    # Evaluate each candidate
    best_S = float('inf')
    best_result = None
    
    for pos, ang in candidates:
        if time.time() > DEADLINE:
            break
        
        S, O, A = min_enclosing_hexagon_size(pos, ang)
        
        if S < best_S and S < float('inf'):
            # Verify the packing is valid
            if check_packing_valid(pos, ang, O, S, A):
                best_S = S
                best_result = (pos, ang, O, S, A)
    
    # If no valid packing found, use a large fallback
    if best_result is None:
        pos, ang = generate_hex_lattice(N)
        S, O, A = min_enclosing_hexagon_size(pos, ang)
        if S == float('inf'):
            S = 50.0
            O = (0.0, 0.0)
            A = 0.0
        best_result = (pos, ang, O, S, A)
    
    # Local optimization on the best result
    pos, ang, O, S, A = best_result
    if time.time() < DEADLINE - 10:
        pos, ang, O, S, A = local_optimize(pos, ang, O, S, A, max_iters=50)
    
    # Final verification
    if not check_packing_valid(pos, ang, O, S, A):
        # Fallback: use a large hexagon
        S = 50.0
        O = (0.0, 0.0)
        A = 0.0
    
    # Format output
    inner = [[x, y, a] for x, y, ang in zip(pos, ang)]
    return inner, (O[0], O[1]), S, A

# EVOLVE-BLOCK-START
# (The construct function is defined above; this marker is for the evolution system)
# EVOLVE-BLOCK-END
