import math
import random
import time

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon with side s, center (cx, cy), 
    and rotation angle_deg.
    Vertices are in counter-clockwise order starting from angle_deg.
    """
    verts = []
    angle_rad = math.radians(angle_deg)
    for k in range(6):
        theta = angle_rad + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hex(px, py, hex_verts):
    """
    Check if point (px, py) is inside a convex polygon defined by hex_verts.
    Uses cross product to check if point is on the same side of all edges.
    """
    # For a convex polygon, if the point is inside, the cross product 
    # of the edge vector and the vector from vertex to point should have the same sign.
    sign = 0
    n = len(hex_verts)
    for i in range(n):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % n]
        # Vector from (x1,y1) to (x2,y2)
        ex = x2 - x1
        ey = y2 - y1
        # Vector from (x1,y1) to (px,py)
        vx = px - x1
        vy = py - y1
        cross = ex * vy - ey * vx
        if cross > 1e-9:
            if sign == -1:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign == 1:
                return False
            sign = -1
    return True

def hex_overlap(h1_verts, h2_verts):
    """
    Check if two convex hexagons overlap.
    They overlap if any vertex of one is inside the other.
    """
    for v in h1_verts:
        if point_in_hex(v[0], v[1], h2_verts):
            return True
    for v in h2_verts:
        if point_in_hex(v[0], v[1], h1_verts):
            return True
    return False

def compute_span_and_overlap(positions, rotations, outer_angle_deg):
    """
    positions: list of (x, y)
    rotations: list of angle_deg
    outer_angle_deg: rotation of the outer hexagon
    Returns: (min_side_length, overlap_count)
    """
    # Precompute vertices for all inner hexagons
    all_verts = []
    for i in range(N):
        x, y = positions[i]
        ang = rotations[i]
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    
    # Compute overlap
    overlap_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if hex_overlap(all_verts[i], all_verts[j]):
                overlap_count += 1

    # Compute spans in 6 directions of the outer hexagon
    # The normals to the sides of the outer hexagon are at angles:
    # outer_angle_deg + 90, +150, +210, +270, +330, +30
    # Or simply, the directions from center to vertices are outer_angle + k*60.
    # The normals are perpendicular to the sides.
    # Side k is between vertex k and k+1.
    # Vertex angle: outer_angle + k * 60
    # Side midpoint angle: outer_angle + k * 60 + 30
    # Normal angle: outer_angle + k * 60 + 30
    # Let's just use the 6 directions of the axes of symmetry.
    # Directions: outer_angle + 0, 60, 120, 180, 240, 300
    # These are directions to vertices. The width in these directions is not the side-to-side distance.
    # The side-to-side distance is along the normals.
    # Normals are at outer_angle + 30 + k*60.
    
    max_span = 0.0
    for k in range(6):
        angle_rad = math.radians(outer_angle_deg + 30 + k * 60)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        min_proj = float('inf')
        max_proj = float('-inf')
        
        for verts in all_verts:
            for vx, vy in verts:
                proj = vx * nx + vy * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
        
        span = max_proj - min_proj
        if span > max_span:
            max_span = span
            
    # Side length S such that 2 * sqrt(3) * S = max_span
    min_side = max_span / (2 * math.sqrt(3))
    
    return min_side, overlap_count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial state: Hexagonal lattice
    # A cluster of 11 hexagons.
    # Center (0,0), 6 around it, 4 more?
    # Let's use a standard hex grid.
    # (0,0)
    # (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1) -> 7 hexes
    # Add 4 more: (2,0), (2,1), (0,2), (-1,2)?
    # Let's just initialize randomly near origin or use a known packing.
    # For N=11, a compact shape is good.
    
    def init_state():
        # Try to place them in a hexagonal pattern
        # dx = 1.5, dy = sqrt(3)/2
        dx = 1.5
        dy = math.sqrt(3) / 2
        positions = []
        rotations = [0.0] * N
        
        # Manually place 11 hexes in a cluster
        # Row 0: 3 hexes
        # Row 1: 5 hexes
        # Row 2: 3 hexes
        # Total 11.
        
        # Center at 0,0
        # Row 1 (y=0): x = -1.5, 0, 1.5, 3.0, 4.5? No, let's center it.
        # Let's use axial coordinates (q, r)
        # (0,0)
        # (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1) -> 7
        # Add: (2,0), (2,-1), (1,1), (-1,1) -> 11
        
        coords = [
            (0, 0),
            (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
            (2, 0), (2, -1), (1, 1), (-1, 1)
        ]
        
        for q, r in coords:
            x = dx * (q + r/2.0)
            y = dy * r
            positions.append((x, y))
            
        return positions, rotations

    positions, rotations = init_state()
    outer_angle = 0.0
    
    # Evaluate initial
    cost, overlaps = compute_span_and_overlap(positions, rotations, outer_angle)
    best_cost = cost
    best_positions = positions[:]
    best_rotations = rotations[:]
    best_outer_angle = outer_angle
    best_valid = (overlaps == 0)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    penalty_factor = 100.0
    
    def get_energy(pos, rot, ang):
        s, ov = compute_span_and_overlap(pos, rot, ang)
        return s + penalty_factor * ov
    
    current_energy = get_energy(positions, rotations, outer_angle)
    
    while time.time() < deadline:
        # Pick a random hexagon
        i = random.randint(0, N - 1)
        
        # Move type: 0=translate, 1=rotate, 2=outer rotate
        move_type = random.randint(0, 2)
        
        old_pos = positions[i]
        old_rot = rotations[i]
        old_ang = outer_angle
        
        if move_type == 0:
            # Translate
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            positions[i] = (old_pos[0] + dx, old_pos[1] + dy)
        elif move_type == 1:
            # Rotate
            drot = random.gauss(0, 10)
            rotations[i] = (old_rot + drot) % 360
        else:
            # Outer rotate
            drot = random.gauss(0, 5)
            outer_angle = (old_ang + drot) % 360
            
        new_energy = get_energy(positions, rotations, outer_angle)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_energy = new_energy
            # Update best
            if current_energy < best_cost:
                best_cost = current_energy
                best_positions = positions[:]
                best_rotations = rotations[:]
                best_outer_angle = outer_angle
                # Check validity
                s, ov = compute_span_and_overlap(best_positions, best_rotations, best_outer_angle)
                best_valid = (ov == 0)
        else:
            # Revert
            positions[i] = old_pos
            rotations[i] = old_rot
            outer_angle = old_ang
            
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher T if stuck?
            # For now, just keep going.

    # Final polish: try to remove any remaining overlaps if best_valid is False
    # If best_valid is False, we might need to do a local search to fix overlaps.
    # Given the penalty, it's likely valid.
    
    # Return in required format
    # inner rows: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        ang = best_rotations[i]
        inner.append([x, y, ang])
        
    # Outer hexagon center (0,0) is optimal for the span calculation?
    # The span calculation is translation invariant.
    # The outer hexagon should be centered such that it contains the points.
    # The "center" returned is just a reference. (0,0) is fine if the points are centered.
    # Our initialization is centered around 0,0.
    
    return inner, (0.0, 0.0), best_cost, best_outer_angle

# EVOLVE-BLOCK-START
def construct():
    # Wrapper to handle the evolve block requirement
    return _construct_impl()

def _construct_impl():
    # This is the actual implementation
    return construct_inner()

def construct_inner():
    # Move the logic here to keep construct() clean?
    # No, just put the logic in construct().
    pass
# EVOLVE-BLOCK-END

# Refactor: Put the whole logic inside the evolve block as per instructions?
# "Keep the function names and signatures of the seed program."
# Seed has `construct()`.
# I will define helper functions outside or inside?
# Inside is safer to avoid namespace pollution, but outside is faster?
# Let's put helpers outside.

# REWRITING THE WHOLE FILE BELOW

import math
import random
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    verts = []
    angle_rad = math.radians(angle_deg)
    for k in range(6):
        theta = angle_rad + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, hex_verts):
    sign = 0
    n = len(hex_verts)
    for i in range(n):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % n]
        ex = x2 - x1
        ey = y2 - y1
        vx = px - x1
        vy = py - y1
        cross = ex * vy - ey * vx
        if cross > 1e-9:
            if sign == -1:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign == 1:
                return False
            sign = -1
    return True

def _hex_overlap(h1_verts, h2_verts):
    for v in h1_verts:
        if _point_in_hex(v[0], v[1], h2_verts):
            return True
    for v in h2_verts:
        if _point_in_hex(v[0], v[1], h1_verts):
            return True
    return False

def _compute_span_and_overlap(positions, rotations, outer_angle_deg):
    all_verts = []
    for i in range(N):
        x, y = positions[i]
        ang = rotations[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    
    overlap_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(all_verts[i], all_verts[j]):
                overlap_count += 1

    max_span = 0.0
    for k in range(6):
        angle_rad = math.radians(outer_angle_deg + 30 + k * 60)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        min_proj = float('inf')
        max_proj = float('-inf')
        
        for verts in all_verts:
            for vx, vy in verts:
                proj = vx * nx + vy * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
        
        span = max_proj - min_proj
        if span > max_span:
            max_span = span
            
    min_side = max_span / (2 * math.sqrt(3))
    
    return min_side, overlap_count

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def init_state():
        dx = 1.5
        dy = math.sqrt(3) / 2
        positions = []
        rotations = [0.0] * N
        coords = [
            (0, 0),
            (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
            (2, 0), (2, -1), (1, 1), (-1, 1)
        ]
        for q, r in coords:
            x = dx * (q + r/2.0)
            y = dy * r
            positions.append((x, y))
        return positions, rotations

    positions, rotations = init_state()
    outer_angle = 0.0
    
    def get_energy(pos, rot, ang):
        s, ov = _compute_span_and_overlap(pos, rot, ang)
        return s + 100.0 * ov

    current_energy = get_energy(positions, rotations, outer_angle)
    best_cost = current_energy
    best_positions = positions[:]
    best_rotations = rotations[:]
    best_outer_angle = outer_angle
    best_valid = (best_cost < 10.0) # Heuristic check
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        move_type = random.randint(0, 2)
        
        old_pos = positions[i]
        old_rot = rotations[i]
        old_ang = outer_angle
        
        if move_type == 0:
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            positions[i] = (old_pos[0] + dx, old_pos[1] + dy)
        elif move_type == 1:
            drot = random.gauss(0, 10)
            rotations[i] = (old_rot + drot) % 360
        else:
            drot = random.gauss(0, 5)
            outer_angle = (old_ang + drot) % 360
            
        new_energy = get_energy(positions, rotations, outer_angle)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_cost:
                best_cost = current_energy
                best_positions = positions[:]
                best_rotations = rotations[:]
                best_outer_angle = outer_angle
                # Re-evaluate validity for best
                s, ov = _compute_span_and_overlap(best_positions, best_rotations, best_outer_angle)
                best_valid = (ov == 0)
        else:
            positions[i] = old_pos
            rotations[i] = old_rot
            outer_angle = old_ang
            
        T *= alpha
        if T < T_min:
            T = T_min

    inner = []
    for i in range(N):
        x, y = best_positions[i]
        ang = best_rotations[i]
        inner.append([x, y, ang])
        
    return inner, (0.0, 0.0), best_cost, best_outer_angle
# EVOLVE-BLOCK-END
