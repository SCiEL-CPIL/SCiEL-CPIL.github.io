"""Packing unit hexagons in a regular hexagon via simulated annealing."""
import time
import math
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle):
    """Vertices of regular hexagon centered at (cx,cy), circumradius r, rotation angle (rad)."""
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return verts

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects p3-p4."""
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

def _polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap (using SAT or edge intersection)."""
    # Check if any edge of one intersects any edge of other
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        for j in range(len(v2)):
            p3 = v2[j]
            p4 = v2[(j + 1) % len(v2)]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    # Check if one is inside the other (point in polygon)
    if _point_in_polygon(v1[0], v2) or _point_in_polygon(v2[0], v1):
        return True
    return False

def _point_in_polygon(pt, poly):
    """Ray casting point in polygon."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _hex_in_hex(v_inner, v_outer):
    """Check if inner hexagon is fully inside outer hexagon."""
    for pt in v_inner:
        if not _point_in_polygon(pt, v_outer):
            return False
    return True

def _min_overlap_penalty(positions, angles, outer_center, outer_side, outer_angle):
    """Compute penalty for overlaps and containment violations."""
    R_outer = outer_side  # circumradius = side for regular hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], R_outer, outer_angle)
    penalty = 0.0
    
    # Check containment
    for i in range(len(positions)):
        x, y = positions[i]
        verts = _hex_vertices(x, y, 1.0, angles[i])
        if not _hex_in_hex(verts, outer_verts):
            # Compute max penetration depth (approximate)
            for pt in verts:
                if not _point_in_polygon(pt, outer_verts):
                    penalty += 10.0
    
    # Check pairwise overlaps
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if _polygons_overlap(v1, v2):
                penalty += 10.0
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 40.0
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Initial configuration: hexagonal lattice arrangement
    # For N hexagons, try a compact arrangement
    # Hexagonal number pattern: 1, 7, 19, 37, ...
    # N=11: not a perfect hex number, so use a rectangular-ish grid
    
    # Try different initial layouts
    def make_initial_layout(side_est):
        """Create initial positions in a hexagonal grid pattern."""
        positions = []
        angles = []
        # Hex grid with spacing 2.0 (touching)
        dx = 2.0
        dy = 2.0 * math.sqrt(3) / 2.0
        
        # Place in a compact pattern
        n_per_row = int(math.ceil(math.sqrt(N * 2 / math.sqrt(3))))
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            x = col * dx + (row % 2) * dx / 2.0
            y = row * dy
            # Center the layout
            positions.append((x, y))
            angles.append(0.0)
        
        # Center the layout
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(p[0] - cx, p[1] - cy) for p in positions]
        return positions, angles
    
    def compute_enclosing_side(positions, angles, center, outer_angle):
        """Compute minimum outer hexagon side to contain all inner hexagons."""
        # For a given outer hexagon orientation, find min side length
        # that contains all inner hexagons
        # The outer hexagon has 6 sides; each inner hexagon must be inside
        
        # Use the fact that for a regular hexagon with circumradius R,
        # a point (x,y) is inside iff max over 6 directions of projection <= R
        # Direction vectors for hexagon with angle theta:
        # The 6 face normals are at angles theta + k*pi/3
        
        min_side = 0.0
        for i in range(len(positions)):
            verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for pt in verts:
                # Relative to center
                rx = pt[0] - center[0]
                ry = pt[1] - center[1]
                # Project onto 6 face normal directions
                for k in range(6):
                    a = outer_angle + k * math.pi / 3.0
                    proj = rx * math.cos(a) + ry * math.sin(a)
                    # For a regular hexagon with circumradius R, the apothem is R*cos(pi/6)
                    # A point is inside if |proj| <= R * cos(pi/6) for all 6 directions
                    # Actually, the support function: max projection = R * cos(pi/6) for the apothem
                    # Wait: for a regular hexagon with side s, circumradius = s, apothem = s*sqrt(3)/2
                    # The condition for a point to be inside is that its projection onto each
                    # face normal direction is <= apothem
                    # apothem = R * sqrt(3)/2 where R is circumradius = side length
                    required_R = proj / (math.sqrt(3) / 2.0)
                    if required_R > min_side:
                        min_side = required_R
        
        return max(min_side, 1.0)
    
    def full_evaluation(positions, angles, center, outer_angle):
        """Full evaluation: compute enclosing side and check validity."""
        side = compute_enclosing_side(positions, angles, center, outer_angle)
        
        # Check for overlaps
        penalty = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                if _polygons_overlap(v1, v2):
                    penalty += 1.0
        
        if penalty > 0:
            return side + penalty * 5.0, False
        return side, True
    
    # Simulated annealing
    # State: positions (N*2), angles (N), center (2), outer_angle (1)
    # Total: 2N + N + 2 + 1 = 3N + 3 = 36 for N=11
    
    # Initialize
    init_side = 10.0
    positions, angles = make_initial_layout(init_side)
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial score
    init_score, init_valid = full_evaluation(positions, angles, center, outer_angle)
    current_score = init_score
    current_positions = [list(p) for p in positions]
    current_angles = list(angles)
    current_center = list(center)
    current_outer_angle = outer_angle
    
    if init_valid and init_score < best_score:
        best_score = init_score
        best_positions = [list(p) for p in current_positions]
        best_angles = list(current_angles)
        best_center = tuple(current_center)
        best_outer_angle = current_outer_angle
    
    # Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    max_iter = 50000
    
    iter_count = 0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        iter_count += 1
        
        # Generate a random perturbation
        # Choose what to perturb: position, angle, center, outer_angle
        r = random.random()
        step_scale = T * 0.5
        
        if r < 0.5:
            # Perturb a position
            i = random.randint(0, N - 1)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            current_positions[i][0] += dx
            current_positions[i][1] += dy
        elif r < 0.7:
            # Perturb an angle
            i = random.randint(0, N - 1)
            da = random.gauss(0, step_scale * 0.3)
            current_angles[i] += da
            # Keep angles in [0, pi)
            current_angles[i] %= math.pi
        elif r < 0.8:
            # Perturb center
            dx = random.gauss(0, step_scale * 0.3)
            dy = random.gauss(0, step_scale * 0.3)
            current_center[0] += dx
            current_center[1] += dy
        elif r < 0.9:
            # Perturb outer angle
            da = random.gauss(0, step_scale * 0.1)
            current_outer_angle += da
        else:
            # Perturb all positions slightly (rigid body)
            dx = random.gauss(0, step_scale * 0.1)
            dy = random.gauss(0, step_scale * 0.1)
            for i in range(N):
                current_positions[i][0] += dx
                current_positions[i][1] += dy
        
        # Evaluate new state
        new_score, new_valid = full_evaluation(current_positions, current_angles, current_center, current_outer_angle)
        
        # Metropolis acceptance
        if new_score < current_score:
            current_score = new_score
            if new_valid and new_score < best_score:
                best_score = new_score
                best_positions = [list(p) for p in current_positions]
                best_angles = list(current_angles)
                best_center = tuple(current_center)
                best_outer_angle = current_outer_angle
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0:
                current_score = new_score
            else:
                if random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Local refinement: try to minimize enclosing side with fixed topology
    # Use a simple gradient-free approach: try to shrink the configuration
    if best_positions is not None and time.time() < deadline:
        # Try to optimize by scaling down
        for _ in range(20):
            if time.time() > deadline:
                break
            # Try small random adjustments to reduce side
            improved = False
            for trial in range(100):
                if time.time() > deadline:
                    break
                trial_positions = [list(p) for p in best_positions]
                trial_angles = list(best_angles)
                trial_center = list(best_center)
                trial_outer_angle = best_outer_angle
                
                # Random small perturbation
                i = random.randint(0, N - 1)
                trial_positions[i][0] += random.gauss(0, 0.05)
                trial_positions[i][1] += random.gauss(0, 0.05)
                if random.random() < 0.3:
                    trial_angles[i] += random.gauss(0, 0.05)
                    trial_angles[i] %= math.pi
                
                trial_score, trial_valid = full_evaluation(
                    trial_positions, trial_angles, trial_center, trial_outer_angle)
                
                if trial_valid and trial_score < best_score:
                    best_score = trial_score
                    best_positions = trial_positions
                    best_angles = trial_angles
                    best_center = tuple(trial_center)
                    best_outer_angle = trial_outer_angle
                    improved = True
    
    # Final: compute the actual enclosing side for the best configuration
    if best_positions is not None:
        final_side = compute_enclosing_side(best_positions, best_angles, best_center, best_outer_angle)
        # Verify no overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                v2 = _hex_vertices(best_positions[j][0], best_positions[j][1], 1.0, best_angles[j])
                if _polygons_overlap(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            inner = [[best_positions[i][0], best_positions[i][1], math.degrees(best_angles[i])] for i in range(N)]
            return inner, best_center, final_side, math.degrees(best_outer_angle)
    
    # Fallback: return the initial configuration
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
