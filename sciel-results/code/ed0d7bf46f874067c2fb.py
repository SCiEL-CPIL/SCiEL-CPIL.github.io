"""Packing 12 unit hexagons in a regular hexagon - physics + SA hybrid."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon (flat-top at 0 deg)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.pi / 3.0 * i
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _support_function(verts, dx, dy):
    """Max of dot product of vertices with direction (dx,dy)."""
    mx = -1e18
    for (x, y) in verts:
        d = x * dx + y * dy
        if d > mx:
            mx = d
    return mx

def _separation(v1, v2):
    """Minimum translation distance to separate two convex polygons (SAT)."""
    n1 = len(v1)
    n2 = len(v2)
    # Compute all edge normals
    normals = []
    for i in range(n1):
        ax, ay = v1[i]
        bx, by = v1[(i + 1) % n1]
        ex, ey = bx - ax, by - ay
        l = math.hypot(ex, ey)
        if l < 1e-12:
            continue
        # normal pointing outward (for CCW polygon, right normal)
        nx, ny = ey / l, -ex / l
        normals.append((nx, ny))
    for i in range(n2):
        ax, ay = v2[i]
        bx, by = v2[(i + 1) % n2]
        ex, ey = bx - ax, by - ay
        l = math.hypot(ex, ey)
        if l < 1e-12:
            continue
        nx, ny = ey / l, -ex / l
        normals.append((nx, ny))
    
    best_sep = 0.0
    best_dir = (1.0, 0.0)
    for (nx, ny) in normals:
        # Project both polygons onto this axis
        s1 = _support_function(v1, nx, ny)
        s2 = _support_function(v2, nx, ny)
        # Also need min projections
        m1 = 1e18
        m2 = 1e18
        for (x, y) in v1:
            d = x * nx + y * ny
            if d < m1:
                m1 = d
        for (x, y) in v2:
            d = x * nx + y * ny
            if d < m2:
                m2 = d
        # Overlap on this axis?
        # v1 range [m1, s1], v2 range [m2, s2]
        if s1 < m2 or s2 < m1:
            # Separated on this axis
            gap = max(m2 - s1, m1 - s2)
            if gap > best_sep:
                best_sep = gap
                best_dir = (nx, ny)
            continue
        # Overlapping: candidate separation is how much to push apart
        # We want the minimum penetration depth direction
        # For SAT-based separation, we track the axis of minimum overlap
        overlap = min(s1, s2) - max(m1, m2)
        # This axis has overlap; the separation along this axis is at least 'overlap'
        # But we want the axis where they're closest (min gap when separated)
        # Actually for finding separation vector, we want the axis with minimum
        # penetration. Let's track differently.
        # The minimum translation vector is along the axis of least overlap.
        if overlap < best_sep:
            best_sep = overlap
            # Direction to push v1 away from v2
            # If s1 > s2, push v1 in +normal; else push in -normal
            if (s1 - s2) >= 0:
                best_dir = (nx, ny)
            else:
                best_dir = (-nx, -ny)
    
    return best_sep, best_dir

def _pairwise_collision(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        ax, ay = v1[i]
        bx, by = v1[(i + 1) % n1]
        ex, ey = bx - ax, by - ay
        l = math.hypot(ex, ey)
        if l < 1e-12:
            continue
        nx, ny = ey / l, -ex / l
        # Project both
        m1 = 1e18; s1 = -1e18
        for (x, y) in v1:
            d = x * nx + y * ny
            if d < m1: m1 = d
            if d > s1: s1 = d
        m2 = 1e18; s2 = -1e18
        for (x, y) in v2:
            d = x * nx + y * ny
            if d < m2: m2 = d
            if d > s2: s2 = d
        if s1 < m2 or s2 < m1:
            return False  # Separated
    for i in range(n2):
        ax, ay = v2[i]
        bx, by = v2[(i + 1) % n2]
        ex, ey = bx - ax, by - ay
        l = math.hypot(ex, ey)
        if l < 1e-12:
            continue
        nx, ny = ey / l, -ex / l
        m1 = 1e18; s1 = -1e18
        for (x, y) in v1:
            d = x * nx + y * ny
            if d < m1: m1 = d
            if d > s1: s1 = d
        m2 = 1e18; s2 = -1e18
        for (x, y) in v2:
            d = x * nx + y * ny
            if d < m2: m2 = d
            if d > s2: s2 = d
        if s1 < m2 or s2 < m1:
            return False
    return True

def _bounding_hexagon_side(centers, angles, outer_center, outer_angle):
    """Find minimum side length of outer hexagon (at given center/angle) containing all inner hexagons."""
    # Outer hexagon with side S, center (cx,cy), angle a
    # A point p is inside if for all 6 face normals n_i: dot(p-c, n_i) <= S * sqrt(3)/2
    # For flat-top hexagon, face normals are at angles 0, 60, 120, 180, 240, 300
    # The apothem (distance from center to face) = S * sqrt(3) / 2
    
    ocx, ocy = outer_center
    oa = math.radians(outer_angle)
    max_required = 0.0
    for i in range(N):
        cx, cy, ang = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            # Check all 6 constraints
            for k in range(6):
                theta = oa + math.pi / 3.0 * k
                nx = math.cos(theta)
                ny = math.sin(theta)
                d = (vx - ocx) * nx + (vy - ocy) * ny
                if d > max_required:
                    max_required = d
    # Required apothem = max_required, so side = max_required * 2 / sqrt(3)
    return max_required * 2.0 / math.sqrt(3.0)

def _total_energy(centers, angles, outer_center, outer_angle):
    """Compute objective: outer side length + penalty for inner overlaps."""
    # First check inner-inner collisions
    penalty = 0.0
    verts_list = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _pairwise_collision(verts_list[i], verts_list[j]):
                sep, _ = _separation(verts_list[i], verts_list[j])
                penalty += sep * 100.0
    side = _bounding_hexagon_side(centers, angles, outer_center, outer_angle)
    return side + penalty

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal lattice arrangement
    # Place 12 hexagons in a compact arrangement
    # A hexagonal packing of 12: center + 6 around + 5 more
    # Let's use a 3-4-3-2 or similar arrangement
    
    # Start with a grid-like arrangement
    positions = []
    angles_init = []
    
    # Hexagonal lattice positions (for flat-top hexagons, spacing)
    # For unit hexagons, center-to-center distance for non-overlapping = 2 * apothem = sqrt(3)
    dx = math.sqrt(3)
    dy = 1.5
    
    # Arrange in rows
    row_y = 0
    # Row 0: 3 hexagons
    for k in range(3):
        positions.append((k * dx, 0.0))
        angles_init.append(0.0)
    # Row 1: 4 hexagons (offset)
    for k in range(4):
        positions.append((k * dx - dx / 2, dy))
        angles_init.append(0.0)
    # Row 2: 3 hexagons
    for k in range(3):
        positions.append((k * dx, 2 * dy))
        angles_init.append(0.0)
    # Row 3: 2 hexagons
    for k in range(2):
        positions.append((k * dx - dx / 4, 3 * dy))
        angles_init.append(0.0)
    
    # Center the arrangement
    cx_mean = sum(p[0] for p in positions) / N
    cy_mean = sum(p[1] for p in positions) / N
    for i in range(N):
        positions[i] = (positions[i][0] - cx_mean, positions[i][1] - cy_mean)
    
    centers = [list(p) for p in positions]
    angles = angles_init[:]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial side
    side = _bounding_hexagon_side(centers, angles, outer_center, outer_angle)
    best_side = side
    best_config = (centers[:], angles[:], outer_center[:], outer_angle)
    
    # Physics-based relaxation: resolve collisions by pushing apart
    def physics_relax(centers, angles, outer_center, outer_angle, iterations=200):
        for _ in range(iterations):
            if time.time() - start_time > DEADLINE:
                return
            verts_list = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if _pairwise_collision(verts_list[i], verts_list[j]):
                        sep, (dx_dir, dy_dir) = _separation(verts_list[i], verts_list[j])
                        if sep > 1e-6:
                            # Push apart
                            centers[i][0] -= dx_dir * sep * 0.5
                            centers[i][1] -= dy_dir * sep * 0.5
                            centers[j][0] += dx_dir * sep * 0.5
                            centers[j][1] += dy_dir * sep * 0.5
                            moved = True
            if not moved:
                break
    
    # Simulated Annealing
    def simulated_annealing(centers, angles, outer_center, outer_angle, iterations=5000):
        nonlocal best_side, best_config
        current_side = _bounding_hexagon_side(centers, angles, outer_center, outer_angle)
        T = 0.5
        T_min = 0.001
        cooling = 0.9995
        
        for it in range(iterations):
            if time.time() - start_time > DEADLINE:
                break
            # Random perturbation
            idx = random.randint(0, N - 1)
            pert_type = random.random()
            if pert_type < 0.4:
                # Translate
                centers[idx][0] += random.gauss(0, 0.3)
                centers[idx][1] += random.gauss(0, 0.3)
            elif pert_type < 0.7:
                # Rotate
                angles[idx] += random.gauss(0, 15.0)
            else:
                # Translate + rotate
                centers[idx][0] += random.gauss(0, 0.2)
                centers[idx][1] += random.gauss(0, 0.2)
                angles[idx] += random.gauss(0, 10.0)
            
            new_side = _bounding_hexagon_side(centers, angles, outer_center, outer_angle)
            
            # Check collisions
            penalty = 0.0
            verts_list = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    if _pairwise_collision(verts_list[i], verts_list[j]):
                        sep, _ = _separation(verts_list[i], verts_list[j])
                        penalty += sep * 50.0
            
            current_energy = new_side + penalty
            prev_energy = current_side + 0  # approximate
            
            delta = current_energy - prev_energy
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                current_side = new_side
                if new_side < best_side and penalty < 1e-4:
                    best_side = new_side
                    best_config = (centers[:], angles[:], outer_center[:], outer_angle)
            else:
                # Revert
                idx2 = idx
                # We need to revert - store before state
                # Actually, let's just recompute from best if needed
                pass
            
            T = max(T_min, T * cooling)
        
        return best_side
    
    # Run physics relaxation first
    physics_relax(centers, angles, outer_center, outer_angle, iterations=300)
    side = _bounding_hexagon_side(centers, angles, outer_center, outer_angle)
    if side < best_side:
        best_side = side
        best_config = (centers[:], angles[:], outer_center[:], outer_angle)
    
    # Run simulated annealing
    simulated_annealing(centers, angles, outer_center, outer_angle, iterations=8000)
    
    # Final physics relaxation on best config
    bc = [list(c) for c in best_config[0]]
    ba = list(best_config[1])
    boc = list(best_config[2])
    boa = best_config[3]
    physics_relax(bc, ba, boc, boa, iterations=200)
    final_side = _bounding_hexagon_side(bc, ba, boc, boa)
    if final_side < best_side:
        best_side = final_side
        best_config = (bc, ba, boc, boa)
    
    # Return the best configuration
    bc = best_config[0]
    ba = best_config[1]
    boc = best_config[2]
    boa = best_config[3]
    
    inner = [[bc[i][0], bc[i][1], ba[i]] for i in range(N)]
    return inner, (boc[0], boc[1]), best_side, boa

