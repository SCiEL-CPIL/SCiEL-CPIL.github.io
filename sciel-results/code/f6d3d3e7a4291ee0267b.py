"""Packing 11 unit regular hexagons inside a regular hexagon.

Structural change: Binary search on container size + physics-based
force-directed relaxation with Verlet integration and proper hexagon-hexagon
repulsion (using SAT-based minimum distance, not center-distance proxy).
"""
import numpy as np
import time
import random
import math
import sys

N = 11
DEADLINE = 145.0
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get 6 vertices of a regular hexagon centered at (cx,cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _hex_edge_normals(cx, cy, angle_deg, side=1.0):
    """Get 6 outward edge normals of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + math.radians(30) + k * math.pi / 3
        normals.append((math.cos(a), math.sin(a)))
    return normals


def _point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon at (cx,cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    ca = math.cos(-angle_rad)
    sa = math.sin(-angle_rad)
    dx = px - cx
    dy = py - cy
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    inradius = side * math.sqrt(3) / 2.0
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        if lx * nx + ly * ny > inradius:
            return False
    return True


def _hex_min_distance(v1, v2):
    """
    Compute minimum distance between two hexagons (lists of 6 vertices each).
    Returns (distance, push_x, push_y) where push is direction to separate.
    Uses SAT: find axis of minimum penetration.
    """
    min_dist = float('inf')
    best_push = (0.0, 0.0)
    
    all_normals = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            ex = verts[j][0] - verts[i][0]
            ey = verts[j][1] - verts[i][1]
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                all_normals.append((1.0, 0.0))
            else:
                all_normals.append((-ey / length, ex / length))
    
    for nx, ny in all_normals:
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        # Gap between the two projections
        if max1 <= min2:
            gap = min2 - max1
            if gap < min_dist:
                min_dist = gap
                best_push = (nx, ny)
        elif max2 <= min1:
            gap = min1 - max2
            if gap < min_dist:
                min_dist = gap
                best_push = (-nx, -ny)
        else:
            # Overlapping on this axis
            return (0.0, 0.0, 0.0)
    
    return (min_dist, best_push[0], best_push[1])


def _hexagons_disjoint(cx1, cy1, a1, cx2, cy2, a2, tol=1e-7):
    """Check if two unit hexagons are disjoint using SAT."""
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    all_norms = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            ex = verts[j][0] - verts[i][0]
            ey = verts[j][1] - verts[i][1]
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                all_norms.append((1.0, 0.0))
            else:
                all_norms.append((-ey / length, ex / length))
    
    for nx, ny in all_norms:
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        if max(proj1) < min(proj2) - tol or max(proj2) < min(proj1) - tol:
            return True
    return False


def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    oa_rad = math.radians(outer_angle)
    ca = math.cos(-oa_rad)
    sa = math.sin(-oa_rad)
    
    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            
            dx = vx - cx
            dy = vy - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = math.sqrt(lx * lx + ly * ly)
            if r < 1e-12:
                continue
            
            theta = math.atan2(ly, lx)
            best_cos = 0.0
            for m in range(6):
                am = math.radians(30 + m * 60)
                d = abs(((theta - am) % (2 * math.pi) + math.pi) % (2 * math.pi))
                d = min(d, 2 * math.pi - d)
                c = math.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / math.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side


def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if configuration is valid: all inner hexagons inside outer and disjoint."""
    cx, cy = outer_center
    inradius = outer_side * math.sqrt(3) / 2.0
    oa_rad = math.radians(outer_angle)
    ca = math.cos(-oa_rad)
    sa = math.sin(-oa_rad)
    
    # Check containment
    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            
            dx = vx - cx
            dy = vy - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            for m in range(6):
                am = math.radians(30 + m * 60)
                nx = math.cos(am)
                ny = math.sin(am)
                if lx * nx + ly * ny > inradius + tol:
                    return False
    
    # Check disjointness
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(inner_positions[i][0], inner_positions[i][1],
                                      inner_positions[i][2],
                                      inner_positions[j][0], inner_positions[j][1],
                                      inner_positions[j][2], tol=tol):
                return False
    
    return True


def _force_directed_relax(positions, outer_side, outer_angle, max_iters=500, damping=0.9):
    """
    Force-directed relaxation using Verlet-like integration.
    Forces:
    - Repulsion between overlapping hexagons (SAT-based)
    - Containment: push hexagons toward center if vertices exceed boundary
    """
    pos = [list(p) for p in positions]
    vel = [[0.0, 0.0] for _ in range(N)]
    
    oa_rad = math.radians(outer_angle)
    outer_normals = []
    for k in range(6):
        a_local = math.radians(30 + 60 * k)
        a_world = a_local + oa_rad
        outer_normals.append((math.cos(a_world), math.sin(a_world)))
    
    inradius = outer_side * math.sqrt(3) / 2.0
    
    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break
        
        forces = [[0.0, 0.0] for _ in range(N)]
        
        # Repulsion between hexagons
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ai = pos[i]
                xj, yj, aj = pos[j]
                
                # Quick check: if centers are far apart, skip
                dx = xj - xi
                dy = yj - yi
                dist_sq = dx * dx + dy * dy
                # Minimum center distance for non-overlapping hexagons ~ 2*sqrt(3)
                if dist_sq > 20.0:  # dist > ~4.5, definitely no overlap
                    continue
                
                v1 = _hex_vertices(xi, yi, ai)
                v2 = _hex_vertices(xj, yj, aj)
                
                dist, px, py = _hex_min_distance(v1, v2)
                
                if dist < 0.01:  # Overlapping or very close
                    # Strong repulsion
                    strength = 2.0 * max(0.01 - dist, 0.01)
                    forces[i][0] -= strength * px
                    forces[i][1] -= strength * py
                    forces[j][0] += strength * px
                    forces[j][1] += strength * py
        
        # Containment force: push vertices back inside
        for i in range(N):
            xi, yi, ai = pos[i]
            angle_rad = math.radians(ai)
            for k in range(6):
                a = angle_rad + k * math.pi / 3
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                
                for nx, ny in outer_normals:
                    proj = vx * nx + vy * ny
                    violation = proj - inradius
                    if violation > 0:
                        # Push hexagon center in opposite direction
                        strength = 3.0 * violation
                        forces[i][0] -= strength * nx
                        forces[i][1] -= strength * ny
        
        # Mild attraction to center for compactness
        for i in range(N):
            forces[i][0] += 0.005 * pos[i][0]
            forces[i][1] += 0.005 * pos[i][1]
        
        # Update positions with damping
        for i in range(N):
            vel[i][0] = damping * vel[i][0] + forces[i][0] * 0.02
            vel[i][1] = damping * vel[i][1] + forces[i][1] * 0.02
            # Clamp velocity
            vlen = math.sqrt(vel[i][0]**2 + vel[i][1]**2)
            if vlen > 0.5:
                vel[i][0] *= 0.5 / vlen
                vel[i][1] *= 0.5 / vlen
            pos[i][0] += vel[i][0]
            pos[i][1] += vel[i][1]
    
    return pos


def _local_search(positions, outer_angle, time_budget=5.0):
    """Local search: try small perturbations to improve the enclosing side."""
    best_pos = [list(p) for p in positions]
    best_side = _min_enclosing_side(best_pos, (0.0, 0.0), outer_angle)
    start_time = time.time()
    
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        if _remaining_time() < 3.0:
            break
        
        # Pick a random hexagon and perturb it
        i = random.randint(0, N - 1)
        new_pos = [list(p) for p in best_pos]
        
        # Small random displacement
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        new_pos[i][0] += dx
        new_pos[i][1] += dy
        
        # Optionally perturb angle slightly
        if random.random() < 0.3:
            new_pos[i][2] += random.gauss(0, 5)
            new_pos[i][2] %= 360
        
        # Check if this is valid and better
        side = _min_enclosing_side(new_pos, (0.0, 0.0), outer_angle)
        if side < best_side - 1e-6:
            # Verify validity
            if _check_validity(new_pos, (0.0, 0.0), side, outer_angle, tol=1e-4):
                best_pos = new_pos
                best_side = side
    
    return best_pos, best_side


def _initial_lattice():
    """Hexagonal lattice arrangement: center + 6 around + 4 more."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * math.sqrt(3)
    for k in range(6):
        a = math.radians(60 * k)
        positions.append([dist * math.cos(a), dist * math.sin(a), 0.0])
    dist2 = dist * math.sqrt(3)
    for k in range(4):
        a = math.radians(30 + 60 * k)
        positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
    return positions


def _initial_circular():
    """Circular arrangement: center + ring."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        a = 2 * math.pi * i / (N - 1)
        r = 3.5
        positions.append([r * math.cos(a), r * math.sin(a), 0.0])
    return positions


def _initial_zigzag():
    """Zigzag/brick arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # Two rows offset
    row1_x = [2.0 * math.sqrt(3), 4.0 * math.sqrt(3), -2.0 * math.sqrt(3), -4.0 * math.sqrt(3)]
    row1_y = [0.0, 0.0, 0.0, 0.0]
    row2_x = [math.sqrt(3), 3.0 * math.sqrt(3), -math.sqrt(3), -3.0 * math.sqrt(3)]
    row2_y = [3.0, 3.0, -3.0, -3.0]
    
    for i in range(4):
        positions.append([row1_x[i], row1_y[i], 0.0])
    for i in range(4):
        positions.append([row2_x[i], row2_y[i], 0.0])
    return positions


def _initial_random():
    """Random initial positions."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.5, 4.5)
        a = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, 360)
        positions.append([r * math.cos(a), r * math.sin(a), angle])
    return positions


def _try_arrangement(positions, outer_angle):
    """Try to find a valid arrangement using relaxation + local search."""
    if _remaining_time() < 5.0:
        return None, float('inf')
    
    # Start with a reasonable estimate of outer side
    init_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    
    # Binary search: try to fit in smaller and smaller containers
    best_pos = None
    best_side = float('inf')
    
    # First, relax at the initial size to get a valid config
    relaxed = _force_directed_relax(positions, init_side * 1.1, outer_angle, max_iters=300)
    relaxed_side = _min_enclosing_side(relaxed, (0.0, 0.0), outer_angle)
    
    if _check_validity(relaxed, (0.0, 0.0), relaxed_side, outer_angle, tol=1e-4):
        best_pos = relaxed
        best_side = relaxed_side
    
    # Binary search for smaller container
    lo = best_side * 0.7 if best_pos else init_side * 0.7
    hi = best_side if best_pos else init_side * 1.1
    
    for _ in range(15):
        if _remaining_time() < 5.0:
            break
        
        mid = (lo + hi) / 2
        
        # Try to relax at this size
        relaxed = _force_directed_relax(positions, mid, outer_angle, max_iters=400)
        relaxed_side = _min_enclosing_side(relaxed, (0.0, 0.0), outer_angle)
        
        if _check_validity(relaxed, (0.0, 0.0), relaxed_side, outer_angle, tol=1e-4):
            if relaxed_side < best_side:
                best_side = relaxed_side
                best_pos = relaxed
            hi = mid
        else:
            lo = mid
    
    # Local search refinement
    if best_pos is not None:
        ls_pos, ls_side = _local_search(best_pos, outer_angle, time_budget=3.0)
        if ls_side < best_side:
            best_pos = ls_pos
            best_side = ls_side
    
    if best_pos is not None:
        final_side = _min_enclosing_side(best_pos, (0.0, 0.0), outer_angle)
        if _check_validity(best_pos, (0.0, 0.0), final_side, outer_angle, tol=1e-4):
            return best_pos, final_side
    
    return None, float('inf')


def construct():
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0]
    
    initial_configs = [
        _initial_lattice(),
        _initial_circular(),
        _initial_zigzag(),
        _initial_random(),
        _initial_random(),
        _initial_random(),
    ]
    
    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break
        
        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break
            
            positions = [list(p) for p in init_pos]
            result, side = _try_arrangement(positions, oa)
            
            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: lattice
    positions = _initial_lattice()
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0


# Test
if __name__ == '__main__':
    result = construct()
    inner, center, side, angle = result
    print(f"Side length: {side:.6f}")
    print(f"Center: {center}")
    print(f"Angle: {angle}")
    print(f"Inner positions: {len(inner)} hexagons")
    for p in inner:
        print(f"  ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.2f}°)")
    print(f"Valid: {_check_validity(inner, center, side, angle)}")
    print(f"Time: {time.time() - START_TIME:.1f}s")
