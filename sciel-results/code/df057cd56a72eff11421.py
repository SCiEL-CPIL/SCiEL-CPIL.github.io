"""Unit regular hexagons inside a regular hexagon - packing optimizer."""
import numpy as np
import time
import random

N = 11
_DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, theta, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), rotated by theta radians."""
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, theta, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with rotation theta."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    ct, st = np.cos(-theta), np.sin(-theta)
    lx = dx * ct - dy * st
    ly = dx * st + dy * ct
    # Regular hexagon with side length s: vertices at angles 0, 60, ..., 300
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with flat-top orientation (theta=0), vertices at (s,0), (s/2, s*sqrt(3)/2), ...
    # Half-plane constraints: |lx| <= s, |ly| <= s*sqrt(3)/2, |lx/2 + ly*sqrt(3)/2| <= s
    # Actually let me use the standard form: for a regular hexagon with circumradius R=side,
    # the inradius is R*sqrt(3)/2
    # The six constraints are: for each edge, dot with outward normal <= inradius
    # Outward normals for hexagon with vertices at angles 0,60,...: 
    # Edge between v0 and v1 has outward normal at angle 30 degrees
    # More simply: the hexagon with circumradius R has support function h(n) = R*sqrt(3)/2 * cos(pi/6 * (angle of n - 30deg mod 60))... 
    # Let me just use the 6 half-plane test directly.
    # For a regular hexagon with circumradius R, the six edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees
    # and the distance from center to each edge is R * cos(pi/6) = R * sqrt(3)/2
    R = side
    inradius = R * np.sqrt(3.0) / 2.0
    normals_angles = np.pi / 6 + np.pi / 3 * np.arange(6)  # 30, 90, 150, 210, 270, 330
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    dists = np.array([lx, ly]) @ normals.T
    return np.all(dists <= inradius + 1e-9)

def _hex_min_enclosing_side(vertices_list):
    """Compute the minimum side length of a regular hexagon that encloses all given vertices.
    A regular hexagon with side S centered at origin contains all points with:
    For each of the 6 edge normals at angles 30+60k, dot(p, n_k) <= S*sqrt(3)/2
    So S >= max_k |dot(p, n_k)| / (sqrt(3)/2) = 2*max_k|dot(p,n_k)|/sqrt(3)
    But we also need to center it optimally. For a given rotation, the optimal center is the midpoint
    of the min and max projections on each axis. The side needed is (max - min)/(sqrt(3)) for each pair.
    Actually for a regular hexagon, the constraint on each edge normal direction n_k is:
    (max_k - min_k) of dot(p, n_k) <= 2 * S * sqrt(3)/2 = S*sqrt(3)
    So S >= (max_k - min_k) / sqrt(3) for each k, and S is the max over k.
    The center is at the midpoint of projections.
    We also need to check that the hexagon with that S and center actually contains all points,
    which requires checking all 6 edge constraints simultaneously.
    """
    all_pts = np.vstack(vertices_list)
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for rot in np.linspace(0, np.pi/3, 7, endpoint=False):
        normals_angles = rot + np.pi/6 + np.pi/3 * np.arange(6)
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        projections = all_pts @ normals.T  # shape (n, 6)
        mins = projections.min(axis=0)
        maxs = projections.max(axis=0)
        widths = maxs - mins
        required_sides = widths / np.sqrt(3.0)
        s = required_sides.max()
        if s < best_side:
            # Optimal center: midpoint of projections for the tightest direction
            # For a regular hexagon, the center in each normal direction is at (min+max)/2
            # But these must be consistent. The optimal center minimizes the max over k of
            # max(|dot(c - center, n_k)|) subject to containment.
            # For now, use the center that makes the widest direction centered:
            tightest = np.argmax(required_sides)
            center_proj = (mins + maxs) / 2.0
            # Convert center projections back to x,y: we need to solve
            # center @ normals.T = center_proj
            # This is a 2x6 system, solve least squares
            center, _, _, _ = np.linalg.lstsq(normals.T, center_proj, rcond=None)
            best_side = s
            best_center = (center[0], center[1])
            best_angle = rot
    return best_side, best_center, best_angle

def _hex_overlap(cx1, cy1, th1, cx2, cy2, th2, side=1.0, tol=1e-7):
    """Check if two regular hexagons overlap using separating axis theorem."""
    if (cx1 - cx2)**2 + (cy1 - cy2)**2 > (4 * side)**2:  # quick reject: max distance between centers
        return False
    v1 = _hex_vertices(cx1, cy1, th1, side)
    v2 = _hex_vertices(cx2, cy2, th2, side)
    # Get all edge normals from both hexagons
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1) % 6] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        if proj1.max() < proj2.min() + tol or proj2.max() < proj1.min() + tol:
            return False
    return True

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check that all inner hexagons are disjoint and inside the outer hexagon."""
    cx, cy = outer_center
    oc, oa = outer_center[0], outer_center[1]
    # Check inner hexagons don't overlap each other
    for i in range(len(inner)):
        x1, y1, a1 = inner[i]
        th1 = np.radians(a1)
        for j in range(i+1, len(inner)):
            x2, y2, a2 = inner[j]
            th2 = np.radians(a2)
            if _hex_overlap(x1, y1, th1, x2, y2, th2, tol=tol):
                return False
    # Check each inner hexagon is inside the outer hexagon
    outer_verts = _hex_vertices(cx, cy, np.radians(outer_angle), outer_side)
    for i in range(len(inner)):
        x1, y1, a1 = inner[i]
        th1 = np.radians(a1)
        inner_verts = _hex_vertices(x1, y1, th1, 1.0)
        for vx, vy in inner_verts:
            if not _point_in_hex(vx, vy, cx, cy, np.radians(outer_angle), outer_side, tol=tol):
                return False
    return True

def _optimize_outer(inner):
    """Given inner hexagon positions, find the minimum enclosing regular hexagon."""
    all_verts = []
    for x, y, a in inner:
        th = np.radians(a)
        all_verts.append(_hex_vertices(x, y, th, 1.0))
    side, center, angle = _hex_min_enclosing_side(all_verts)
    # Verify
    if _check_validity(inner, center, side, np.degrees(angle)):
        return side, center, angle
    # If not valid (shouldn't happen with exact computation), expand slightly
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        if _check_validity(inner, center, side + eps, np.degrees(angle)):
            return side + eps, center, angle
    return side, center, angle

def _try_arrangement(positions, angles):
    """Try an arrangement and return (side, center, angle) if valid."""
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(len(positions))]
    side, center, ang = _optimize_outer(inner)
    return side, center, ang, inner

def _generate_hexagonal_lattice():
    """Generate positions in a hexagonal lattice pattern."""
    positions = []
    angles = [0.0] * N
    # Center hexagon
    positions.append([0.0, 0.0])
    # Ring 1: 6 hexagons at distance 2 (center-to-center for touching unit hexagons is 2)
    for k in range(6):
        ang = np.pi / 3 * k
        positions.append([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
    # Ring 2: we need 4 more
    # Positions at distance ~4 from center in various directions
    for k in range(6):
        ang = np.pi / 3 * k + np.pi/6
        positions.append([3.464 * np.cos(ang), 3.464 * np.sin(ang)])  # 2*sqrt(3)
    # Take first N
    return positions[:N], angles[:N]

def _generate_star():
    """Star pattern."""
    positions = [[0.0, 0.0]]
    angles = [0.0]
    for k in range(N-1):
        ang = 2 * np.pi * k / (N-1)
        r = 2.0
        positions.append([r * np.cos(ang), r * np.sin(ang)])
        angles.append(0.0)
    return positions, angles

def _generate_line():
    """Linear arrangement."""
    positions = []
    angles = []
    for i in range(N):
        positions.append([(i - N//2) * 2.0, 0.0])
        angles.append(0.0)
    return positions, angles

def _generate_spiral():
    """Spiral arrangement."""
    positions = [[0.0, 0.0]]
    angles = [0.0]
    r = 0.0
    ang = 0.0
    for i in range(1, N):
        ang += 2.0
        r += 0.8
        positions.append([r * np.cos(ang), r * np.sin(ang)])
        angles.append(0.0)
    return positions, angles

def _force_directed_relax(positions, angles, max_iters=500, lr=0.1):
    """Force-directed relaxation to resolve overlaps."""
    positions = np.array(positions, dtype=float).copy()
    angles_arr = np.array(angles, dtype=float).copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        forces = np.zeros_like(positions)
        overlap_count = 0
        for i in range(N):
            for j in range(i+1, N):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    d = np.array([random.random()-0.5, random.random()-0.5])
                    dist = np.linalg.norm(d)
                if dist < 2.0:  # overlapping or close
                    overlap_count += 1
                    # Push apart
                    f = (2.0 - dist) / dist * d * 2.0
                    forces[i] += f
                    forces[j] -= f
        positions += lr * forces
        # Dampen
        lr *= 0.995
        if overlap_count == 0:
            break
    return positions.tolist(), angles_arr.tolist()

def _simulate_annealing(positions, angles, n_iters=2000, t_start=1.0, t_end=0.01):
    """Simulated annealing to minimize enclosing hexagon side."""
    positions = np.array(positions, dtype=float).copy()
    angles_arr = np.array(angles, dtype=float).copy()
    
    inner = [[positions[i][0], positions[i][1], angles_arr[i]] for i in range(N)]
    cur_side, cur_center, cur_angle = _optimize_outer(inner)
    best_side = cur_side
    best_positions = positions.copy()
    best_angles = angles_arr.copy()
    
    if cur_side == float('inf') or not np.isfinite(cur_side):
        # Invalid, try to fix
        positions, angles_list = _force_directed_relax(positions.tolist(), angles_arr.tolist())
        inner = [[positions[i][0], positions[i][1], angles_list[i]] for i in range(N)]
        cur_side, cur_center, cur_angle = _optimize_outer(inner)
        best_side = cur_side
        best_positions = positions.copy()
        best_angles = np.array(angles_list)
    
    for it in range(n_iters):
        if time.time() > _DEADLINE - 10:
            break
        t = t_start * (t_end / t_start) ** (it / n_iters)
        
        # Random perturbation
        new_positions = positions + np.random.randn(N, 2) * 0.3
        new_angles = angles_arr + np.random.randn(N) * 10.0
        new_angles = np.degrees(np.arctan2(np.sin(np.radians(new_angles)), 
                                           np.cos(np.radians(new_angles))))
        
        inner = [[new_positions[i][0], new_positions[i][1], new_angles[i]] for i in range(N)]
        new_side, _, _ = _optimize_outer(inner)
        
        if not np.isfinite(new_side):
            new_side = 1e10
        
        delta = new_side - cur_side
        if delta < 0 or random.random() < np.exp(-delta / max(t, 1e-9)):
            cur_side = new_side
            positions = new_positions
            angles_arr = new_angles
            if cur_side < best_side:
                best_side = cur_side
                best_positions = positions.copy()
                best_angles = angles_arr.copy()
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    side, center, angle = _optimize_outer(inner)
    return side, center, angle, inner

def _get_valid_fallback():
    """Return a valid (though large) arrangement as fallback."""
    positions, angles = _generate_hexagonal_lattice()
    positions, angles = _force_directed_relax(positions, angles, max_iters=100)
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    side, center, angle = _optimize_outer(inner)
    if not np.isfinite(side) or side > 100:
        # Last resort: put them in a line with big margin
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return inner, center, angle

def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    generators = [_generate_hexagonal_lattice, _generate_star, _generate_line, _generate_spiral]
    
    best_result = None
    best_side = float('inf')
    
    for gen_func in generators:
        if time.time() > _DEADLINE - 30:
            break
        
        positions, angles = gen_func()
        # Force-directed relaxation first
        positions, angles = _force_directed_relax(positions, angles, max_iters=200)
        
        # Evaluate
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        side, center, angle = _optimize_outer(inner)
        
        if np.isfinite(side) and side < best_side:
            best_side = side
            best_result = (inner, center, angle)
        
        # Simulated annealing from this configuration
        if time.time() > _DEADLINE - 20:
            break
        
        sa_side, sa_center, sa_angle, sa_inner = _simulate_annealing(
            np.array(positions), np.array(angles), 
            n_iters=1500, t_start=0.5, t_end=0.01
        )
        
        if np.isfinite(sa_side) and sa_side < best_side:
            best_side = sa_side
            best_result = (sa_inner, sa_center, sa_angle)
    
    # Additional multi-start with random perturbations of best
    if best_result is not None and time.time() < _DEADLINE - 15:
        base_inner = best_result[0]
        base_positions = np.array([[p[0], p[1]] for p in base_inner])
        base_angles = np.array([p[2] for p in base_inner])
        
        for restart in range(3):
            if time.time() > _DEADLINE - 10:
                break
            perturbed_pos = base_positions + np.random.randn(N, 2) * 0.5
            perturbed_ang = base_angles + np.random.randn(N) * 15.0
            
            sa_side, sa_center, sa_angle, sa_inner = _simulate_annealing(
                perturbed_pos, perturbed_ang,
                n_iters=1000, t_start=0.3, t_end=0.01
            )
            if np.isfinite(sa_side) and sa_side < best_side:
                best_side = sa_side
                best_result = (sa_inner, sa_center, sa_angle)
    
    if best_result is None:
        inner, center, side, angle = _get_valid_fallback()
        return inner, center, side, angle
    
    inner, center, angle = best_result
    # Final verification
    if not _check_validity(inner, center, angle, 0.0):
        # Slightly expand
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            if _check_validity(inner, center, angle + eps, 0.0):
                return inner, center, angle + eps, 0.0
        # Fallback
        fb_inner, fb_center, fb_side, fb_angle = _get_valid_fallback()
        return fb_inner, fb_center, fb_side, fb_angle
    
    return inner, center, angle, 0.0
