"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: use known dense hexagonal lattice arrangements, then optimize
outer hexagon center, side, and angle to minimize enclosing side length.
Uses exact computation of minimum enclosing hexagon for fixed orientation.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 145.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        # The normal to edge i points in direction angle + (i+0.5)*pi/3
        mid_angle = angle + np.pi * (i + 0.5) / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Given inner hexagon centers and angles, and outer hexagon angle,
    compute the minimum side length of a regular hexagon (with that angle)
    that encloses all inner hexagons. Also return the optimal center.
    
    For a regular hexagon with side s and angle a, a point (px,py) is inside
    iff for all 6 edge normals n_i: dot((px,py) - center, n_i) <= s * inradius
    where inradius = s * sqrt(3)/2.
    
    So: dot(point - center, n_i) <= s * sqrt(3)/2 for all i
    => s >= dot(point - center, n_i) / (sqrt(3)/2) for all i
    => s >= max_i dot(point - center, n_i) / (sqrt(3)/2)
    
    To minimize s over center, we need to find center that minimizes
    max_i dot(point_i - center, n_i) over all points and all i.
    
    This is equivalent to: minimize max_i (dot(point_i, n_i) - dot(center, n_i))
    = max_i dot(point_i, n_i) - dot(center, n_i)
    
    For a fixed set of constraint directions, the optimal center is the
    center of the minimum enclosing circle in the dual space... but for a
    hexagon, we can use the fact that the optimal center is where the
    6 constraints are balanced.
    
    Actually, for a regular hexagon enclosing a set of points, the minimum
    side length hexagon is found by: for each of the 6 edge normals, the
    constraint is dot(p - c, n) <= r where r = s*sqrt(3)/2.
    So s = max_i (max_p dot(p, n_i) - dot(c, n_i)) / (sqrt(3)/2)
    
    We want to minimize over c: max_i (max_p dot(p, n_i) - dot(c, n_i))
    
    Let A_i = max_p dot(p, n_i) for each direction i.
    We want to minimize max_i (A_i - dot(c, n_i)).
    
    This is a convex optimization: minimize t subject to dot(c, n_i) >= A_i - t for all i.
    By symmetry of the hexagon, the optimal solution has the property that
    at least 3 constraints are tight.
    
    For a regular hexagon with normals at angles 30, 90, 150, 210, 270, 330 (for angle=0),
    the optimal center can be found by solving the 2D problem.
    
    Let's use a simpler approach: the optimal center is the centroid of the
    "extreme" points in each direction, adjusted. Actually, let's just use
    the fact that for a regular hexagon, the minimum enclosing hexagon
    (with fixed orientation) has its center at the point that equalizes
    the maximum projections.
    
    Simple approach: the center that minimizes max_i (A_i - dot(c,n_i))
    is found by noting that the gradient of the objective w.r.t. c is
    -n_i for the active constraint i. At optimum, sum of active normals * weights = 0.
    
    For a hexagon, at most 3 constraints are active and their normals sum to 0
    (since they're 120 degrees apart in the hexagon's normal set... wait, no).
    
    Let me just use a numerical approach: minimize over c using the fact that
    the objective is convex. I'll use a simple iterative method.
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)  # 6x2 array
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        all_points.append(verts)
    all_points = np.vstack(all_points)  # (6*N, 2)
    
    # For each normal direction, compute max projection
    # A_i = max_p dot(p, n_i)
    projections = all_points @ normals.T  # (6N, 6)
    A = projections.max(axis=0)  # (6,)
    
    # Now minimize over c: max_i (A_i - dot(c, n_i))
    # This is equivalent to: find c minimizing max_i (A_i - c·n_i)
    # Let f(c) = max_i (A_i - c·n_i)
    # f is convex in c.
    
    # Use subgradient method or just try the centroid of extreme points
    # Actually, for a regular hexagon, the optimal c satisfies:
    # The 6 values (A_i - c·n_i) should be as equal as possible.
    # By symmetry, if we set c = 0, we get A_i - 0 = A_i.
    # The optimal c shifts to balance these.
    
    # Simple approach: iterate c update
    c = np.array([0.0, 0.0])
    for _ in range(2000):
        vals = A - c @ normals.T  # (6,)
        # Active constraints are those with max val
        max_val = vals.max()
        active = vals >= max_val - 1e-12
        # Gradient of f w.r.t. c is -average of active normals (subgradient)
        # Move c in direction of active normals to reduce the max
        grad = normals[active].mean(axis=0)  # direction to move c
        # c_new = c + alpha * grad
        alpha = 0.1
        c_new = c + alpha * grad
        # Check if it improves
        new_vals = A - c_new @ normals.T
        if new_vals.max() < max_val:
            c = c_new
        else:
            alpha *= 0.5
            if alpha < 1e-12:
                break
            c = c + alpha * grad
    
    # Compute side length
    vals = A - c @ normals.T
    max_proj = vals.max()
    side = max_proj / (sqrt3 / 2)
    
    return side, c


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check if all inner hexagons are inside outer and don't overlap."""
    sqrt3 = np.sqrt(3)
    
    # Check containment: each inner vertex must be inside outer hexagon
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check non-overlap between inner hexagons
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using separating axis theorem."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1 @ np.array([ax, ay])
        proj2 = v2 @ np.array([ax, ay])
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _optimize_outer(centers, inner_angles):
    """For fixed inner positions, find optimal outer hexagon (center, side, angle)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a range of outer angles
    for outer_angle in np.linspace(0, 60, 61):
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return the best side length found."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Verify no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    # If not valid, try to fix by adjusting
    return float('inf'), None


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate various hexagonal lattice arrangements
    arrangements = []
    
    # Arrangement 1: Center + 6 ring + 4 second ring (standard)
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # Try all combinations of 4 from the 12 second-ring positions
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # First ring
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Try all C(12,4) = 495 subsets of second ring
    from itertools import combinations as comb
    for subset in comb(range(12), 4):
        if time.time() > DEADLINE:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        side, result = _try_arrangement(centers)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Also try with rotated inner hexagons (30 degree rotation)
    if time.time() < DEADLINE - 30:
        for subset in comb(range(12), 4):
            if time.time() > DEADLINE:
                break
            centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
            # Rotate all inner hexagons by 30 degrees
            inner_angles = [30.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Try different lattice spacings (slightly compressed or expanded)
    if time.time() < DEADLINE - 20:
        for scale in [0.98, 0.99, 1.0, 1.01, 1.02]:
            if time.time() > DEADLINE:
                break
            d1s = d1 * scale
            d2s = d2 * scale
            first_ring_s = []
            for i in range(6):
                angle = np.pi / 3 * i
                first_ring_s.append((d1s * np.cos(angle), d1s * np.sin(angle)))
            second_ring_s = []
            for i in range(6):
                angle = np.pi / 3 * i
                second_ring_s.append((d2s * np.cos(angle), d2s * np.sin(angle)))
                angle2 = np.pi / 3 * i + np.pi / 6
                second_ring_s.append((d2s * np.cos(angle2), d2s * np.sin(angle2)))
            
            # Just try the standard subset
            subset = [1, 3, 5, 7]  # alternating positions
            centers = [(0.0, 0.0)] + first_ring_s + [second_ring_s[i] for i in subset]
            side, result = _try_arrangement(centers)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
