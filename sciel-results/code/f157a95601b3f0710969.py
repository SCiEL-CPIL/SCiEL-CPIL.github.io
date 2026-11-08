"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Use hexagonal lattice arrangements, optimize outer hexagon
via exact projection method, and refine inner positions with gradient descent.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 145.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3.0)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi * (i + 0.5) / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length and center of regular hexagon (with given angle)
    enclosing all inner hexagons.
    """
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([
        _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for i in range(len(centers))
    ])
    
    # A_i = max projection onto each normal
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Minimize max_i (A_i - c·n_i) over c
    # Use iterative balancing
    c = np.array([0.0, 0.0])
    for _ in range(5000):
        vals = A - c @ normals.T
        max_val = vals.max()
        active = vals >= max_val - 1e-10
        if active.sum() == 0:
            break
        grad = normals[active].mean(axis=0)
        alpha = 0.5
        for _ in range(50):
            c_new = c + alpha * grad
            new_max = (A - c_new @ normals.T).max()
            if new_max < max_val - 1e-15:
                c = c_new
                break
            alpha *= 0.5
        else:
            break
    
    vals = A - c @ normals.T
    side = vals.max() / (SQRT3 / 2)
    return side, c


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm > 1e-12:
                axes.append((-ey/norm, ex/norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        p1 = v1 @ np.array([ax, ay])
        p2 = v2 @ np.array([ax, ay])
        if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
            return False
    return True


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """Check containment and non-overlap."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                if (vx - outer_center[0])*nx + (vy - outer_center[1])*ny > inradius + tol:
                    return False
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    return True


def _optimize_outer(centers, inner_angles):
    """Find best outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def _try_arrangement(centers, inner_angles=None):
    """Evaluate an arrangement, return (side, result_tuple) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Quick overlap check
    for i in range(N):
        for j in range(i+1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None


def _refine_positions(centers, inner_angles, outer_center, outer_side, outer_angle, max_iter=2000):
    """Refine inner positions to reduce outer side length while maintaining validity."""
    centers = [list(c) for c in centers]
    inner_angles = list(inner_angles)
    best_side = outer_side
    best_centers = [list(c) for c in centers]
    best_angles = list(inner_angles)
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute current outer side
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        
        if side < best_side:
            best_side = side
            best_centers = [list(c) for c in centers]
            best_angles = list(inner_angles)
        
        # Compute gradient: for each inner hex, move to reduce max projection
        normals = _hex_edge_normals(outer_angle)
        all_points = np.vstack([
            _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
            for i in range(N)
        ])
        projections = all_points @ normals.T
        A = projections.max(axis=0)
        vals = A - center @ normals.T
        active = vals >= vals.max() - 1e-8
        
        # Gradient for each hex: sum of active normals weighted by which hex contributes
        grads = np.zeros((N, 2))
        for i in range(N):
            hex_proj = all_points[i*6:(i+1)*6] @ normals.T
            hex_max = hex_proj.max(axis=0)
            # This hex contributes to constraint i if its max is close to A_i
            contrib = np.where((active) & (np.abs(hex_max - A) < 0.01), 1.0, 0.0)
            if contrib.sum() > 0:
                grads[i] = normals[contrib > 0].mean(axis=0)
        
        # Apply gradient step
        for i in range(N):
            centers[i][0] -= lr * grads[i][0]
            centers[i][1] -= lr * grads[i][1]
        
        # Decay learning rate
        if it % 100 == 99:
            lr *= 0.9
        
        # Check validity
        if not _check_validity(centers, inner_angles, center, side, outer_angle):
            # Revert
            centers = [best_centers[i][:] for i in range(N)]
            inner_angles = best_angles[:]
            lr *= 0.5
    
    return best_side, best_centers, best_angles, center, best_side, outer_angle


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    d1 = SQRT3  # distance from center to first ring
    d2 = 2 * SQRT3  # distance to second ring
    
    # First ring positions
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring positions (12 positions)
    second_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # Try all C(12,4) subsets of second ring
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE - 20:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        side, result = _try_arrangement(centers)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try with 30-degree rotation of inner hexagons
    if time.time() < DEADLINE - 30:
        for subset in combinations(range(12), 4):
            if time.time() > DEADLINE - 20:
                break
            centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
            inner_angles = [30.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Refine the best arrangement
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, inner_angles, outer_center, outer_side, outer_angle = best_result
        refined_side, refined_centers, refined_angles, _, _, _ = _refine_positions(
            centers, inner_angles, outer_center, outer_side, outer_angle,
            max_iter=3000
        )
        if refined_side < best_side:
            side, center, angle = _optimize_outer(refined_centers, refined_angles)
            if _check_validity(refined_centers, refined_angles, center, side, angle):
                best_side = side
                best_result = (refined_centers, refined_angles, center, side, angle)
    
    # Fallback
    if best_result is None:
        centers = [(2 * SQRT3 * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
