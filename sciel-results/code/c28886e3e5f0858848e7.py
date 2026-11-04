"""Packing 11 unit hexagons in a regular hexagon - direct optimization."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def poly_distance(v1, v2):
    """Minimum distance between two convex polygons (positive if disjoint, 0 if touching).
    Using separating axis theorem for convex polygons."""
    n1, n2 = len(v1), len(v2)
    min_dist = float('inf')
    # Check all edges of both polygons as potential separating axes
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        if np.linalg.norm(normal) < 1e-12:
            continue
        normal = normal / np.linalg.norm(normal)
        # Project both polygons onto this normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If separated
        if max1 <= min2 + 1e-10 or max2 <= min1 + 1e-10:
            gap = max(min2 - max1, min1 - max2, 0)
            if gap < min_dist:
                min_dist = gap
            # Also check if one polygon is inside the other
            # For convex polygons, if no separating axis found, they overlap
            return 0.0
        else:
            # Overlap on this axis, check actual distance
            # Find closest points
            pass
    # If no separating axis, polygons overlap
    # Check if one is inside the other
    def point_in_poly(p, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            if ((poly[i][1] > p[1]) != (poly[j][1] > p[1])) and \
               (p[0] < (poly[j][0] - poly[i][0]) * (p[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
                inside = not inside
            j = i
        return inside
    
    # Check if any vertex of v1 is inside v2
    for v in v1:
        if point_in_poly(v, v2):
            return 0.0
    # Check if any vertex of v2 is inside v1
    for v in v2:
        if point_in_poly(v, v1):
            return 0.0
    return min_dist

def min_separation(params):
    """Return minimum separation between all pairs of inner hexagons.
    params: 3*N array = [x1,y1,a1, x2,y2,a2, ...] where a is in radians"""
    N_local = len(params) // 3
    min_sep = float('inf')
    verts = []
    for i in range(N_local):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        v = hex_vertices(x, y, 1.0, np.degrees(a))
        verts.append(v)
    for i in range(N_local):
        for j in range(i+1, N_local):
            d = poly_distance(verts[i], verts[j])
            if d < min_sep:
                min_sep = d
    return min_sep

def max_projection(params, outer_angle):
    """Return the maximum projection of all inner hexagon vertices onto the outer hexagon's constraint axes.
    This gives us the minimum side length needed for the outer hexagon."""
    N_local = len(params) // 3
    # Outer hexagon has 6 constraint directions (normals to its edges)
    # For a regular hexagon with rotation outer_angle, the normals are at angles:
    # outer_angle + i*pi/3 for i=0..5
    max_proj = 0.0
    for i in range(N_local):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        v = hex_vertices(x, y, 1.0, np.degrees(a))
        for k in range(6):
            theta = np.radians(outer_angle) + k * np.pi / 3.0
            normal = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(v, normal)
            if np.max(proj) > max_proj:
                max_proj = np.max(proj)
    return max_proj

def objective(params_with_outer):
    """Objective: outer side length + penalty for collisions.
    params_with_outer: [x1,y1,a1, ..., xN,yN,aN, outer_angle]"""
    N_local = N
    outer_angle = params_with_outer[-1]
    inner_params = params_with_outer[:3*N_local]
    
    # Check collisions
    min_sep = min_separation(inner_params)
    if min_sep < 0.0:
        # Collision penalty
        penalty = 100.0 * abs(min_sep)
    else:
        penalty = 0.0
    
    # Compute required outer side length
    max_proj = max_projection(inner_params, outer_angle)
    # The outer hexagon side length must be at least max_proj / cos(0) ... 
    # Actually for a regular hexagon, the inradius = side * sqrt(3)/2
    # The maximum projection onto a normal direction = inradius + (distance from center to farthest point)
    # Wait, let me think more carefully.
    
    # For a regular hexagon centered at origin with side s and rotation theta,
    # the vertices are at distance s from center.
    # The support function in direction u is: s * max over vertices of (v·u/|v|)
    # For a regular hexagon, the support function h(u) = s * max(cos(angle_diff)) where angle_diff is the angle between u and the nearest vertex direction.
    # The maximum of cos over the 6 vertex directions gives us the support.
    
    # Actually, the minimum enclosing regular hexagon (with given rotation) of a set of points:
    # We need the smallest s such that all points are inside.
    # A point p is inside the hexagon (centered at origin, side s, rotation theta) iff
    # for all 6 edge normals n_k: p·n_k <= s * cos(pi/6) = s * sqrt(3)/2
    # Wait no. Let me reconsider.
    
    # Regular hexagon with side s, centered at origin, one vertex at angle theta:
    # Vertices at angles theta + k*pi/3, distance s from center.
    # The inradius (distance from center to edge midpoint) = s * sqrt(3)/2
    # Edge normals are at angles theta + pi/6 + k*pi/3 (perpendicular to edges)
    # For a point p to be inside: p·n_k <= inradius = s*sqrt(3)/2 for all k
    
    # So the minimum s = max over all points p and all 6 normals n_k of (p·n_k) / (sqrt(3)/2)
    # But we can also translate the outer hexagon. Let's fix center at origin for now
    # and allow the inner hexagons to be placed anywhere.
    
    # Actually, we should optimize the outer center too. Let's simplify:
    # Fix outer center at origin. The inner hexagons can be anywhere.
    # Then s = max_{p in all vertices, k in 0..5} (p·n_k) / (sqrt(3)/2)
    # where n_k are the 6 edge normals of the outer hexagon.
    
    # But we want to minimize s, so we also want to center the cluster.
    # Let's compute the centroid of all inner vertices and shift everything.
    
    all_verts = []
    for i in range(N_local):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        v = hex_vertices(x, y, 1.0, np.degrees(a))
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    # Shift all points so centroid is at origin
    shifted = all_verts - centroid
    
    # Now compute min enclosing hexagon side length
    max_val = 0.0
    for k in range(6):
        theta = np.radians(outer_angle) + np.pi/6 + k * np.pi / 3.0
        normal = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(shifted, normal)
        if np.max(proj) > max_val:
            max_val = np.max(proj)
    
    s = max_val / (np.sqrt(3)/2)
    
    return s + penalty

def construct():
    """Main entry point."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Initial guess: hexagonal lattice
    def initial_params():
        params = []
        # Place 11 hexagons in a hexagonal arrangement
        # Center + 6 around + 4 more
        positions = [
            (0, 0),
            (2.2, 0), (1.1, 1.9), (-1.1, 1.9), (-2.2, 0), (-1.1, -1.9), (1.1, -1.9),
            (3.3, 1.9), (0, 3.8), (-3.3, 1.9), (3.3, -1.9)
        ]
        for x, y in positions:
            params.extend([x, y, 0.0])
        params.append(0.0)  # outer angle
        return params
    
    bounds = [(-10, 10), (-10, 10), (-np.pi, np.pi)] * N + [(-np.pi/6, np.pi/6)]
    
    best_result = None
    best_score = float('inf')
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            init='sobol'
        )
        if time.time() > deadline - 10:
            # Time's up, use best found so far
            pass
        else:
            score = result.fun
            if score < best_score:
                best_score = score
                best_result = result.x
    except Exception:
        pass
    
    if best_result is None:
        # Fallback: use initial guess
        best_result = np.array(initial_params())
        best_score = objective(best_result)
    
    # Extract parameters
    inner_params = best_result[:3*N]
    outer_angle = best_result[-1]
    
    # Compute centroid and shift
    all_verts = []
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        v = hex_vertices(x, y, 1.0, np.degrees(a))
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    # Build inner hexagons list
    inner = []
    for i in range(N):
        x, y, a = inner_params[3*i] - centroid[0], inner_params[3*i+1] - centroid[1], inner_params[3*i+2]
        inner.append([x, y, np.degrees(a)])
    
    # Compute outer side length
    shifted = all_verts - centroid
    max_val = 0.0
    for k in range(6):
        theta = np.radians(outer_angle) + np.pi/6 + k * np.pi / 3.0
        normal = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(shifted, normal)
        if np.max(proj) > max_val:
            max_val = np.max(proj)
    
    s = max_val / (np.sqrt(3)/2)
    
    return inner, (0.0, 0.0), s, np.degrees(outer_angle)

# EVOLVE-BLOCK-START
def construct():
    """Main entry point for packing 11 unit hexagons."""
    deadline = time.time() + 140.0
    
    def initial_params():
        params = []
        positions = [
            (0, 0),
            (2.2, 0), (1.1, 1.9), (-1.1, 1.9), (-2.2, 0), (-1.1, -1.9), (1.1, -1.9),
            (3.3, 1.9), (0, 3.8), (-3.3, 1.9), (3.3, -1.9)
        ]
        for x, y in positions:
            params.extend([x, y, 0.0])
        params.append(0.0)
        return params
    
    bounds = [(-10, 10), (-10, 10), (-np.pi, np.pi)] * N + [(-np.pi/6, np.pi/6)]
    
    best_result = None
    best_score = float('inf')
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=80,
            popsize=12,
            tol=1e-5,
            seed=42,
            polish=True,
            init='sobol'
        )
        score = result.fun
        if score < best_score:
            best_score = score
            best_result = result.x
    except Exception:
        pass
    
    if best_result is None:
        best_result = np.array(initial_params())
    
    inner_params = best_result[:3*N]
    outer_angle = best_result[-1]
    
    all_verts = []
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        v = hex_vertices(x, y, 1.0, np.degrees(a))
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    inner = []
    for i in range(N):
        x, y, a = inner_params[3*i] - centroid[0], inner_params[3*i+1] - centroid[1], inner_params[3*i+2]
        inner.append([x, y, np.degrees(a)])
    
    shifted = all_verts - centroid
    max_val = 0.0
    for k in range(6):
        theta = np.radians(outer_angle) + np.pi/6 + k * np.pi / 3.0
        normal = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(shifted, normal)
        if np.max(proj) > max_val:
            max_val = np.max(proj)
    
    s = max_val / (np.sqrt(3)/2)
    
    return inner, (0.0, 0.0), s, np.degrees(outer_angle)
# EVOLVE-BLOCK-END
