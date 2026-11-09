import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(side, cx, cy, angle_deg):
    """Return the 6 vertices of a regular hexagon."""
    theta = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 + angle
    # Circumradius = side for a regular hexagon
    verts = []
    for k in range(6):
        a = theta + math.radians(60.0 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def hexagon_support(verts, direction):
    """Return the maximum dot product of vertices with direction vector."""
    return np.max(np.dot(verts, direction))

def check_validity(inner, center, side, angle_deg):
    """
    Check if the arrangement is valid:
    1. All inner hexagons are inside the outer hexagon
    2. Inner hexagons are pairwise disjoint
    Returns (is_valid, max_violation)
    """
    outer_verts = hex_vertices(side, center[0], center[1], angle_deg)
    inner_verts = []
    for (x, y, ang) in inner:
        inner_verts.append(hex_vertices(1.0, x, y, ang))
    
    max_violation = 0.0
    
    # Check each inner hexagon is inside outer hexagon
    # For convex polygons, check that each vertex of inner is inside outer
    # Using support function: for all directions, support(inner, d) <= support(outer, d)
    # But that's expensive. Instead, check vertices are inside.
    # A point is inside a convex polygon if it's to the left of all edges (for CCW)
    
    def point_in_convex_polygon(px, py, poly):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            # Cross product to check if point is to the left of edge
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-9:
                return False
        return True
    
    for iv in inner_verts:
        for (px, py) in iv:
            if not point_in_convex_polygon(px, py, outer_verts):
                max_violation = max(max_violation, 1e6)
                return False, max_violation
    
    # Check pairwise disjointness of inner hexagons
    # Two convex polygons are disjoint if their intersection is empty
    # Simple check: for each pair, check if any vertex of one is inside the other
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            vi = inner_verts[i]
            vj = inner_verts[j]
            # Check if any vertex of i is in j
            for (px, py) in vi:
                if point_in_convex_polygon(px, py, vj):
                    max_violation = max(max_violation, 1e6)
                    return False, max_violation
            # Check if any vertex of j is in i
            for (px, py) in vj:
                if point_in_convex_polygon(px, py, vi):
                    max_violation = max(max_violation, 1e6)
                    return False, max_violation
            # Also check edge intersections (simplified: check midpoint of edges)
            # For now, vertex containment is a good approximation for well-separated hexagons
    
    return True, 0.0

def compute_outer_hexagon(inner, center, angle_deg):
    """
    Given inner hexagons, center and rotation of outer hexagon,
    compute the minimum side length of outer hexagon that contains all inner hexagons.
    """
    outer_verts = None
    min_side = float('inf')
    
    # The outer hexagon has center `center` and rotation `angle_deg`
    # We need to find minimum `side` such that all inner hexagons are inside.
    
    # For a regular hexagon with center (cx, cy) and rotation angle, the 6 supporting lines
    # are at angles angle_deg + k*60, k=0..5
    # The distance from center to each edge is inradius = side * sqrt(3)/2
    
    # For each inner hexagon vertex, compute its distance from the outer hexagon edges
    # The required side is determined by the maximum distance to any edge
    
    # Outer hexagon edges are at angles: for k in 0..5, the edge normal is at angle (angle_deg + k*60)
    # The edge is at distance inradius from center in the direction of the normal
    
    # A point (px, py) is inside if for all k:
    # dot((px-cx, py-cy), n_k) <= inradius
    # where n_k = (cos(angle_deg + k*60), sin(angle_deg + k*60))
    
    cx, cy = center
    max_dot = 0.0
    
    for (x, y, ang) in inner:
        verts = hex_vertices(1.0, x, y, ang)
        for k in range(6):
            a = math.radians(angle_deg + 60.0 * k)
            nx = math.cos(a)
            ny = math.sin(a)
            for (px, py) in verts:
                d = (px - cx) * nx + (py - cy) * ny
                if d > max_dot:
                    max_dot = d
    
    # inradius = side * sqrt(3) / 2
    # So side = inradius * 2 / sqrt(3)
    inradius = max(max_dot, 0.0)
    side = inradius * 2.0 / math.sqrt(3.0)
    
    return side

def objective_and_constraints(x, inner_template, center_template, angle_template):
    """
    Objective: minimize outer hexagon side length
    Constraints: all inner hexagons must be pairwise disjoint
    """
    # x = [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    # where a is in radians
    
    inner = []
    for k in range(N):
        xk = x[3*k]
        yk = x[3*k+1]
        ak = math.degrees(x[3*k+2])
        inner.append([xk, yk, ak])
    
    side = compute_outer_hexagon(inner, center_template, angle_template)
    
    # Check disjointness
    verts = []
    for (xk, yk, ak) in inner:
        verts.append(hex_vertices(1.0, xk, yk, ak))
    
    for i in range(N):
        for j in range(i+1, N):
            vi = verts[i]
            vj = verts[j]
            # Check if any vertex of i is inside j
            for (px, py) in vi:
                if point_in_convex_polygon(px, py, vj):
                    return side, -1e6  # Violation
            for (px, py) in vj:
                if point_in_convex_polygon(px, py, vi):
                    return side, -1e6  # Violation
    
    return side, 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # 100 second budget
    
    # Initial good configuration: flower arrangement
    d = 2.0 * math.sqrt(3.0)  # distance between centers for touching hexagons
    
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(60.0 * k)
        positions.append((d * math.cos(theta), d * math.sin(theta)))
    for k in range(5):
        theta = math.radians(60.0 * k + 30.0)
        positions.append((2.0 * d * math.cos(theta), 2.0 * d * math.sin(theta)))
    
    inner0 = [[x, y, 0.0] for (x, y) in positions]
    center0 = (0.0, 0.0)
    angle0 = 0.0
    
    side0 = compute_outer_hexagon(inner0, center0, angle0)
    valid0, _ = check_validity(inner0, center0, side0, angle0)
    
    best_side = side0
    best_inner = inner0
    best_center = center0
    best_angle = angle0
    
    if not valid0:
        # Need to adjust
        # Scale positions slightly
        for scale in [1.01, 1.02, 1.05, 1.1]:
            positions_s = [(x*scale, y*scale) for (x, y) in positions]
            inner_s = [[x, y, 0.0] for (x, y) in positions_s]
            side_s = compute_outer_hexagon(inner_s, center0, angle0)
            valid_s, _ = check_validity(inner_s, center0, side_s, angle0)
            if valid_s:
                best_side = side_s
                best_inner = inner_s
                break
    
    # Try to optimize using scipy
    if time.time() < deadline:
        try:
            # Flatten initial configuration
            x0 = []
            for (xk, yk, ak) in best_inner:
                x0.append(xk)
                x0.append(yk)
                x0.append(math.radians(ak))
            x0 = np.array(x0)
            
            # Objective function for scipy
            def obj(x):
                inner_x = []
                for k in range(N):
                    inner_x.append([x[3*k], x[3*k+1], math.degrees(x[3*k+2])])
                return compute_outer_hexagon(inner_x, best_center, best_angle)
            
            # Penalty for overlap
            def penalty(x):
                inner_x = []
                for k in range(N):
                    inner_x.append([x[3*k], x[3*k+1], math.degrees(x[3*k+2])])
                verts = [hex_vertices(1.0, xk, yk, ak) for (xk, yk, ak) in inner_x]
                p = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        for (px, py) in verts[i]:
                            if point_in_convex_polygon(px, py, verts[j]):
                                p += 10.0
                        for (px, py) in verts[j]:
                            if point_in_convex_polygon(px, py, verts[i]):
                                p += 10.0
                return p
            
            def obj_penalized(x):
                return obj(x) + penalty(x)
            
            result = minimize(obj_penalized, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.success or result.fun < best_side + 1:
                x_opt = result.x
                inner_opt = []
                for k in range(N):
                    inner_opt.append([x_opt[3*k], x_opt[3*k+1], math.degrees(x_opt[3*k+2])])
                side_opt = compute_outer_hexagon(inner_opt, best_center, best_angle)
                valid_opt, _ = check_validity(inner_opt, best_center, side_opt, best_angle)
                
                if valid_opt and side_opt < best_side:
                    best_side = side_opt
                    best_inner = inner_opt
                    
        except Exception as e:
            pass
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
