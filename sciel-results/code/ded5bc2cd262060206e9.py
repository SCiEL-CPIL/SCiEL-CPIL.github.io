import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = np.zeros((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k, 0] = cx + side * np.cos(a)
        verts[k, 1] = cy + side * np.sin(a)
    return verts

def get_hex_edge_normals(verts):
    """Return outward normals and edge midpoints for a hexagon."""
    normals = np.zeros((6, 2))
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        edge = b - a
        # Normal is perpendicular to edge, pointing outwards
        # For CCW vertices, right normal is inward, left is outward?
        # Let's check: edge (1,0) to (0,1). Edge vector (-1, 1).
        # Left normal: (-1, -1). Right normal: (1, 1).
        # Center is (0,0). Midpoint (0.5, 0.5).
        # Dot product with Left normal: -1 - 1 = -2.
        # Dot product with Right normal: 1 + 1 = 2.
        # So Right Normal points away from origin if center is origin.
        # Let's compute normal and check direction relative to center.
        edge_vec = b - a
        normal = np.array([-edge_vec[1], edge_vec[0]])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm
        # Check if it points away from center
        mid = (a + b) / 2.0
        # Assume center is roughly (0,0) for local check? No, we have cx, cy.
        # Better: just return the normal, and handle sign in projection.
        normals[i] = normal
    return normals

def compute_gaps(verts1, verts2, normals1, normals2):
    """
    Compute the gap in each of the 12 normal directions.
    Returns array of 12 gaps.
    Gap > 0 means separated in that direction.
    Gap < 0 means overlapping in that direction (projection intervals overlap).
    """
    gaps = []
    
    # Project verts2 onto normals1
    for i in range(6):
        n = normals1[i]
        proj2 = np.dot(verts2, n)
        min2, max2 = proj2.min(), proj2.max()
        
        # Project edge i of verts1 onto n
        # The projection of the polygon verts1 onto n is [min1, max1]
        # But we want the gap between the *edges*?
        # No, for SAT, we project the *entire* polygons.
        # If the intervals [min1, max1] and [min2, max2] do not overlap, gap is positive.
        # If they overlap, gap is 0 (for SAT separation test).
        # But for penetration depth, we need the *amount* of overlap.
        # The gap in direction n is:
        # If max1 < min2: gap = min2 - max1
        # If max2 < min1: gap = min1 - max2
        # Else: gap = 0 (they overlap in this direction)
        
        proj1 = np.dot(verts1, n)
        min1, max1 = proj1.min(), proj1.max()
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        gaps.append(gap)
        
    # Project verts1 onto normals2
    for i in range(6):
        n = normals2[i]
        proj1 = np.dot(verts1, n)
        min1, max1 = proj1.min(), proj1.max()
        
        proj2 = np.dot(verts2, n)
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        gaps.append(gap)
    
    return np.array(gaps)

def smooth_penetration_depth(gaps, beta=10.0):
    """
    Compute a smooth approximation of the penetration depth.
    If any gap > 0, depth is 0.
    If all gaps <= 0, depth is -min(gaps).
    We use soft-min to make it differentiable.
    soft_min(gaps) = -1/beta * log(sum(exp(-beta * gaps)))
    If gaps are all negative, exp(-beta * gap) is large for the most negative gap.
    So soft_min will be close to min(gaps).
    Depth = -soft_min(gaps).
    If there is a positive gap, exp(-beta * pos_gap) is small, but the sum is dominated by the negative ones?
    Actually, if there is a separating axis, the true depth is 0.
    The soft-min of a set containing a positive number and negative numbers will be negative.
    So we need to clamp: depth = max(0, -soft_min(gaps)).
    """
    # To avoid overflow, shift gaps
    max_gap = np.max(gaps)
    if max_gap > 0:
        # There is a separating axis, but soft-min might still be negative if other gaps are very negative.
        # We want depth to be 0 if separated.
        # A better approach:
        # If max_gap > 0, return 0.
        # Else, return -soft_min(gaps).
        # But this is not differentiable at max_gap = 0.
        # Let's use a smooth max.
        pass
    
    # Use log-sum-exp trick
    # We want min(gaps).
    # min(g) = -max(-g).
    # max(-g) = (1/beta) log(sum(exp(beta * (-g)))) = -(1/beta) log(sum(exp(-beta * g)))
    # So min(g) = -(1/beta) log(sum(exp(-beta * g)))
    
    # To ensure stability:
    g_max = np.max(gaps)
    if g_max > 1e-6:
        # Separated
        return 0.0
    
    # If not clearly separated, compute soft min
    # Shift by max to avoid overflow
    shifted = gaps - g_max
    sum_exp = np.sum(np.exp(-beta * shifted))
    if sum_exp < 1e-300:
        return 0.0
    soft_min = g_max - (1.0 / beta) * np.log(sum_exp)
    
    depth = -soft_min
    if depth < 0:
        depth = 0.0
    return depth

def penalty_overlap(h1, h2, beta=10.0):
    """
    Compute smooth overlap penalty between two hexagons.
    h1, h2: lists of (cx, cy, angle)
    """
    v1 = get_hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = get_hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    n1 = get_hex_edge_normals(v1)
    n2 = get_hex_edge_normals(v2)
    
    gaps = compute_gaps(v1, v2, n1, n2)
    depth = smooth_penetration_depth(gaps, beta)
    return depth * depth

def penalty_boundary(h, outer_center, outer_side, outer_angle, beta=10.0):
    """
    Compute penalty if hexagon h is outside the outer hexagon.
    """
    v_inner = get_hex_vertices(h[0], h[1], 1.0, h[2])
    v_outer = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    n_outer = get_hex_edge_normals(v_outer)
    
    # For each normal of the outer hexagon, check if inner hex is inside.
    # If inner hex projects beyond the outer edge, it's outside.
    max_violation = 0.0
    for i in range(6):
        n = n_outer[i]
        # Project outer vertex i onto n to get the limit
        # Actually, the support function of the outer hex in direction n is the max projection.
        # But the edge is at a specific distance.
        # Let's use the same gap logic.
        # Gap between inner and outer in direction n.
        # We want inner to be inside outer.
        # So min_inner_proj >= min_outer_proj and max_inner_proj <= max_outer_proj.
        
        proj_inner = np.dot(v_inner, n)
        min_in, max_in = proj_inner.min(), proj_inner.max()
        
        proj_outer = np.dot(v_outer, n)
        min_out, max_out = proj_outer.min(), proj_outer.max()
        
        # Violation if max_in > max_out or min_in < min_out
        viol1 = max_in - max_out
        viol2 = min_out - min_in
        
        max_violation = max(max_violation, viol1, viol2)
    
    if max_violation > 0:
        return max_violation * max_violation
    else:
        return 0.0

def objective(x):
    """
    Objective function for optimization.
    x: [x1, y1, th1, ..., x12, y12, th12, X, Y, Phi, L]
    """
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        th = x[i*3+2]
        inner.append((cx, cy, th))
    
    X = x[36]
    Y = x[37]
    Phi = x[38]
    L = x[39]
    
    if L < 3.0:
        return 1000.0 + (3.0 - L)**2
    
    outer_center = (X, Y)
    outer_angle = Phi
    
    # Boundary penalty
    bound_pen = 0.0
    for i in range(N):
        bound_pen += penalty_boundary(inner[i], outer_center, L, outer_angle)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            overlap_pen += penalty_overlap(inner[i], inner[j])
    
    # We want to minimize L, but keep penalties low.
    # Use large weights for penalties to enforce constraints.
    weight_bound = 100.0
    weight_overlap = 100.0
    
    return L + weight_bound * bound_pen + weight_overlap * overlap_pen

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: Hexagonal lattice
    # dx = 3.0, dy = sqrt(3)
    dx = 3.0
    dy = np.sqrt(3)
    
    x0 = np.zeros(40)
    idx = 0
    for row in range(3):
        for col in range(4):
            cx = col * dx + (row % 2) * dx * 0.5
            cy = row * dy
            x0[idx] = cx
            x0[idx+1] = cy
            x0[idx+2] = 0.0
            idx += 3
    
    # Outer hexagon initial guess
    # Center of the lattice
    x0[36] = 3.0 # Approx center x
    x0[37] = 1.5 * np.sqrt(3) # Approx center y
    x0[38] = 0.0 # Rotation
    x0[39] = 5.0 # Side length L
    
    # Optimization
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((-180, 180)) # angle
    bounds.append((-10, 10)) # X
    bounds.append((-10, 10)) # Y
    bounds.append((-180, 180)) # Phi
    bounds.append((3.0, 10.0)) # L
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_x = res.x
    best_val = res.fun
    
    # Refine with a second pass if time permits
    if time.time() - START_TIME < TIME_LIMIT - 30:
        # Increase penalty weights for a tighter solution
        global weight_bound, weight_overlap
        weight_bound = 500.0
        weight_overlap = 500.0
        
        def objective_tight(x):
            inner = []
            for i in range(N):
                cx = x[i*3]
                cy = x[i*3+1]
                th = x[i*3+2]
                inner.append((cx, cy, th))
            X = x[36]
            Y = x[37]
            Phi = x[38]
            L = x[39]
            if L < 3.0:
                return 1000.0 + (3.0 - L)**2
            bound_pen = 0.0
            for i in range(N):
                bound_pen += penalty_boundary(inner[i], (X, Y), L, Phi)
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    overlap_pen += penalty_overlap(inner[i], inner[j])
            return L + weight_bound * bound_pen + weight_overlap * overlap_pen
        
        res2 = minimize(objective_tight, best_x, method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if res2.fun < best_val:
            best_x = res2.x
            best_val = res2.fun

    # Extract result
    inner = []
    for i in range(N):
        inner.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
    
    X = best_x[36]
    Y = best_x[37]
    Phi = best_x[38]
    L = best_x[39]
    
    # Final validity check and polish
    # If invalid, we might need to adjust L slightly up or reject.
    # For now, return the best found.
    
    return inner, (X, Y), L, Phi

# Global variables for penalty weights
weight_bound = 100.0
weight_overlap = 100.0
