"""Packing unit hexagons in a regular hexagon via smooth Nelder-Mead optimization."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle):
    """Vertices of regular hexagon centered at (cx,cy), circumradius r, rotation angle (rad)."""
    verts = np.zeros((6, 2))
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts[i, 0] = cx + r * math.cos(a)
        verts[i, 1] = cy + r * math.sin(a)
    return verts

def _support_function_diff(v1, v2, theta):
    """
    Compute the signed distance between two convex polygons along direction theta.
    Positive if separated, negative if overlapping.
    Uses support functions.
    """
    # v1, v2 are (6,2) arrays
    # Support function h_P(u) = max v . u
    u = np.array([math.cos(theta), math.sin(theta)])
    
    h1 = np.max(v1 @ u)
    h2 = np.max(v2 @ u)
    
    # Distance from P1 in direction u minus distance from P2 in direction -u
    # If h1(u) + h2(-u) > 0, they are separated along u?
    # Actually, separation distance along normal n is h_P1(n) - h_P2(-n)
    # Wait, if n points from P2 to P1, dist = h_P1(n) + h_P2(-n) ? No.
    # Let's use the standard: min distance = min over directions of (h1(n) + h2(-n)) ?
    # No, for two convex sets A, B, dist(A,B) = min_{x in A, y in B} ||x-y||.
    # This is equivalent to min_{||n||=1} (h_A(n) + h_B(-n))? No.
    # The gap along normal n is h_A(n) - h_B(n) if we align them?
    # Let's just compute the minimum distance between the two polygons directly using a robust method
    # or use the support function difference in the direction of centers.
    
    # For a smooth penalty, we can approximate the non-overlap constraint.
    # A common trick: penalty = max(0, overlap_depth)^2.
    # Overlap depth can be estimated by the negative of the minimum distance.
    
    # To be fast and smooth, we can use the distance between centers and the sum of circumradii?
    # No, that's too loose.
    
    # Let's use the minimum distance between edges/vertices.
    # This is not smooth at the boundary.
    
    # Alternative: Use the support function in the direction of the vector connecting centers.
    # This is a heuristic but differentiable almost everywhere.
    
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    d = c2 - c1
    dist_c = np.linalg.norm(d)
    if dist_c < 1e-9:
        n = np.array([1.0, 0.0])
    else:
        n = d / dist_c
        
    h1 = np.max(v1 @ n)
    h2 = np.max(v2 @ (-n)) # Support of v2 in direction -n
    
    # The "gap" in this direction is h1 + h2 - dist_c ?
    # Let's check: if v1 is at origin, v2 at d.
    # Max projection of v1 on n is h1. Min projection of v2 on n is h2 - dist_c (since h2 is max on -n, min on n is -h2 relative to center 2? No).
    # Let's just compute the actual minimum distance. It's safer.
    return _min_dist(v1, v2)

def _min_dist(v1, v2):
    """
    Compute minimum distance between two convex polygons.
    Returns 0 if they intersect.
    """
    # Check intersection first for speed (if they intersect, dist is 0)
    if _polygons_intersect(v1, v2):
        return 0.0
    
    min_d = float('inf')
    # Distance from each vertex of v1 to each edge of v2
    for pt in v1:
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            d = _point_segment_dist(pt, p1, p2)
            if d < min_d:
                min_d = d
    # Distance from each vertex of v2 to each edge of v1
    for pt in v2:
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            d = _point_segment_dist(pt, p1, p2)
            if d < min_d:
                min_d = d
    return min_d

def _point_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _polygons_intersect(v1, v2):
    """Check if two convex polygons intersect."""
    # SAT (Separating Axis Theorem)
    # Check normals of all edges
    for i in range(6):
        n1 = v1[(i+1)%6] - v1[i]
        normal = np.array([-n1[1], n1[0]])
        
        # Project v1 and v2 onto normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    # Check normals of v2
    for i in range(6):
        n2 = v2[(i+1)%6] - v2[i]
        normal = np.array([-n2[1], n2[0]])
        
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _smooth_overlap_penalty(v1, v2, smoothness=0.1):
    """
    Compute a smooth penalty for overlap.
    Uses the minimum distance. If d > 0, penalty is 0.
    If d < 0 (which shouldn't happen with _min_dist returning 0 on intersect), we need a different metric.
    
    Actually, _min_dist returns 0 if they intersect. We need a measure of "how much" they intersect.
    A better metric for smoothness:
    Penalty = max(0, (R1 + R2 - dist_centers)^2) is too loose.
    
    Let's use the support function gap in the direction of centers.
    If the gap is negative, they overlap.
    """
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    d = c2 - c1
    dist_c = np.linalg.norm(d)
    
    if dist_c < 1e-9:
        # Overlap heavily
        return 10.0
        
    n = d / dist_c
    
    # Support of v1 in direction n
    h1 = np.max(v1 @ n)
    # Support of v2 in direction -n
    h2 = np.max(v2 @ (-n))
    
    # The "gap" is h1 + h2 - dist_c ?
    # Let's verify.
    # If v1 is a point at 0, v2 is a point at d. h1=0, h2=0. gap = -dist_c.
    # Wait, support function h_P(n) = max_{x in P} x.n.
    # For v2 centered at c2, x = c2 + u. x.n = c2.n + u.n.
    # max (c2.n + u.n) = c2.n + max(u.n).
    # So h2(-n) = c2.(-n) + max(u.(-n)) = -c2.n + h_{local2}(-n).
    # h1(n) = c1.n + h_{local1}(n).
    # Gap = h1(n) + h2(-n) = c1.n + h1_loc + -c2.n + h2_loc = -(c2-c1).n + h1_loc + h2_loc = -dist_c + h1_loc + h2_loc.
    # This is the distance between the supporting planes.
    # If this is negative, the supporting planes have crossed, implying intersection (for convex sets).
    
    gap = h1 + h2 - dist_c
    
    if gap > 0:
        return 0.0
    else:
        # Smooth penalty for negative gap
        # Use a smooth approximation of max(0, -gap)^2
        # Or just -gap^2
        return -gap * gap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = float('inf')
    best_x = None
    
    # Objective function for Nelder-Mead
    # x = [x1, y1, a1, ..., xN, yN, aN, cx, cy, outer_angle]
    # Total size: 3*N + 3 = 36 for N=11
    
    def objective(x):
        # Extract positions, angles, center, outer angle
        positions = x[:3*N].reshape(N, 3)
        cx = x[3*N]
        cy = x[3*N+1]
        outer_angle = x[3*N+2]
        
        # Compute outer side length required to contain all hexagons
        # Outer hexagon center (cx, cy), side S, angle outer_angle.
        # Apothem A = S * sqrt(3)/2.
        # A point (px, py) is inside if for all 6 face normals n_k:
        # (px-cx, py-cy) . n_k <= A
        
        min_side = 1.0 # Minimum side is 1 (circumradius)
        
        # Precompute face normals for outer hexagon
        normals = []
        for k in range(6):
            a = outer_angle + k * math.pi / 3.0
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        for i in range(N):
            px, py, ai = positions[i]
            verts = _hex_vertices(px, py, 1.0, ai)
            
            for pt in verts:
                rel = np.array([pt[0] - cx, pt[1] - cy])
                for n in normals:
                    proj = np.dot(rel, n)
                    # proj <= A = S * sqrt(3)/2
                    # S >= proj / (sqrt(3)/2)
                    req_S = proj / (math.sqrt(3) / 2.0)
                    if req_S > min_side:
                        min_side = req_S
        
        # Compute overlap penalty
        penalty = 0.0
        for i in range(N):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
            for j in range(i+1, N):
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                penalty += _smooth_overlap_penalty(v1, v2)
        
        # Total score
        return min_side + penalty * 10.0
        
    def is_valid(x):
        positions = x[:3*N].reshape(N, 3)
        # Check overlaps strictly
        for i in range(N):
            v1 = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
            for j in range(i+1, N):
                v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                if _polygons_intersect(v1, v2):
                    return False
        return True
        
    # Initial guess: Hexagonal lattice
    def get_initial_x():
        positions = []
        # Simple grid
        dx = 2.0
        dy = 2.0 * math.sqrt(3) / 2.0
        n_per_row = 4
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            x = col * dx + (row % 2) * dx / 2.0
            y = row * dy
            positions.append([x, y, 0.0])
            
        # Center
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        for p in positions:
            p[0] -= cx
            p[1] -= cy
            
        x = np.array([p for p in positions for p in p] + [0.0, 0.0, 0.0])
        return x
        
    # Multi-start
    num_starts = 10
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        x0 = get_initial_x()
        # Add noise
        x0 = x0 + np.random.normal(0, 1.0, x0.shape)
        
        # Run Nelder-Mead
        # maxiter is important, but we have a time limit
        options = {
            'maxiter': 500,
            'xatol': 1e-4,
            'fatol': 1e-4
        }
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options, 
                           callback=lambda x: None) # No callback to save time, check time in loop? 
            # Nelder-Mead in scipy doesn't have a time limit option.
            # We must rely on maxiter being small enough or checking time externally.
            # Since we can't break out of minimize easily, we keep maxiter low.
            
            if res.success or res.fun < best_score:
                if is_valid(res.x):
                    # Calculate true side length
                    positions = res.x[:3*N].reshape(N, 3)
                    cx = res.x[3*N]
                    cy = res.x[3*N+1]
                    outer_angle = res.x[3*N+2]
                    
                    min_side = 1.0
                    normals = []
                    for k in range(6):
                        a = outer_angle + k * math.pi / 3.0
                        normals.append(np.array([math.cos(a), math.sin(a)]))
                    
                    for i in range(N):
                        px, py, ai = positions[i]
                        verts = _hex_vertices(px, py, 1.0, ai)
                        for pt in verts:
                            rel = np.array([pt[0] - cx, pt[1] - cy])
                            for n in normals:
                                proj = np.dot(rel, n)
                                req_S = proj / (math.sqrt(3) / 2.0)
                                if req_S > min_side:
                                    min_side = req_S
                                    
                    if min_side < best_score:
                        best_score = min_side
                        best_x = res.x.copy()
                        
        except Exception as e:
            pass
            
        if time.time() > deadline:
            break

    if best_x is not None:
        positions = best_x[:3*N].reshape(N, 3)
        cx = best_x[3*N]
        cy = best_x[3*N+1]
        outer_angle = best_x[3*N+2]
        
        inner = [[positions[i][0], positions[i][1], math.degrees(positions[i][2])] for i in range(N)]
        return inner, (cx, cy), best_score, math.degrees(outer_angle)
        
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
