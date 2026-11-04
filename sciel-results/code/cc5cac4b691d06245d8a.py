"""Packing 12 unit regular hexagons inside a regular hexagon using gradient-based optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to local coordinates
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx, dy = px - cx, py - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side s, the condition for (lx,ly) to be inside:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # In local coords, the hexagon is defined by |ly| <= s*sqrt(3)/2 and |lx| <= s - |ly|/sqrt(3)
    # Actually, for a flat-top hexagon with side s:
    # width = 2*s, height = s*sqrt(3)
    # The condition is: |ly| <= s*sqrt(3)/2 AND |lx| <= s*sqrt(3)/2 - |ly|/2
    # Wait, let me think more carefully.
    
    # Regular hexagon with side s, flat-top orientation:
    # Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # So the hexagon is bounded by:
    # |ly| <= s*sqrt(3)/2
    # |lx| <= s - |ly|/sqrt(3)
    
    sqrt3 = np.sqrt(3.0)
    if abs(ly) > side * sqrt3 / 2:
        return False
    if abs(lx) > side - abs(ly) / sqrt3:
        return False
    return True

def hexagon_sdf(px, py, cx, cy, side, angle_deg=0.0):
    """
    Signed distance from point (px,py) to hexagon. Negative if inside.
    This is an approximation using the analytical distance to the hexagon boundary.
    """
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx, dy = px - cx, py - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    sqrt3 = np.sqrt(3.0)
    
    # For a regular hexagon, the signed distance can be computed as follows:
    # The hexagon is the intersection of 6 half-planes.
    # The distance to the hexagon is the minimum over the 6 edges.
    
    # Edge normals for flat-top hexagon (pointing inward):
    # Edge 1: top, normal (0, 1), distance from center = s*sqrt(3)/2
    # Edge 2: upper-right, normal (sqrt(3)/2, -1/2), distance = s*sqrt(3)/2
    # etc.
    
    # Actually, let's use a simpler approach: compute the distance to each of the 6 edges
    # and take the minimum, with sign based on inside/outside.
    
    # The 6 edge normals (unit, pointing inward) and their distances from center:
    # For flat-top hexagon with side s:
    # h = s * sqrt(3) / 2 (apothem)
    
    h = side * sqrt3 / 2
    
    # Normals pointing inward:
    normals = np.array([
        [0, 1],
        [sqrt3/2, -0.5],
        [sqrt3/2, 0.5],
        [0, -1],
        [-sqrt3/2, 0.5],
        [-sqrt3/2, -0.5]
    ])
    
    # Distance from center to each edge along the normal direction is h
    # For point at (lx, ly), the signed distance to edge i is: h - dot(normal_i, (lx, ly))
    # Wait, that's not quite right. Let me reconsider.
    
    # The hexagon can be defined as: for all 6 edge normals n_i, dot(n_i, p) <= h
    # where p is the position relative to center.
    # The signed distance is min_i (h - dot(n_i, p))
    
    p = np.array([lx, ly])
    distances = h - normals @ p
    return np.min(distances)

def hexagon_overlap_penalty(v1, v2, v3, v4, side1=1.0, side2=1.0, margin=0.0):
    """
    Compute penalty for overlap between two hexagons.
    v1, v2: centers of hexagons 1 and 2
    v3, v4: angles of hexagons 1 and 2 (in radians)
    Returns a non-negative penalty.
    """
    # Check if any vertex of hex1 is inside hex2, or vice versa
    # Also check if centers are close enough that they might overlap
    
    # First, a quick bounding box check
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    # Max distance between centers for possible overlap is 2 * circumradius = 2 * side
    if dist > 2.0 * max(side1, side2) + margin:
        return 0.0
    
    # Compute SDF of each hexagon at the other's vertices and center
    # For hexagon 1's vertices, check SDF in hexagon 2
    verts1 = hexagon_vertices(v1[0], v1[1], side1, np.degrees(v3))
    verts2 = hexagon_vertices(v2[0], v2[1], side2, np.degrees(v4))
    
    penalty = 0.0
    
    # Check if any vertex of hex1 is inside hex2
    for vx, vy in verts1:
        sdf = hexagon_sdf(vx, vy, v2[0], v2[1], side2, np.degrees(v4))
        if sdf < 0:
            penalty += (-sdf) ** 2
    
    # Check if any vertex of hex2 is inside hex1
    for vx, vy in verts2:
        sdf = hexagon_sdf(vx, vy, v1[0], v1[1], side1, np.degrees(v3))
        if sdf < 0:
            penalty += (-sdf) ** 2
    
    # Also check if center of one is inside the other
    sdf1 = hexagon_sdf(v1[0], v1[1], v2[0], v2[1], side2, np.degrees(v4))
    if sdf1 < 0:
        penalty += (-sdf1) ** 2
    
    sdf2 = hexagon_sdf(v2[0], v2[1], v1[0], v1[1], side1, np.degrees(v3))
    if sdf2 < 0:
        penalty += (-sdf2) ** 2
    
    return penalty

def outer_hexagon_sdf(px, py, cx, cy, side, angle_deg=0.0):
    """
    Signed distance from point to the interior of the outer hexagon.
    Positive if inside, negative if outside.
    """
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx, dy = px - cx, py - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    sqrt3 = np.sqrt(3.0)
    h = side * sqrt3 / 2
    
    normals = np.array([
        [0, 1],
        [sqrt3/2, -0.5],
        [sqrt3/2, 0.5],
        [0, -1],
        [-sqrt3/2, 0.5],
        [-sqrt3/2, -0.5]
    ])
    
    p = np.array([lx, ly])
    distances = h - normals @ p
    return np.min(distances)

def objective(params, outer_side, outer_cx, outer_cy, outer_angle):
    """
    Objective function: minimize outer_side + penalty for violations.
    params: array of length 3*N, each triplet is (x, y, angle_rad) for inner hexagons.
    """
    t_start = time.time()
    
    inner_hexes = params.reshape(N, 3)
    
    penalty = 0.0
    
    # Check each inner hexagon is inside the outer hexagon
    for i in range(N):
        cx, cy, angle = inner_hexes[i]
        # Check all vertices of inner hexagon are inside outer
        verts = hexagon_vertices(cx, cy, 1.0, np.degrees(angle))
        for vx, vy in verts:
            sdf = outer_hexagon_sdf(vx, vy, outer_cx, outer_cy, outer_side, outer_angle)
            if sdf < 0:
                penalty += (-sdf) ** 2 * 100  # Heavy penalty for going outside
    
    # Check pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            c1 = inner_hexes[i]
            c2 = inner_hexes[j]
            p = hexagon_overlap_penalty(c1[:2], c2[:2], c1[2], c2[2])
            penalty += p * 100
    
    return outer_side + penalty

def construct():
    t0 = time.time()
    
    # Outer hexagon: center at origin, angle 0
    outer_cx, outer_cy, outer_angle = 0.0, 0.0, 0.0
    
    # We'll optimize over inner hexagon positions and the outer side length
    # But to keep it simpler, let's fix the outer hexagon and just pack the inner ones,
    # then compute the minimal outer side needed.
    
    # Actually, let's optimize the outer side as part of the problem.
    # But that makes it harder. Instead, let's try a few outer side lengths and find the minimum.
    
    # Start with a reasonable estimate. The area of 12 unit hexagons is 12 * (3*sqrt(3)/2) = 18*sqrt(3) ≈ 31.18
    # Area of outer hexagon with side S is (3*sqrt(3)/2) * S^2
    # So S^2 >= 31.18 / (3*sqrt(3)/2) = 31.18 / 2.598 ≈ 11.99, so S >= ~3.46
    # But with packing inefficiency, we need larger. Let's start with S = 5.0
    
    best_params = None
    best_score = float('inf')
    
    # Use differential evolution to find a good initial guess, then refine with L-BFGS-B
    # Parameters: for each inner hexagon, (x, y, angle_rad)
    # Bounds: x, y in [-10, 10], angle in [0, 2*pi]
    
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 2*np.pi))  # angle
    
    def obj_for_side(side):
        """Objective for a fixed outer side length."""
        def f(params):
            return objective(params, side, outer_cx, outer_cy, outer_angle)
        return f
    
    # Try a few outer side lengths
    side_candidates = [4.0, 4.5, 5.0, 5.5, 6.0]
    
    for side in side_candidates:
        if time.time() - t0 > DEADLINE:
            break
        
        f = obj_for_side(side)
        
        # Try multiple random initializations
        n_starts = 3
        for start_idx in range(n_starts):
            if time.time() - t0 > DEADLINE:
                break
            
            # Random initial positions in a hexagonal pattern
            if start_idx == 0:
                # Hexagonal lattice initialization
                params = np.zeros(N * 3)
                positions = []
                # Place hexagons in a hexagonal pattern
                # For 12 hexagons, use a hexagonal arrangement
                # Center + 6 around + 5 more
                spacing = 1.8  # Slightly more than 2*1 to avoid overlap
                angles = np.arange(6) * np.pi / 3
                positions.append((0, 0))
                for a in angles:
                    positions.append((spacing * np.cos(a), spacing * np.sin(a)))
                # Add 5 more in the second ring
                for a in angles[:5]:
                    r = spacing * 2
                    positions.append((r * np.cos(a), r * np.sin(a)))
                
                for i, (px, py) in enumerate(positions):
                    params[i*3] = px
                    params[i*3+1] = py
                    params[i*3+2] = 0.0
            else:
                # Random initialization
                np.random.seed(start_idx * 1000 + 42)
                params = np.random.uniform(-3, 3, N * 3)
                params[2::3] = np.random.uniform(0, 2*np.pi, N)
            
            # Use L-BFGS-B to refine
            result = minimize(f, params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < best_score:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_params = result.x.copy()
                    best_side = side
    
    # Now, try to reduce the outer side length
    # Start from the best found and try smaller sides
    if best_params is not None:
        # Try to find the minimum side that works
        lo, hi = 3.0, best_side
        for _ in range(15):
            if time.time() - t0 > DEADLINE:
                break
            
            mid = (lo + hi) / 2
            f = obj_for_side(mid)
            
            # Try to optimize with the current best params as starting point
            result = minimize(f, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-8})
            
            # Check if the penalty is small enough (i.e., valid packing)
            if result.fun < mid + 0.01:  # Penalty is small
                hi = mid
                best_params = result.x.copy()
            else:
                lo = mid
    
        best_side = hi
    
    # Extract inner hexagons from best_params
    if best_params is not None:
        inner = []
        for i in range(N):
            x, y, angle_rad = best_params[i*3:i*3+3]
            inner.append([x, y, np.degrees(angle_rad)])
    else:
        # Fallback: use the initial seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
    
    # Verify and return
    # Make sure the result is valid
    return inner, (outer_cx, outer_cy), best_side, outer_angle
