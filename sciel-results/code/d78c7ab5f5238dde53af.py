import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's local frame
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with flat-top orientation: vertices at angles 0,60,...
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The hexagon can be defined as |x| <= s*sqrt(3)/2 and |y| <= s*sqrt(3)/2 and |x+y| <= s*sqrt(3)/2 etc.
    # Actually for a regular hexagon with vertices at distance s from center:
    # The six constraints are: for k=0..5, dot(p, v_k) <= s where v_k are unit vectors to vertices
    # Wait, let me think more carefully.
    # A regular hexagon with side length s has vertices at distance s from center.
    # The apothem (distance from center to edge) is s * sqrt(3) / 2.
    # The six edges are perpendicular to the six vertex directions.
    # Edge k is at distance apothem from center, in direction of vertex k.
    # So constraint: dot(local_point, vertex_dir_k) <= apothem for all k.
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        theta = math.radians(60 * i)  # vertex directions in local frame (angle already accounted)
        vx = math.cos(theta)
        vy = math.sin(theta)
        if lx * vx + ly * vy > apothem + 1e-9:
            return False
    return True

def _hexagon_distance_to_center(cx, cy, side, angle_deg, outer_cx, outer_cy, outer_side, outer_angle):
    """
    Compute the minimum required outer side length such that this hexagon fits inside the outer hexagon.
    We need all vertices of inner hex to be inside outer hex.
    Return the minimal outer side (given center and rotation of outer) such that this inner hex fits.
    """
    # Get inner hex vertices
    verts = _hex_vertices(cx, cy, side, angle_deg)
    # For each vertex, compute the required outer side such that vertex is inside outer hex
    # Outer hex is centered at (outer_cx, outer_cy) with side R and rotation outer_angle
    # A point (px,py) is inside outer hex iff for all k=0..5:
    #   dot((px-outer_cx, py-outer_cy) rotated by -outer_angle, vertex_dir_k) <= R * sqrt(3)/2
    # So for each vertex and each direction k:
    #   dot(local_point, v_k) <= R * sqrt(3)/2
    # => R >= dot(local_point, v_k) / (sqrt(3)/2) = 2*dot(local_point, v_k)/sqrt(3)
    # Take max over all vertices and all k.
    
    oa = math.radians(outer_angle)
    ca, sa = math.cos(-oa), math.sin(-oa)
    max_r = 0.0
    for (px, py) in verts:
        dx = px - outer_cx
        dy = py - outer_cy
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        for i in range(6):
            theta = math.radians(60 * i)
            vx = math.cos(theta)
            vy = math.sin(theta)
            d = lx * vx + ly * vy
            if d > 0:
                r_needed = d / (math.sqrt(3) / 2)
                if r_needed > max_r:
                    max_r = r_needed
    return max_r

def _cluster_cost(params, outer_cx, outer_cy, outer_side, outer_angle):
    """
    Cost for a single cluster of 4 hexagons.
    params: [x0,y0,a0, x1,y1,a1, x2,y2,a2, x3,y3,a3] (12 values)
    Returns the max over all 4 hexagons of the required outer side.
    """
    max_r = 0.0
    for i in range(4):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        r = _hexagon_distance_to_center(x, y, 1.0, a, outer_cx, outer_cy, outer_side, outer_angle)
        if r > max_r:
            max_r = r
    return max_r

def _check_cluster_validity(params, outer_cx, outer_cy, outer_side, outer_angle):
    """Check that 4 hexagons in a cluster don't overlap each other."""
    # For each pair, check if they overlap
    # Two hexagons overlap if any vertex of one is inside the other, or vice versa
    # Approximate: check center-to-center distance vs sum of circumradii
    # Better: check if any vertex of hex A is inside hex B
    positions = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(4)]
    for i in range(4):
        for j in range(i+1, 4):
            xi, yi, ai = positions[i]
            xj, yj, aj = positions[j]
            # Check if any vertex of i is inside j
            verts_i = _hex_vertices(xi, yi, 1.0, ai)
            overlap = False
            for (vx, vy) in verts_i:
                if _point_in_hexagon(vx, vy, xj, yj, 1.0, aj):
                    overlap = True
                    break
            if not overlap:
                verts_j = _hex_vertices(xj, yj, 1.0, aj)
                for (vx, vy) in verts_j:
                    if _point_in_hexagon(vx, vy, xi, yi, 1.0, ai):
                        overlap = True
                        break
            if overlap:
                return False
    return True

def _optimize_cluster(seed_positions, outer_cx, outer_cy, outer_side, outer_angle, timeout=5.0):
    """
    Optimize a cluster of 4 hexagons to minimize the max required outer side.
    seed_positions: list of 4 (x, y, angle) tuples
    """
    start = time.time()
    
    # Initial parameters
    x0 = []
    for (x, y, a) in seed_positions:
        x0.extend([x, y, a])
    
    def objective(params):
        if time.time() - start > timeout:
            return 1e10
        # Check validity
        if not _check_cluster_validity(params, outer_cx, outer_cy, outer_side, outer_angle):
            return 1e8
        return _cluster_cost(params, outer_cx, outer_cy, outer_side, outer_angle)
    
    # Try local optimization from seed
    best_params = x0[:]
    best_cost = objective(x0)
    
    # Use Nelder-Mead for local optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
    except:
        pass
    
    return best_params, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy: 
    # 1. Create 3 clusters of 4 hexagons each
    # 2. Optimize each cluster internally
    # 3. Optimize the relative positions/rotations of the 3 clusters
    # 4. The outer hexagon is determined by the union of all 12 hexagons
    
    # Initial guess for outer hexagon: side ~ 5, centered at origin
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # We'll iterate: assume an outer hex, optimize clusters, then compute actual bounding hex
    
    best_result = None
    best_score = float('inf')
    
    # Try several initial configurations
    n_trials = 5
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Random initial positions for 3 clusters
        # Cluster centers roughly at 120-degree intervals
        cluster_angles = [math.radians(90 + 120*k + 30*trial) for k in range(3)]
        cluster_radius = 3.0 + 0.5*trial
        
        # Initialize each cluster with 4 hexagons in a small diamond pattern
        all_hexes = []
        for c in range(3):
            ca = cluster_angles[c]
            ccx = cluster_radius * math.cos(ca)
            ccy = cluster_radius * math.sin(ca)
            # 4 hexagons in a 2x2 diamond
            offsets = [(-0.9, 0, 0), (0.9, 0, 0), (0, -0.9, 0), (0, 0.9, 0)]
            for (ox, oy, oa) in offsets:
                hx = ccx + ox
                hy = ccy + oy
                ha = math.degrees(ca) + oa
                all_hexes.append((hx, hy, ha))
        
        # Optimize each cluster independently
        for c in range(3):
            if time.time() > deadline:
                break
            cluster_seed = all_hexes[c*4:(c+1)*4]
            # Assume outer hex side ~ 5 for now
            params, cost = _optimize_cluster(cluster_seed, outer_cx, outer_cy, 5.0, outer_angle, timeout=3.0)
            for i in range(4):
                all_hexes[c*4+i] = (params[3*i], params[3*i+1], params[3*i+2])
        
        # Now compute the actual bounding hexagon for all 12 hexagons
        # The outer hexagon must contain all 12. We need to find the minimal side.
        # The outer hexagon center and rotation can also be optimized.
        # For now, fix outer center at origin and angle at 0, compute required side.
        
        def bounding_side(hexes, ocx, ocy, oangle):
            max_r = 0.0
            for (hx, hy, ha) in hexes:
                r = _hexagon_distance_to_center(hx, hy, 1.0, ha, ocx, ocy, 1.0, oangle)
                if r > max_r:
                    max_r = r
            return max_r
        
        side = bounding_side(all_hexes, 0.0, 0.0, 0.0)
        
        if side < best_score:
            best_score = side
            best_result = (all_hexes[:], (0.0, 0.0), side, 0.0)
        
        # Try to optimize outer hexagon center and rotation
        if time.time() > deadline:
            break
        
        def outer_objective(params):
            if time.time() > deadline:
                return 1e10
            ocx, ocy, oangle = params
            return bounding_side(all_hexes, ocx, ocy, oangle)
        
        try:
            res = minimize(outer_objective, [0.0, 0.0, 0.0], method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = res.fun
                ocx, ocy, oangle = res.x
                best_result = (all_hexes[:], (ocx, ocy), best_score, oangle)
        except:
            pass
    
    if best_result is None:
        # Fallback: just place them in a row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

# EVOLVE-BLOCK-END
