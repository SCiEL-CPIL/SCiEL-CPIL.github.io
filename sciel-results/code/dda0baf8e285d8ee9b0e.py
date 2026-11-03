"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside the hexagon."""
        # Transform point to hexagon's local frame
        a = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        rx = dx * math.cos(-a) - dy * math.sin(-a)
        ry = dx * math.sin(-a) + dy * math.cos(-a)
        
        # For a regular hexagon with side s, the distance from center to each edge is s*sqrt(3)/2
        # The hexagon in local coords (flat-top) has vertices at angles 0, 60, 120, 180, 240, 300
        # Actually, let's use the fact that a regular hexagon can be described by 6 half-planes
        # In local coordinates, the edges are at distance h = side * sqrt(3)/2 from center
        # The normals to the edges are at angles 0, 60, 120, 180, 240, 300 (for flat-top)
        # Wait, for pointy-top (vertex at top), normals are at 30, 90, 150, 210, 270, 330
        
        # Let me use a simpler approach: check against all 6 edges
        verts = hex_vertices(cx, cy, side, angle_deg)
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            # Cross product to check if point is on the correct side of edge
            ex = x2 - x1
            ey = y2 - y1
            px1 = px - x1
            py1 = py - y1
            cross = ex * py1 - ey * px1
            # For a convex polygon, all cross products should have the same sign
            # We need to determine the sign convention
            # Let's check: for a CCW polygon, interior is to the left, so cross should be >= 0
            # But our vertices might be CW or CCW depending on angle
            # Let's just check if point is inside by checking all edges
            # Actually, let me use a different approach: the hexagon is the intersection of 6 half-planes
            # In local frame (after rotation by -angle), the hexagon centered at origin with side s
            # has the property that max(|x|/s, |y|/(s*sqrt(3)/2)) <= 1 ... no that's not right
            # 
            # Let me use the standard hexagon inequality. For a regular hexagon with "radius" r (center to vertex) = s,
            # in local coordinates with a vertex at angle 0:
            # The hexagon is: |x| <= s, |y| <= s*sqrt(3)/2, |x*sqrt(3)/2 + y/2| <= s*sqrt(3)/2 ... 
            # This is getting complicated. Let me just use the cross product method properly.
            pass
        
        # Simpler: use the fact that a regular hexagon with side s has inradius = s*sqrt(3)/2
        # and the hexagon is the set of points whose distance to each of the 6 edges is non-negative
        # Let me just check using the vertices and cross products
        verts = hex_vertices(cx, cy, side, angle_deg)
        # Determine orientation
        area = 0
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            area += x1 * y2 - x2 * y1
        area /= 2
        
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex = x2 - x1
            ey = y2 - y1
            px1 = px - x1
            py1 = py - y1
            cross = ex * py1 - ey * px1
            if area > 0 and cross < 0:
                return False
            if area < 0 and cross > 0:
                return False
        return True
    
    def hexagon_contains_hexagon(cx1, cy1, s1, a1, cx2, cy2, s2, a2):
        """Check if hexagon 2 is entirely inside hexagon 1."""
        verts2 = hex_vertices(cx2, cy2, s2, a2)
        for vx, vy in verts2:
            if not point_in_hexagon(vx, vy, cx1, cy1, s1, a1):
                return False
        return True
    
    def hexagons_overlap(cx1, cy1, a1, cx2, cy2, a2):
        """Check if two unit hexagons overlap (interior intersection)."""
        # Two convex polygons overlap if one vertex of each is inside the other,
        # or if any edges intersect. For simplicity, check if any vertex of one is inside the other.
        verts1 = hex_vertices(cx1, cy1, 1.0, a1)
        verts2 = hex_vertices(cx2, cy2, 1.0, a2)
        
        for vx, vy in verts1:
            if point_in_hexagon(vx, vy, cx2, cy2, 1.0, a2):
                return True
        for vx, vy in verts2:
            if point_in_hexagon(vx, vy, cx1, cy1, 1.0, a1):
                return True
        return False
    
    def compute_outer_side(config):
        """Given configuration of inner hexagons, compute minimum outer hexagon side length."""
        # config: flat array [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, ox, oy, oangle]
        # First 3*N: inner hexagon params
        # Last 3: outer hexagon center and angle
        
        inner_params = config[:3 * N]
        ox, oy, oangle = config[3 * N], config[3 * N + 1], config[3 * N + 2]
        
        # Check for overlaps between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2],
                                     inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]):
                    return 1e10  # Penalty for overlap
        
        # For each inner hexagon, find the maximum "radius" needed from outer center
        # The outer hexagon must contain all inner hexagons
        # For a given outer hexagon orientation, the constraint is that all vertices of inner hexagons
        # must be inside the outer hexagon
        
        # The minimum side length of outer hexagon that contains all inner hexagons
        # with center (ox, oy) and angle oangle
        # 
        # For a regular hexagon with side S and center (ox,oy) and angle oangle,
        # a point (px,py) is inside if and only if for all 6 edges, the point is on the interior side.
        # The distance from center to each edge is S*sqrt(3)/2.
        # 
        # In the local frame of the outer hexagon (rotated by -oangle), the hexagon is:
        # max over 6 directions of (projection onto normal) <= S*sqrt(3)/2
        # 
        # Actually, for a regular hexagon, the support function in direction theta is:
        # h(theta) = S * max(cos(theta - i*pi/3) for i=0..5) ... no
        # 
        # Let me think differently. The outer hexagon with side S, center (ox,oy), angle oangle
        # contains a point p if and only if:
        # For each of the 6 edge normals n_i (at angles oangle + i*pi/3 + pi/6 for flat-top, or similar):
        # dot(p - center, n_i) <= S * sqrt(3) / 2
        # 
        # Wait, the inradius is S*sqrt(3)/2, so the distance from center to each edge is S*sqrt(3)/2.
        # The edge normals are at angles oangle + pi/6 + i*pi/3 for i=0..5 (for a hexagon with a flat edge at top)
        # or oangle + i*pi/3 (for a hexagon with a vertex at top)
        # 
        # Let's define: the outer hexagon has vertices at angles oangle + i*pi/3 for i=0..5
        # Then the edge normals are at angles oangle + pi/6 + i*pi/3
        # The constraint is: dot(p - (ox,oy), n_i) <= S*sqrt(3)/2 for all i
        
        a_outer = math.radians(oangle)
        max_required = 0.0
        for i in range(N):
            cx = inner_params[3*i]
            cy = inner_params[3*i+1]
            a_inner = inner_params[3*i+2]
            verts = hex_vertices(cx, cy, 1.0, a_inner)
            for vx, vy in verts:
                dx = vx - ox
                dy = vy - oy
                for j in range(6):
                    n_angle = a_outer + math.pi/6 + j * math.pi / 3
                    nx = math.cos(n_angle)
                    ny = math.sin(n_angle)
                    dist = dx * nx + dy * ny
                    # Required S such that dist <= S*sqrt(3)/2
                    # S >= 2*dist/sqrt(3)
                    required_S = 2.0 * dist / math.sqrt(3.0)
                    if required_S > max_required:
                        max_required = required_S
        
        # Also need to consider that the hexagon side must be positive
        if max_required < 0:
            max_required = 0.0
            
        return max_required
    
    def objective(config):
        """Objective function for optimization."""
        if time.time() > deadline:
            return 1e10
        return compute_outer_side(config)
    
    # Initial guess: place hexagons in a hexagonal pattern
    # Hexagonal packing: centers at distance 2*sqrt(3) apart for non-overlapping unit hexagons
    # Actually, two unit hexagons (side 1) are non-overlapping if their centers are at distance >= 2*sqrt(3)
    # Wait, the distance between centers of two non-overlapping unit hexagons...
    # The "radius" of a unit hexagon (center to vertex) is 1, and the inradius is sqrt(3)/2
    # Two hexagons don't overlap if the distance between centers is at least 2*sqrt(3) (if aligned)
    # More precisely, the minimum distance between centers of two non-overlapping unit hexagons
    # depends on their relative rotation. For the same rotation, it's 2*sqrt(3).
    
    d = 2.0 * math.sqrt(3.0) + 0.1  # Small margin
    
    # Hexagonal lattice positions
    positions = []
    # Center at origin
    positions.append((0.0, 0.0))
    # 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        positions.append((d * math.cos(angle), d * math.sin(angle)))
    # 12 next-nearest neighbors (but we only need 12 total, so 1 + 6 + 5 = 12)
    for i in range(6):
        angle = i * math.pi / 3 + math.pi / 6
        positions.append((d * math.sqrt(3) * math.cos(angle), d * math.sqrt(3) * math.sin(angle)))
        if len(positions) >= 12:
            break
    
    # If we have more than 12, trim
    positions = positions[:12]
    
    # Build initial config
    initial_config = []
    for px, py in positions:
        initial_config.append(px)
        initial_config.append(py)
        initial_config.append(0.0)  # angle
    initial_config.append(0.0)  # outer center x
    initial_config.append(0.0)  # outer center y
    initial_config.append(0.0)  # outer angle
    
    # Bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 60.0))    # angle (hexagon has 60-degree symmetry)
    bounds.append((-10.0, 10.0))  # outer x
    bounds.append((-10.0, 10.0))  # outer y
    bounds.append((0.0, 60.0))    # outer angle
    
    best_result = None
    best_score = 1e10
    
    # Try differential evolution with limited time
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=10,
            tol=0.01,
            seed=42,
            polish=True,
            update_interval=1
        )
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        # Return best found so far
        if best_result is not None:
            inner = []
            for i in range(N):
                inner.append([best_result[3*i], best_result[3*i+1], best_result[3*i+2]])
            return inner, (best_result[3*N], best_result[3*N+1]), best_score, best_result[3*N+2]
        else:
            # Fallback to initial configuration
            inner = []
            for px, py in positions:
                inner.append([px, py, 0.0])
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Local refinement with Nelder-Mead
    if best_result is not None:
        try:
            result2 = minimize(
                objective, best_result,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_result = result2.x
        except Exception:
            pass
    
    inner = []
    for i in range(N):
        inner.append([best_result[3*i], best_result[3*i+1], best_result[3*i+2]])
    return inner, (best_result[3*N], best_result[3*N+1]), best_score, best_result[3*N+2]
# EVOLVE-BLOCK-END
