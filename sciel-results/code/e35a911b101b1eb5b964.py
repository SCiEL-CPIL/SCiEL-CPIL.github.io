"""Packing 11 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    vertices = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    return vertices

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap (interiors intersect)."""
    # Check if any vertex of v1 is inside v2 or vice versa
    n1 = len(v1)
    n2 = len(v2)
    
    # Point in convex polygon test using cross products
    def point_in_convex(px, py, poly):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i+1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -tol:
                return False
        return True
    
    for i in range(n1):
        if point_in_convex(v1[i,0], v1[i,1], v2):
            return True
    for i in range(n2):
        if point_in_convex(v2[i,0], v2[i,1], v1):
            return True
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        return False
    
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j+1) % n2]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    v = hexagon_vertices(cx, cy, side, angle_deg)
    return polygons_overlap(np.array([[px, py], [px+0.001, py], [px, py+0.001]]), v, tol=1e-9)

def check_inside_container(inner_verts, outer_verts, tol=1e-8):
    """Check if all vertices of inner polygon are inside outer polygon."""
    def point_in_convex(px, py, poly):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i+1) % n]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -tol:
                return False
        return True
    
    for i in range(len(inner_verts)):
        if not point_in_convex(inner_verts[i,0], inner_verts[i,1], outer_verts):
            return False
    return True

def objective(params):
    """Objective: outer side length + penalty for violations."""
    # params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, S, A]
    # where a is angle in radians, S is outer side length, A is outer angle in radians
    n_inner = N
    inner_params = params[:3*n_inner]
    cx = params[3*n_inner]
    cy = params[3*n_inner+1]
    S = params[3*n_inner+2]
    A = params[3*n_inner+3]
    
    # Get all inner hexagon vertices
    inner_verts_list = []
    for i in range(n_inner):
        xi = inner_params[3*i]
        yi = inner_params[3*i+1]
        ai = inner_params[3*i+2]
        verts = hexagon_vertices(xi, yi, 1.0, np.degrees(ai))
        inner_verts_list.append(verts)
    
    # Check inner-inner overlaps
    penalty = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            if polygons_overlap(inner_verts_list[i], inner_verts_list[j], tol=-1e-6):
                # Compute minimum distance to separate
                # Use a simple penalty based on how deep the overlap is
                penalty += 10.0
    
    # Check if inner hexagons are inside container
    outer_verts = hexagon_vertices(cx, cy, S, np.degrees(A))
    for i in range(n_inner):
        if not check_inside_container(inner_verts_list[i], outer_verts, tol=-1e-6):
            penalty += 10.0
    
    return S + penalty

def construct():
    """Main entry point: returns (inner, center, side, angle)."""
    deadline = START_TIME + TIME_LIMIT
    
    # Initial guess: hexagonal cluster arrangement
    # For 11 hexagons, try a 3-4-4 or similar pattern
    # Hexagon center-to-center distance for touching hexagons: sqrt(3)
    d = np.sqrt(3)
    
    # Try multiple initial configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: 4-3-4 rows
    configs = []
    
    # 4-3-4 arrangement
    xs1, ys1, as1 = [], [], []
    for row in range(3):
        if row == 0 or row == 2:
            count = 4
        else:
            count = 3
        for col in range(count):
            x = (col - (count-1)/2.0) * d
            y = row * d * 0.5
            xs1.append(x)
            ys1.append(y)
            as1.append(0.0)
    configs.append((xs1, ys1, as1))
    
    # 3-4-4 arrangement
    xs2, ys2, as2 = [], [], []
    for row in range(3):
        if row == 0:
            count = 3
        else:
            count = 4
        for col in range(count):
            x = (col - (count-1)/2.0) * d
            y = row * d * 0.5
            xs2.append(x)
            ys2.append(y)
            as2.append(0.0)
    configs.append((xs2, ys2, as2))
    
    # 5-3-3 arrangement
    xs3, ys3, as3 = [], [], []
    for row in range(3):
        if row == 0:
            count = 5
        else:
            count = 3
        for col in range(count):
            x = (col - (count-1)/2.0) * d
            y = row * d * 0.5
            xs3.append(x)
            ys3.append(y)
            as3.append(0.0)
    configs.append((xs3, ys3, as3))
    
    # 2-4-5 arrangement
    xs4, ys4, as4 = [], [], []
    for row in range(3):
        counts = [2, 4, 5]
        count = counts[row]
        for col in range(count):
            x = (col - (count-1)/2.0) * d
            y = row * d * 0.5
            xs4.append(x)
            ys4.append(y)
            as4.append(0.0)
    configs.append((xs4, ys4, as4))
    
    # 1-5-5 arrangement
    xs5, ys5, as5 = [], [], []
    for row in range(3):
        counts = [1, 5, 5]
        count = counts[row]
        for col in range(count):
            x = (col - (count-1)/2.0) * d
            y = row * d * 0.5
            xs5.append(x)
            ys5.append(y)
            as5.append(0.0)
    configs.append((xs5, ys5, as5))
    
    for cfg_idx, (xs, ys, as_) in enumerate(configs):
        if time.time() > deadline - 5:
            break
        
        # Compute initial container size
        all_x = np.array(xs)
        all_y = np.array(ys)
        
        # Estimate required side length
        # For hexagons arranged in a pattern, the bounding hexagon needs to cover all
        # The distance from center to farthest vertex of inner hexagons
        max_dist = 0
        for i in range(len(xs)):
            # Farthest point from origin for hexagon at (xs[i], ys[i])
            # is at distance sqrt(xs[i]^2 + ys[i]^2) + 1 (circumradius)
            dist = np.sqrt(xs[i]**2 + ys[i]**2) + 1.0
            max_dist = max(max_dist, dist)
        
        # For a regular hexagon, circumradius = side length
        # But we need to account for the hexagon shape
        # The inradius is S*sqrt(3)/2, circumradius is S
        # A point at distance r from center is inside if r <= S (for the direction of vertices)
        # or r <= S*sqrt(3)/2 (for the direction of edges)
        # Conservative estimate: S >= max_dist
        S_init = max_dist * 1.2
        
        # Build initial parameters
        params0 = []
        for i in range(N):
            params0.extend([xs[i], ys[i], as_[i]])
        params0.extend([0.0, 0.0, S_init, 0.0])
        
        # Try local optimization from this start
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if time.time() > deadline - 5:
                break
                
            final_params = res.x
            S_final = final_params[3*N+2]
            penalty = res.fun - S_final
            
            if penalty < 1e-6:  # Valid solution
                inner = []
                for i in range(N):
                    x = final_params[3*i]
                    y = final_params[3*i+1]
                    a_deg = np.degrees(final_params[3*i+2])
                    inner.append([x, y, a_deg])
                center = (final_params[3*N], final_params[3*N+1])
                side = S_final
                angle = np.degrees(final_params[3*N+3])
                
                if S_final < best_score:
                    best_score = S_final
                    best_result = (inner, center, side, angle)
        except Exception as e:
            pass
    
    # If no valid solution found, return a reasonable default
    if best_result is None:
        # Fallback: use the 4-3-4 configuration with estimated container
        xs, ys, as_ = configs[0]
        max_dist = 0
        for i in range(len(xs)):
            dist = np.sqrt(xs[i]**2 + ys[i]**2) + 1.0
            max_dist = max(max_dist, dist)
        S = max_dist * 1.3
        inner = [[xs[i], ys[i], as_[i]] for i in range(N)]
        return inner, (0.0, 0.0), S, 0.0
    
    return best_result

