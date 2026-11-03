import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2, eps=1e-9):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 - eps or max_x2 < min_x1 - eps or max_y1 < min_y2 - eps or max_y2 < min_y1 - eps:
            return True

        # Point in Polygon check (Ray casting)
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2
        for pt in poly1:
            if point_in_poly(pt, poly2):
                return False
        
        # Check vertices of v2 in v1
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return False
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        
        return True

    def is_valid_config(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(polys[i], polys[j]):
                    return False
        return True

    def objective(params):
        """
        Objective function for optimization.
        Returns outer side length if valid, otherwise a large penalty.
        """
        if time.time() > deadline - 5:
            return 1e6
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if penalty > 1.0:
                    break
                if not check_disjoint(polys[i], polys[j]):
                    # Estimate overlap depth?
                    # Just add a large penalty.
                    penalty += 10.0
            if penalty > 1.0:
                break
        
        if penalty > 0:
            return penalty + calc_min_outer_side(positions, angles, 0.0)
            
        return calc_min_outer_side(positions, angles, 0.0)

    best_score = float('inf')
    best_positions = None
    best_angles = None

    # Strategy 1: Generate initial candidates from Honeycomb Lattice (Polyhexes)
    # This provides high-density, valid starting points.
    sqrt3 = math.sqrt(3)
    
    # Generate lattice points
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Candidate 1: 12 closest points (Compact cluster)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    if is_valid_config(init_positions, init_angles):
        score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if score < best_score:
            best_score = score
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()

    # Strategy 2: Differential Evolution (Global Search)
    # This allows escaping local minima of the lattice structure.
    # Bounds: Positions in [-5, 5], Angles in [0, 360]
    bounds = [(-5, 5)] * (2*N) + [(0, 360)] * N
    
    # Use a smaller population and maxiter to fit in time budget
    # DE is stochastic, so it might find non-lattice solutions.
    # We restrict maxiter and use a tight deadline check.
    
    try:
        # Differential Evolution
        # seed is fixed for reproducibility if needed, but random is fine for search
        # tol is tight to ensure convergence
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-5, 
            seed=42,
            polish=False  # We will polish manually
        )
        
        if res_de.fun < best_score:
            new_positions = res_de.x[:2*N].reshape(N, 2)
            new_angles = res_de.x[2*N:]
            
            if is_valid_config(new_positions, new_angles):
                best_score = res_de.fun
                best_positions = new_positions
                best_angles = new_angles
    except Exception:
        pass

    # Strategy 3: Local Refinement (Nelder-Mead) on the best candidate
    if best_positions is not None:
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res_nm = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                new_positions = res_nm.x[:2*N].reshape(N, 2)
                new_angles = res_nm.x[2*N:]
                
                if is_valid_config(new_positions, new_angles):
                    best_score = res_nm.fun
                    best_positions = new_positions
                    best_angles = new_angles
        except Exception:
            pass

    # Fallback if no valid configuration was found
    if best_positions is None:
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        best_positions = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        best_angles = np.zeros(N)
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
