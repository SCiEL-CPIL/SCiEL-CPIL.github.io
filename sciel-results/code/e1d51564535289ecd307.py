import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    def calc_min_outer_side(positions, angles, outer_angle):
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

    def check_disjoint(poly1, poly2):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint, False if overlapping.
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return True

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

        for pt in poly1:
            if point_in_poly(pt, poly2):
                return False
        
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return False
        
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

    def objective(params):
        if time.time() > deadline:
            return 1e10

        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:2*N+N]
        outer_angle = params[2*N+N]

        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if penalty > 10.0:
                    break
                if not check_disjoint(polys[i], polys[j]):
                    penalty += 1.0
            if penalty > 10.0:
                break
        
        if penalty > 0:
            return 1000.0 + penalty + calc_min_outer_side(positions, angles, outer_angle)
            
        return calc_min_outer_side(positions, angles, outer_angle)

    # Bounds for differential evolution
    # Positions: likely within [-5, 5]
    # Angles: [0, 60] due to symmetry
    # Outer angle: [0, 60]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
    for _ in range(N):
        bounds.append((0.0, 60.0)) # inner angle
    bounds.append((0.0, 60.0)) # outer angle

    # Initial best score using a simple heuristic (honeycomb)
    best_score = float('inf')
    best_params = None
    
    sqrt3 = math.sqrt(3)
    # Honeycomb centers
    init_coords = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            init_coords.append((x, y))
    
    # Select 12 closest
    init_coords.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(init_coords[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_angles = np.zeros(N)
    init_outer = 0.0
    
    init_params = np.concatenate([init_pos.flatten(), init_angles, [init_outer]])
    
    # Check if initial is valid
    polys_init = [get_hex_verts(init_pos[i][0], init_pos[i][1], 1.0, 0.0) for i in range(N)]
    valid_init = True
    for i in range(N):
        for j in range(i+1, N):
            if not check_disjoint(polys_init[i], polys_init[j]):
                valid_init = False
                break
        if not valid_init: break
    
    if valid_init:
        best_score = calc_min_outer_side(init_pos, init_angles, init_outer)
        best_params = init_params.copy()

    # Run Differential Evolution
    # maxiter is set low to allow multiple restarts or just one long run?
    # Given 150s limit, DE can be slow. 
    # Let's try one run with reasonable iterations.
    # popsize=15, maxiter=200
    # If it times out, we return the best found so far? 
    # DE doesn't easily return partial results without a callback or exception.
    # We can use a callback to track best.
    
    de_best = [best_score]
    de_best_params = [best_params.copy()] if best_params is not None else [None]
    
    def de_callback(xk, convergence):
        # xk is the current best individual
        # Check time
        if time.time() > deadline - 5:
            raise Exception("Time limit exceeded")
        
        # Calculate score for xk
        # Note: xk might contain invalid configs (high penalty)
        # We rely on the objective function to penalize them.
        # But we should verify validity for the "best" tracking?
        # The objective returns a high value for invalid ones, so DE will avoid them if possible.
        # However, if all are invalid, it might converge to a "least invalid" one.
        
        # Let's just record the parameter vector. We will evaluate it later.
        de_best_params[0] = xk.copy()
        
        # We can't easily get the exact score without calling objective again, 
        # but DE tracks it. We can assume xk is the best found by DE's internal logic.
        # To be safe, we'll evaluate it if it looks promising? 
        # No, just store it.

    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=20, 
            tol=1e-5, 
            seed=42,
            callback=de_callback,
            update_interval=10 # Check callback every 10 iterations
        )
        
        # Evaluate the result
        if res.fun < best_score:
            # Verify validity
            final_params = res.x
            positions = final_params[:2*N].reshape(N, 2)
            angles = final_params[2*N:2*N+N]
            outer_angle = final_params[2*N+N]
            
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = res.fun
                best_params = final_params.copy()
                
    except Exception as e:
        # Time limit or other error
        # Use the best params stored in callback if they are better
        if de_best_params[0] is not None:
            candidate_params = de_best_params[0]
            positions = candidate_params[:2*N].reshape(N, 2)
            angles = candidate_params[2*N:2*N+N]
            outer_angle = candidate_params[2*N+N]
            
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score = calc_min_outer_side(positions, angles, outer_angle)
                if score < best_score:
                    best_score = score
                    best_params = candidate_params.copy()

    # If DE didn't find anything better, use the initial heuristic
    if best_params is None:
        best_params = init_params
        best_score = calc_min_outer_side(init_pos, init_angles, init_outer)

    final_params = best_params
    final_positions = final_params[:2*N].reshape(N, 2)
    final_angles = final_params[2*N:2*N+N]
    final_outer_angle = final_params[2*N+N]
    
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

