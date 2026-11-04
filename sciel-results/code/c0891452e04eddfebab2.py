import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # 1. Generate initial configuration (Honeycomb cluster)
    # Hex 0 at origin
    # Hex 1-6 around it
    # Hex 7-10 added to the periphery
    
    sqrt3 = np.sqrt(3)
    # Distances
    d_center = sqrt3
    
    # Initial positions (x, y, theta)
    # Angles in radians for calculation, convert to degrees for output
    init_pos = []
    
    # Center
    init_pos.append([0.0, 0.0, 0.0])
    
    # Ring 1 (6 hexes)
    for k in range(6):
        angle = k * np.pi / 3
        x = d_center * np.cos(angle)
        y = d_center * np.sin(angle)
        init_pos.append([x, y, 0.0])
        
    # Ring 2 (add 4 more to make 11)
    # Place them at the "corners" of the ring 1 to extend the shape
    # Positions relative to ring 1 hexes
    # Hex 1 (0 deg) -> add at 0 deg further
    # Hex 2 (60 deg) -> add at 60 deg further
    # Hex 4 (180 deg) -> add at 180 deg further
    # Hex 5 (240 deg) -> add at 240 deg further
    # This creates a cross-like shape
    
    # Indices in init_pos: 0 is center, 1-6 are ring 1
    # Let's add to 1, 2, 4, 5
    additions = [1, 2, 4, 5]
    for idx in additions:
        base_x, base_y, base_theta = init_pos[idx]
        # Determine outward direction
        # The angle of the base hex from center
        angle_to_center = np.arctan2(base_y, base_x)
        # New position is base + d_center * (cos(angle), sin(angle))
        new_x = base_x + d_center * np.cos(angle_to_center)
        new_y = base_y + d_center * np.sin(angle_to_center)
        init_pos.append([new_x, new_y, 0.0])
        
    # We have 1 + 6 + 4 = 11 hexes.
    
    # Flatten to params vector
    x0 = np.array([p for pos in init_pos for p in pos])
    
    # Helper: Compute vertices of a hexagon
    def get_vertices(x, y, theta):
        # Regular hexagon side 1
        # Vertices at radius 1 from center
        # Base angles: 0, 60, 120, 180, 240, 300
        # If theta=0, vertex at (1,0)
        angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + theta
        vx = x + np.cos(angles)
        vy = y + np.sin(angles)
        return np.column_stack((vx, vy))
    
    # Helper: Check overlap and compute penalty
    # Using Separating Axis Theorem (SAT) for convex polygons
    # If separated, return 0. If not, return penalty.
    # To make it smooth-ish, we can return the minimum distance if negative (penetration)
    # But SAT gives boolean. 
    # Alternative: Compute min distance between two convex polygons.
    # This is computationally expensive but N is small.
    # For speed, we'll use a simple check: if any vertex of A is inside B or vice versa, or edges intersect.
    # Given the initial guess is valid and Nelder-Mead steps are small, we can use a hard penalty
    # if overlap is detected.
    
    def point_in_polygon(pt, poly):
        # Ray casting or winding number
        # Simple winding number
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            if ((y1 > y) != (y2 > y)):
                x_int = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < x_int:
                    inside = not inside
        return inside

    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def polygons_overlap(poly1, poly2):
        # Check if any vertex of poly1 is in poly2
        for v in poly1:
            if point_in_polygon(v, poly2):
                return True
        # Check if any vertex of poly2 is in poly1
        for v in poly2:
            if point_in_polygon(v, poly1):
                return True
        # Check edge intersections
        n1 = len(poly1)
        n2 = len(poly2)
        for i in range(n1):
            p1 = poly1[i]
            p2 = poly1[(i+1)%n1]
            for j in range(n2):
                p3 = poly2[j]
                p4 = poly2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    # Objective function
    def objective(params):
        # Unpack
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            theta = params[3*i+2]
            positions.append((x, y, theta))
            
        # Compute vertices for all
        vertices = []
        for x, y, theta in positions:
            vertices.append(get_vertices(x, y, theta))
            
        # Compute required outer side length S
        # Outer hexagon centered at (0,0), flat-top.
        # Normals to sides are at 30, 90, 150, 210, 270, 330 degrees
        # n_k = (cos(30+k*60), sin(30+k*60))
        # Condition: n_k . v <= apothem = sqrt(3)/2 * S
        # S >= 2/sqrt(3) * max(n_k . v)
        
        normals = []
        for k in range(6):
            angle = np.pi/6 + k * np.pi/3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals) # 6x2
        
        max_proj = 0.0
        for v_set in vertices:
            # v_set is 6x2
            # Project all vertices onto all normals
            # proj = v_set @ normals.T  (6x6)
            proj = v_set @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        S = (2.0 / sqrt3) * max_proj
        
        # Check overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap(vertices[i], vertices[j]):
                    penalty += 100.0 # Large penalty
                    
        return S + penalty

    # Run Nelder-Mead
    best_result = None
    best_score = np.inf
    
    # Multiple restarts with slight perturbations if time permits
    # Start with the initial guess
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if time.time() > deadline:
         # Return best found so far (initial if no result)
         # But we must return a valid result.
         # If minimize crashed or timed out, use initial
         pass
    else:
         if res.success or res.fun < best_score:
             best_score = res.fun
             best_result = res.x

    # If we have time, try a second run with a slightly different start or tighter tol
    if time.time() < deadline - 5.0:
         # Perturb initial guess
         np.random.seed(42)
         x0_perturbed = x0 + np.random.normal(0, 0.1, size=x0.shape)
         res2 = minimize(objective, x0_perturbed, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
         if time.time() < deadline - 1.0:
             if res2.fun < best_score:
                 best_score = res2.fun
                 best_result = res2.x

    # Use the best result
    if best_result is None:
        best_result = x0
        
    # Extract final parameters
    final_params = best_result
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        theta_rad = final_params[3*i+2]
        theta_deg = np.degrees(theta_rad)
        inner.append([x, y, theta_deg])
        
    # Compute final side length S from the best result
    # Re-evaluate objective to get S (penalty should be 0 if valid)
    # We can just re-run the S calculation part
    positions = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        theta = final_params[3*i+2]
        positions.append((x, y, theta))
        
    vertices = []
    for x, y, theta in positions:
        vertices.append(get_vertices(x, y, theta))
        
    normals = []
    for k in range(6):
        angle = np.pi/6 + k * np.pi/3
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for v_set in vertices:
        proj = v_set @ normals.T
        max_proj = max(max_proj, np.max(proj))
        
    final_S = (2.0 / sqrt3) * max_proj
    
    # Return
    return inner, (0.0, 0.0), final_S, 0.0

# EVOLVE-BLOCK-END
