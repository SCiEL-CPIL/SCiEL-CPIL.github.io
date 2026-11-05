import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Unit hexagon support function
    # h(phi) for unit hexagon (side 1)
    # phi is the angle of the direction vector
    def h_unit(phi):
        # Period is pi/3
        phi_mod = phi % (math.pi / 3)
        # Center of the period is pi/6
        # h(phi) = (sqrt(3)/2) / cos(phi_mod - pi/6)
        return (math.sqrt(3) / 2) / math.cos(phi_mod - math.pi / 6)
    
    # Calculate the required outer side length S for a given configuration
    # centers: (N, 2) array
    # angles: (N,) array of inner hexagon rotations
    # psi: outer hexagon rotation
    def calc_side_length(centers, angles, psi):
        # Normals of the outer hexagon
        # The outer hexagon sides are perpendicular to these normals
        # Normals are at psi, psi + pi/3, psi + 2pi/3
        max_h = 0.0
        for k in range(3):
            u_angle = psi + k * math.pi / 3
            u_x = math.cos(u_angle)
            u_y = math.sin(u_angle)
            
            # Support function of the union is the max of supports of individuals
            # h_union(u) = max_i ( c_i . u + h_unit(angle_i - u_angle) )
            # Note: h_unit takes the relative angle between the direction and the hexagon's orientation
            # If the hexagon is rotated by theta_i, its "local" axes are rotated.
            # The direction u is at angle u_angle.
            # The relative angle is u_angle - theta_i.
            
            current_max = -np.inf
            for i in range(N):
                cx, cy = centers[i]
                dot = cx * u_x + cy * u_y
                rel_angle = u_angle - angles[i]
                h_i = h_unit(rel_angle)
                val = dot + h_i
                if val > current_max:
                    current_max = val
            
            if current_max > max_h:
                max_h = current_max
        
        # S = (2/sqrt(3)) * max_h
        return (2.0 / math.sqrt(3)) * max_h

    # Objective function for optimization
    # Variables: x0, y0, x1, y1, ..., xN-1, yN-1, psi
    def objective(params):
        centers = params[:2*N].reshape(N, 2)
        psi = params[2*N]
        # Assume all inner hexagons are aligned (angles = 0)
        angles = np.zeros(N)
        return calc_side_length(centers, angles, psi)

    # Initial configuration: 3x4 grid (11 points)
    # Hexagonal grid coordinates
    # Let's place them in a compact cluster
    # Axial coordinates (q, r) -> Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # We want 11 points.
    # A 3x4 block has 12. Remove one corner.
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Generate 11 points
    points = []
    # Rows of 3, 4, 3, 1? 
    # Let's try a standard 3x4 minus one.
    # q in 0..2, r in 0..3
    for r in range(4):
        for q in range(3):
            if r == 3 and q == 2: # Skip one to make it 11
                continue
            points.append(axial_to_cartesian(q, r))
            
    initial_centers = np.array(points)
    # Center the configuration
    initial_centers -= np.mean(initial_centers, axis=0)
    
    initial_psi = 0.0
    initial_params = np.hstack([initial_centers.flatten(), [initial_psi]])
    
    best_params = initial_params.copy()
    best_score = objective(best_params)
    
    # Optimization Loop
    # Use Nelder-Mead from scipy if available, else simple random search
    try:
        from scipy.optimize import minimize
        # Run multiple restarts
        restarts = 5
        for i in range(restarts):
            if time.time() > deadline:
                break
            
            # Perturb initial guess
            if i == 0:
                x0 = initial_params
            else:
                x0 = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
            
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
    except Exception:
        # Fallback to simple random search if scipy fails
        for i in range(10000):
            if time.time() > deadline:
                break
            params = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
            score = objective(params)
            if score < best_score:
                best_score = score
                best_params = params

    # Extract best configuration
    best_centers = best_params[:2*N].reshape(N, 2)
    best_psi = best_params[2*N]
    
    # Construct inner list
    inner = []
    for i in range(N):
        x, y = best_centers[i]
        inner.append([float(x), float(y), 0.0]) # Angle 0.0
    
    # Outer hexagon
    outer_center = [0.0, 0.0]
    outer_side = float(best_score)
    outer_angle = float(best_psi)
    
    return inner, outer_center, outer_side, outer_angle

