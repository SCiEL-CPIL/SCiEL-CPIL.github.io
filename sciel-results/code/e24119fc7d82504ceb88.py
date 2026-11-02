import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 115.0  # 120s limit, leave 5s margin

def construct():
    start_time = time.time()
    
    # Parameters:
    # r1: radius of inner ring (6 hexagons)
    # r2: radius of outer ring (6 hexagons)
    # theta1: angular offset of inner ring
    # theta2: angular offset of outer ring
    # theta_in: orientation of all inner hexagons
    # S: side length of outer hexagon
    # theta_out: orientation of outer hexagon
    
    # Initial guess
    # Inner ring at sqrt(3), aligned with 0, 60...
    # Outer ring at 3.0, offset by 30 deg
    # Inner hexagons aligned with 0
    # Outer hexagon aligned with 0
    x0 = [np.sqrt(3), 3.0, 0.0, np.pi/6, 0.0, 5.0, 0.0]
    
    def objective(params):
        r1, r2, theta1, theta2, theta_in, S, theta_out = params
        
        # Ensure positive radii and side length
        if r1 < 0.1 or r2 < 0.1 or S < 1.0:
            return 1e6
        
        # Generate centers
        centers = np.zeros((N, 2))
        
        # Inner ring (6 hexagons)
        for i in range(6):
            angle = i * np.pi / 3 + theta1
            centers[i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
            
        # Outer ring (6 hexagons)
        for i in range(6):
            angle = i * np.pi / 3 + theta2
            centers[6 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
        penalty = 0.0
        
        # 1. Check Overlaps between inner hexagons
        # Distance between centers must be >= sqrt(3)
        min_dist_sq = 3.0 # (sqrt(3))^2
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    # Smooth penalty
                    penalty += (min_dist_sq - dist_sq) * 100.0
        
        # 2. Check Containment
        # Outer hexagon center at (0,0) for simplicity? 
        # The problem allows outer center to move. 
        # Due to symmetry of the inner packing (if theta1, theta2 are optimized), 
        # the optimal outer center is likely (0,0). 
        # We can fix outer center to (0,0) to reduce variables.
        # Let's fix outer center to (0,0).
        
        # Outer hexagon vertices/normal check
        # Normals for outer hexagon
        angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta_out
        normals = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
        
        # Distance from center to side of outer hexagon
        d_side = S * np.sqrt(3) / 2
        
        # For each inner hexagon, check its vertices
        # Inner hexagon orientation is theta_in
        angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + theta_in
        verts_in = np.column_stack((np.cos(angles_in), np.sin(angles_in)))
        
        for i in range(N):
            cx, cy = centers[i]
            # Vertices of this inner hexagon
            verts = verts_in + np.array([cx, cy])
            
            # Vector from outer center (0,0) to each vertex
            # dots = normals @ verts.T
            # Violation if dot > d_side
            
            # Efficient check: max dot product of verts with normals
            # verts is (6, 2), normals is (6, 2)
            # We need max over normals for each vertex? 
            # No, for a vertex to be inside, dot(v, n) <= d_side for ALL normals n.
            # So for each vertex v, check if max(dot(v, n) for all n) <= d_side.
            
            # Compute all dots: (6 verts) x (6 normals)
            dots = verts @ normals.T
            max_dots = np.max(dots, axis=1) # Max projection for each vertex
            
            viol = np.maximum(0, max_dots - d_side)
            penalty += np.sum(viol**2) * 10.0
            
        return S + penalty

    best_params = x0
    best_score = objective(x0)
    
    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and handles the penalty well
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # If time allows, try a second run with different initial conditions
    if time.time() - start_time < DEADLINE - 20:
        x0_2 = [np.sqrt(3), 2.8, 0.0, 0.0, 0.0, 4.5, 0.0]
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    r1, r2, theta1, theta2, theta_in, S, theta_out = best_params
    
    # Construct inner list
    inner = []
    # Inner ring
    for i in range(6):
        angle = i * np.pi / 3 + theta1
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        inner.append([float(x), float(y), float(np.degrees(theta_in))])
        
    # Outer ring
    for i in range(6):
        angle = i * np.pi / 3 + theta2
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        inner.append([float(x), float(y), float(np.degrees(theta_in))])
        
    # Outer hexagon center is (0,0)
    return inner, (0.0, 0.0), float(S), float(np.degrees(theta_out))
