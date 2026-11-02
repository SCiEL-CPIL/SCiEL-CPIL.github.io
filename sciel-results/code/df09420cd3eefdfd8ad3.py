import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # 6 normals for the outer hexagon (unit vectors)
    # Outer hexagon is regular. Let's align it such that normals are at 0, 60, 120...
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Vertices of a unit hexagon (radius 1)
    # Angles 0, 60, 120, 180, 240, 300
    hex_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_vertices = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))

    def get_vertices(params, i):
        x = params[3*i]
        y = params[3*i+1]
        theta = params[3*i+2]
        
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate vertices and translate
        verts = (R @ hex_vertices.T).T + np.array([x, y])
        return verts

    def objective(params):
        # Calculate required S
        max_apothem = 0.0
        for k in range(6):
            u = normals[k]
            h_k = 0.0
            for i in range(N):
                verts = get_vertices(params, i)
                # Support function: max dot product
                dots = verts @ u
                h_i = np.max(dots)
                if h_i > h_k:
                    h_k = h_i
            if h_k > max_apothem:
                max_apothem = h_k
        
        S = (2.0 / np.sqrt(3.0)) * max_apothem
        
        # Overlap penalty
        # We want d_ij >= sqrt(3) ~ 1.732
        # If they are closer, penalize.
        penalty = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(N):
            x_i = params[3*i]
            y_i = params[3*i+1]
            for j in range(i+1, N):
                x_j = params[3*j]
                y_j = params[3*j+1]
                d = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                if d < min_dist:
                    penalty += (min_dist - d)**2
        
        return S + 10.0 * penalty

    # Initial guess: Hexagonal cluster
    # Center at 0,0
    # 6 neighbors
    # 4 more
    # Let's place them in a rough hexagonal pattern
    init_params = np.zeros(3*N)
    
    # Helper to set position
    def set_pos(idx, x, y, theta=0.0):
        init_params[3*idx] = x
        init_params[3*idx+1] = y
        init_params[3*idx+2] = theta

    # 0: Center
    set_pos(0, 0.0, 0.0)
    
    # 1-6: Neighbors
    d = np.sqrt(3.0)
    for k in range(6):
        angle = k * np.pi / 3.0
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        set_pos(k+1, x, y)
        
    # 7-10: Next ring
    # Place them further out
    d2 = 2.0 * np.sqrt(3.0)
    # Angles offset by 30 degrees?
    for k in range(4):
        angle = (k * 2 + 1) * np.pi / 3.0 # 60, 180, 300, 60? No, distinct
        # Let's just put them at 0, 90, 180, 270 roughly
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        set_pos(7+k, x, y)

    # Optimize
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    # Extract results
    best_params = result.x
    
    # Calculate final S
    max_apothem = 0.0
    for k in range(6):
        u = normals[k]
        h_k = 0.0
        for i in range(N):
            verts = get_vertices(best_params, i)
            dots = verts @ u
            h_i = np.max(dots)
            if h_i > h_k:
                h_k = h_i
        if h_k > max_apothem:
            max_apothem = h_k
            
    S = (2.0 / np.sqrt(3.0)) * max_apothem
    
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle_deg = np.degrees(best_params[3*i+2])
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), S, 0.0

