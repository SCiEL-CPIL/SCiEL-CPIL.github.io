import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(x, y, theta, side=1.0):
    """
    Returns vertices of a regular hexagon with side length 'side',
    centered at (x, y), rotated by theta radians.
    Vertices are ordered counter-clockwise.
    """
    # Angles for vertices of a regular hexagon (pointy top at theta=0)
    # Standard pointy top: vertices at 90, 150, 210, 270, 330, 30 degrees?
    # Let's use: 0, 60, 120, 180, 240, 300 degrees relative to center.
    # This is a "flat top" hexagon if we consider x-axis horizontal?
    # Let's stick to a standard definition.
    # Vertex k at angle theta + k * pi/3.
    angles = theta + np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return np.array([x + side * np.cos(angles), y + side * np.sin(angles)]).T

def _signed_dist_point_to_hex(pt, center, theta, side=1.0):
    """
    Calculate signed distance from point pt to the boundary of a hexagon.
    Positive if inside, negative if outside.
    """
    # Transform point to hexagon's local coordinate system (centered at origin, unrotated)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    
    # Rotate by -theta
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # Hexagon inequalities for side 'side' (flat-top or pointy-top?)
    # Our vertices were defined as theta + k*pi/3.
    # If theta=0, vertices are at 0, 60, ...
    # This is a "pointy-top" orientation relative to the x-axis?
    # Vertex 0 is at (side, 0). Vertex 1 is at (side/2, side*sqrt(3)/2).
    # So the "flat" sides are perpendicular to the bisectors of the vertices.
    # The apothem (distance from center to middle of side) is side * sqrt(3)/2.
    # The sides are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Equation of side at angle alpha: x*cos(alpha) + y*sin(alpha) <= apothem.
    
    apothem = side * np.sqrt(3) / 2.0
    # Angles of the normals to the sides
    normal_angles = np.pi/6 + np.linspace(0, 2 * np.pi, 6, endpoint=False)
    
    cos_n = np.cos(normal_angles)
    sin_n = np.sin(normal_angles)
    
    # Distance from origin to each side line (signed)
    # dist = apothem - (lx*cos_n + ly*sin_n)
    # If point is inside, lx*cos_n + ly*sin_n <= apothem for all n.
    # So dist >= 0.
    dists = apothem - (lx * cos_n + ly * sin_n)
    
    # The signed distance to the boundary is the minimum of these distances.
    # If inside, min dist > 0.
    # If outside, min dist < 0.
    return np.min(dists)

def _pairwise_penalty(hex_i_verts, hex_i_center, hex_i_theta, hex_j_verts, hex_j_center, hex_j_theta):
    """
    Calculate smooth penalty for overlap between two hexagons.
    """
    penalty = 0.0
    
    # Check vertices of i against hex j
    for v in hex_i_verts:
        d = _signed_dist_point_to_hex(v, hex_j_center, hex_j_theta, 1.0)
        if d < 0:
            penalty += d * d
            
    # Check vertices of j against hex i
    for v in hex_j_verts:
        d = _signed_dist_point_to_hex(v, hex_i_center, hex_i_theta, 1.0)
        if d < 0:
            penalty += d * d
            
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    # Initial parameters: x1..xN, y1..yN, theta1..thetaN, S
    # Total 34 parameters
    
    # Initial guess
    # Center hex
    x0 = [0.0]
    y0 = [0.0]
    t0 = [0.0]
    
    # 6 surrounding hexes
    for k in range(6):
        angle = k * np.pi / 3.0
        r = 1.8 # Slightly less than 2 to allow rotation
        x0.append(r * np.cos(angle))
        y0.append(r * np.sin(angle))
        t0.append(angle)
        
    # 4 outer hexes
    for k in range(4):
        angle = np.pi/6 + k * np.pi / 2.0 # 30, 90, 150, 210
        r = 3.2
        x0.append(r * np.cos(angle))
        y0.append(r * np.sin(angle))
        t0.append(angle)
        
    S0 = 4.0
    
    x0 = np.array(x0)
    y0 = np.array(y0)
    t0 = np.array(t0)
    
    params0 = np.concatenate([x0, y0, t0, [S0]])
    
    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        x = params[:N]
        y = params[N:2*N]
        theta = params[2*N:3*N]
        S = params[3*N]
        
        if S < 2.5: # Lower bound
            return 1e9
            
        # Precompute vertices for all inner hexes
        verts_list = []
        centers_list = []
        thetas_list = []
        
        for i in range(N):
            v = _hex_vertices(x[i], y[i], theta[i], 1.0)
            verts_list.append(v)
            centers_list.append((x[i], y[i]))
            thetas_list.append(theta[i])
            
        # Outer Hexagon
        # Center (0,0), Rotation 0, Side S
        outer_verts = _hex_vertices(0.0, 0.0, 0.0, S)
        outer_center = (0.0, 0.0)
        outer_theta = 0.0
        
        penalty_out = 0.0
        for i in range(N):
            for v in verts_list[i]:
                d = _signed_dist_point_to_hex(v, outer_center, outer_theta, S)
                if d < 0:
                    penalty_out += d * d
                    
        penalty_ov = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = _pairwise_penalty(
                    verts_list[i], centers_list[i], thetas_list[i],
                    verts_list[j], centers_list[j], thetas_list[j]
                )
                penalty_ov += p
                
        # We want to minimize S, but heavily penalize violations.
        # If penalty is large, S doesn't matter as much, but we want S small.
        # Use a weighted sum.
        cost = S + 100.0 * penalty_out + 100.0 * penalty_ov
        
        return cost

    # Run optimization
    best_cost = 1e9
    best_params = params0
    
    # Try a few random restarts
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            p_init = params0
        else:
            # Perturb initial guess
            p_init = params0.copy()
            p_init[:2*N] += np.random.uniform(-0.5, 0.5, 2*N)
            p_init[2*N:3*N] += np.random.uniform(-0.1, 0.1, N)
            p_init[3*N] = 4.0 + np.random.uniform(-0.5, 0.5)
            
        res = minimize(objective, p_init, method='Powell', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
            
        if time.time() > deadline - 10.0:
            break
            
    # Extract best result
    x = best_params[:N]
    y = best_params[N:2*N]
    theta = best_params[2*N:3*N]
    S = best_params[3*N]
    
    # Convert theta to degrees
    theta_deg = np.degrees(theta)
    
    inner = [[x[i], y[i], theta_deg[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
