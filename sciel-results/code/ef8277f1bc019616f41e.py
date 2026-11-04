import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial configuration: Hexagonal lattice
    # Lattice vectors
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 3/2])
    
    # Select 11 lattice points
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # (2,0), (-2,0), (0,2), (0,-2)
    indices = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    init_pos = []
    for i, j in indices:
        pos = i * u + j * v
        init_pos.append(pos)
    init_pos = np.array(init_pos)
    
    # Initial rotations: 0
    init_rot = np.zeros(N)
    
    # Initial outer hexagon
    init_phi = 0.0
    init_cx = 0.0
    init_cy = 0.0
    
    # Flatten parameters: [x0, y0, ..., x10, y10, theta0, ..., theta10, phi, cx, cy]
    x0 = np.concatenate([init_pos.flatten(), init_rot, [init_phi, init_cx, init_cy]])
    
    # Standard vertices for unit hexagon (rotation 0)
    # Vertices at angles 0, 60, ..., 300
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    std_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for outer hexagon (rotation 0)
    # Normals at angles 30, 90, ..., 330?
    # For a hexagon with vertex at 0, the side is at 30 deg?
    # Vertices at 0, 60...
    # Side 0 is between vertex 0 and 1. Midpoint angle 30. Normal angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330.
    normal_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 7)[:-1]
    std_normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def objective(params):
        if time.time() > deadline:
            return 1e6 # Large penalty to stop search? Or just return current best?
            # Better to let the optimizer handle it, but we can't easily break out.
            # We'll rely on the outer loop check.
            
        # Unpack
        pos = params[:22].reshape(N, 2)
        rots = params[22:33]
        phi = params[33]
        cx = params[34]
        cy = params[35]
        
        # Calculate vertices for all inner hexagons
        # pos: (N, 2), rots: (N,)
        # We need to rotate std_vertices by each rot and add pos.
        # Using broadcasting
        # cos_rots = np.cos(rots)
        # sin_rots = np.sin(rots)
        # R = ...
        # vertices = pos[:, np.newaxis, :] + ...
        
        # Efficient calculation
        cos_r = np.cos(rots)
        sin_r = np.sin(rots)
        
        # Rotation matrix for each hexagon
        # v_rot = [cos*r - sin*y, sin*r + cos*y]
        # std_vertices is (6, 2)
        # We want (N, 6, 2)
        
        # vx = std_vertices[:, 0], vy = std_vertices[:, 1]
        # vx_rot = cos_r[:, np.newaxis] * vx[np.newaxis, :] - sin_r[:, np.newaxis] * vy[np.newaxis, :]
        # vy_rot = sin_r[:, np.newaxis] * vx[np.newaxis, :] + cos_r[:, np.newaxis] * vy[np.newaxis, :]
        
        vx = std_vertices[:, 0]
        vy = std_vertices[:, 1]
        
        vx_rot = cos_r[:, np.newaxis] * vx[np.newaxis, :] - sin_r[:, np.newaxis] * vy[np.newaxis, :]
        vy_rot = sin_r[:, np.newaxis] * vx[np.newaxis, :] + cos_r[:, np.newaxis] * vy[np.newaxis, :]
        
        # Add center positions
        verts_x = pos[:, 0:1] + vx_rot
        verts_y = pos[:, 1:2] + vy_rot
        
        # Calculate outer hexagon normals
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        nx = std_normals[:, 0]
        ny = std_normals[:, 1]
        
        nx_rot = cos_phi * nx - sin_phi * ny
        ny_rot = sin_phi * nx + cos_phi * ny
        
        # Calculate projections
        # For each vertex (N, 6) and each normal (6), calculate dot product with (vertex - center)
        # dx = verts_x - cx, dy = verts_y - cy
        dx = verts_x - cx
        dy = verts_y - cy
        
        # Projection: dx * nx + dy * ny
        # We want the max over all N, 6, 6 combinations.
        # proj = dx[:, :, np.newaxis] * nx_rot[np.newaxis, np.newaxis, :] + dy[:, :, np.newaxis] * ny_rot[np.newaxis, np.newaxis, :]
        # max_proj = np.max(proj)
        
        # To save memory/time, we can loop over normals
        max_proj = -np.inf
        for j in range(6):
            proj_j = dx * nx_rot[j] + dy * ny_rot[j]
            max_j = np.max(proj_j)
            if max_j > max_proj:
                max_proj = max_j
        
        # Calculate overlap penalty
        # d_km >= sqrt(3)
        penalty = 0.0
        min_dist = np.sqrt(3)
        for k in range(N):
            for m in range(k+1, N):
                dx_km = pos[k, 0] - pos[m, 0]
                dy_km = pos[k, 1] - pos[m, 1]
                d_km = np.sqrt(dx_km**2 + dy_km**2)
                if d_km < min_dist:
                    deficit = min_dist - d_km
                    penalty += deficit**2
        
        lambda_penalty = 100.0
        return max_proj + lambda_penalty * penalty

    # Run Nelder-Mead
    # maxiter needs to be high enough
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    if time.time() > deadline:
         # If we are close to deadline, we might have a result from a previous run or we just use the best from res if it finished
         # But if it's still running, we can't interrupt it easily.
         # We assume it finishes or we accept the risk.
         # To be safe, we can check time before returning.
         pass
         
    if res.success or res.fun < 100: # Check if it found a reasonable solution
        best_params = res.x
    else:
        # Fallback to initial guess if optimization failed
        best_params = x0
        
    # Extract results
    pos = best_params[:22].reshape(N, 2)
    rots = best_params[22:33]
    phi = best_params[33]
    cx = best_params[34]
    cy = best_params[35]
    
    # Calculate S
    # max_proj is the apothem A
    # S = 2 * A / sqrt(3)
    # We need to re-evaluate the max_proj for the best params to get S
    # Or we can just use res.fun - penalty? No, res.fun includes penalty.
    # Let's re-calculate max_proj for the best params.
    
    cos_r = np.cos(rots)
    sin_r = np.sin(rots)
    vx = std_vertices[:, 0]
    vy = std_vertices[:, 1]
    vx_rot = cos_r[:, np.newaxis] * vx[np.newaxis, :] - sin_r[:, np.newaxis] * vy[np.newaxis, :]
    vy_rot = sin_r[:, np.newaxis] * vx[np.newaxis, :] + cos_r[:, np.newaxis] * vy[np.newaxis, :]
    verts_x = pos[:, 0:1] + vx_rot
    verts_y = pos[:, 1:2] + vy_rot
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    nx = std_normals[:, 0]
    ny = std_normals[:, 1]
    nx_rot = cos_phi * nx - sin_phi * ny
    ny_rot = sin_phi * nx + cos_phi * ny
    
    dx = verts_x - cx
    dy = verts_y - cy
    
    max_proj = -np.inf
    for j in range(6):
        proj_j = dx * nx_rot[j] + dy * ny_rot[j]
        max_j = np.max(proj_j)
        if max_j > max_proj:
            max_proj = max_j
            
    S = 2 * max_proj / np.sqrt(3)
    
    # Format output
    inner = []
    for k in range(N):
        inner.append([pos[k, 0], pos[k, 1], np.degrees(rots[k])])
    
    center = [cx, cy]
    side = S
    angle = np.degrees(phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
