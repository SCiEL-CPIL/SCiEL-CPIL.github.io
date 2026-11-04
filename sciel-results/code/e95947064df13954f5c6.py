import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    # Initial positions: 3x4 grid
    y_coords = [-SQRT3, 0.0, SQRT3]
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    
    init_pos = []
    for y in y_coords:
        for x in x_coords:
            init_pos.append([x, y])
            
    init_pos = np.array(init_pos)
    
    # Initial angles for inner hexagons: 0
    init_angles = np.zeros(N)
    
    # Initial container angle: 0
    init_container_angle = 0.0
    
    # Flatten parameters: [x1, y1, a1, ..., x12, y12, a12, theta_container]
    init_inner_params = np.column_stack([init_pos, init_angles])
    init_full_params = np.concatenate([init_inner_params.flatten(), [init_container_angle]])
    
    def objective(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, theta_container]
        inner_params = params[:3*N].reshape(N, 3)
        theta_c = params[3*N]
        
        x = inner_params[:, 0]
        y = inner_params[:, 1]
        a_rad = inner_params[:, 2] * np.pi / 180.0
        
        # Base vertices for unit hexagon (flat-top, angle 0)
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, SQRT3/2],
            [-0.5, SQRT3/2],
            [-1.0, 0.0],
            [-0.5, -SQRT3/2],
            [0.5, -SQRT3/2]
        ])
        
        all_verts = []
        for i in range(N):
            c = np.array([x[i], y[i]])
            theta = a_rad[i]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot = np.array([
                [cos_t, -sin_t],
                [sin_t, cos_t]
            ])
            verts = (base_verts @ rot.T) + c
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts) # Shape (72, 2)
        
        # Container normals
        # We define the container such that its sides are perpendicular to vectors at angles:
        # theta_c, theta_c + 60, ...
        # This means the "flat" sides are at angles theta_c+30, etc.
        # This is a valid regular hexagon.
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        normals = np.column_stack([np.cos(base_angles + theta_c), np.sin(base_angles + theta_c)])
        
        # Project all vertices onto normals
        projections = all_verts @ normals.T
        
        # Max absolute projection for each normal
        max_abs_proj = np.max(np.abs(projections), axis=0)
        
        # Required apothem
        required_apothem = np.max(max_abs_proj)
        
        # Side length S
        S = 2 * required_apothem / SQRT3
        
        return S

    # Optimization
    # Nelder-Mead is robust for this low-dimensional non-smooth problem (max of projections)
    res = minimize(objective, init_full_params, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = res.x
    best_inner = best_params[:3*N].reshape(N, 3)
    best_container_angle = best_params[3*N]
    
    inner_list = []
    for i in range(N):
        inner_list.append([best_inner[i, 0], best_inner[i, 1], best_inner[i, 2]])
    
    center = [0.0, 0.0]
    side_length = res.fun
    angle_degrees = best_container_angle * 180.0 / np.pi
    
    return inner_list, center, side_length, angle_degrees
# EVOLVE-BLOCK-END
