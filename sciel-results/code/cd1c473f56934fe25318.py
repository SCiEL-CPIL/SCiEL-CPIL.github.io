import numpy as np
from scipy.optimize import minimize
import itertools

def get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def sat_overlap_depth(p1, p2):
    axes = []
    for p in [p1, p2]:
        for i in range(len(p)):
            edge = p[(i+1)%len(p)] - p[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n = n / norm
                axes.append(n)
    
    min_depth = float('inf')
    for n in axes:
        proj1 = np.dot(p1, n)
        proj2 = np.dot(p2, n)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def objective(params):
    # params: 12*(x,y,angle) + 1 (theta_out)
    # 12*3 + 1 = 37
    inner_params = params[:36].reshape(12, 3)
    theta_out = params[36]
    
    vertices_list = []
    for i in range(12):
        x, y, angle = inner_params[i]
        v = get_hex_vertices(x, y, 1.0, angle)
        vertices_list.append(v)
        
    # Calculate required outer side S
    # Outer hex normals
    theta_out_rad = np.deg2rad(theta_out)
    normals = []
    for k in range(6):
        theta = theta_out_rad + k * np.pi / 3
        # Normal to side
        n = np.array([np.cos(theta), np.sin(theta)])
        normals.append(n)
        
    max_proj = 0.0
    for n in normals:
        # Max projection of any vertex of any inner hex
        for v in vertices_list:
            proj = np.dot(v, n)
            if np.max(proj) > max_proj:
                max_proj = np.max(proj)
                
    apothem = max_proj
    S = (2.0 / np.sqrt(3.0)) * apothem
    
    # Check overlaps
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            depth = sat_overlap_depth(vertices_list[i], vertices_list[j])
            if depth > 0:
                penalty += depth * depth * 100.0
                
    return S + penalty

def construct():
    # Initial guess: Hex grid
    # 4 rows of 3? Or 3 rows of 4?
    # Let's try 3 rows of 4.
    # x = c * sqrt(3) + (r % 2) * sqrt(3)/2
    # y = r * 1.5
    
    inner = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2.0
            y = r * 1.5
            # Center the grid
            x -= 1.5 * np.sqrt(3)
            y -= 1.5
            inner.append([x, y, 0.0])
            
    params0 = np.array(inner).flatten()
    params0 = np.append(params0, 0.0) # theta_out
    
    # Optimize
    res = minimize(objective, params0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000})
    
    best_params = res.x
    inner_params = best_params[:36].reshape(12, 3)
    theta_out = best_params[36]
    
    # Reconstruct inner list
    inner_result = []
    for i in range(12):
        x, y, angle = inner_params[i]
        inner_result.append([x, y, angle])
        
    # Calculate final S
    vertices_list = []
    for i in range(12):
        x, y, angle = inner_result[i]
        v = get_hex_vertices(x, y, 1.0, angle)
        vertices_list.append(v)
        
    theta_out_rad = np.deg2rad(theta_out)
    normals = []
    for k in range(6):
        theta = theta_out_rad + k * np.pi / 3
        n = np.array([np.cos(theta), np.sin(theta)])
        normals.append(n)
        
    max_proj = 0.0
    for n in normals:
        for v in vertices_list:
            proj = np.dot(v, n)
            if np.max(proj) > max_proj:
                max_proj = np.max(proj)
                
    S = (2.0 / np.sqrt(3.0)) * max_proj
    
    return inner_result, (0.0, 0.0), S, theta_out

