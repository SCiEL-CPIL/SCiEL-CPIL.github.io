import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = np.array([[cx + side * np.cos(a + k * np.pi / 3),
                       cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])
    return verts

def get_normals(verts):
    normals = []
    for i in range(len(verts)):
        edge = verts[(i + 1) % len(verts)] - verts[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals.append(normal / norm)
    return normals

def penetration_depth(v1, v2):
    normals = get_normals(v1) + get_normals(v2)
    min_overlap = np.inf
    for normal in normals:
        p1 = v1 @ normal
        p2 = v2 @ normal
        overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap

def compute_outer_side(inner_verts_list, cx, cy):
    # Outer hexagon centered at (cx, cy), rotation 0
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj = -np.inf
    for verts in inner_verts_list:
        rel_verts = verts - np.array([cx, cy])
        projs = rel_verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    inner_params = params[:3 * N]
    cx, cy = params[3 * N:3 * N + 2]
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side(inner_verts_list, cx, cy)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += max(0.0, pd)
            
    return s + 1000.0 * total_penalty

def initial_guess(seed):
    np.random.seed(seed)
    spacing = np.sqrt(3) * 1.05
    positions = []
    positions.append((0.0, 0.0))
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        x += np.random.uniform(-0.2, 0.2)
        y += np.random.uniform(-0.2, 0.2)
        a = np.random.uniform(0, 60)
        params.extend([x, y, a])
    params.extend([0.0, 0.0])
    return np.array(params)

def construct():
    best_score = np.inf
    best_params = None
    
    for seed in range(15):
        x0 = initial_guess(seed)
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success:
                inner_params = result.x[:3 * N]
                inner_verts_list = []
                for i in range(N):
                    x, y, a = inner_params[3*i:3*i+3]
                    verts = hex_vertices(x, y, 1.0, a)
                    inner_verts_list.append(verts)
                
                max_overlap = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                        if pd > max_overlap:
                            max_overlap = pd
                
                if max_overlap < 1e-6:
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_params = result.x
        except Exception:
            pass

    if best_params is None:
        x0 = initial_guess(0)
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
        best_params = result.x
        inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in 
                            [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]]
        cx, cy = best_params[3*N], best_params[3*N+1]
        best_score = compute_outer_side(inner_verts_list, cx, cy)

    params = best_params
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy = params[3*N:3*N+2]
    
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side(inner_verts_list, cx, cy)
    
    return inner, (float(cx), float(cy)), float(s), 0.0
