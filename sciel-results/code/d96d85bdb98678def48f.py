import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def inside_violation(inner_verts, outer_verts):
    outer_normals = get_normals(outer_verts)
    max_viol = 0.0
    for normal in outer_normals:
        inner_max = (inner_verts @ normal).max()
        outer_max = (outer_verts @ normal).max()
        viol = inner_max - outer_max
        if viol > max_viol:
            max_viol = viol
    return max_viol

def objective_smooth(params):
    # params: [x1,y1,a1, ..., x12,y12,a12, cx, cy, s, psi]
    inner_params = params[:3 * N]
    cx, cy, s, psi = params[3 * N:3 * N + 4]
    
    if s <= 0.5:
        return 1e6
    
    outer_verts = hex_vertices(cx, cy, s, psi)
    
    total_penalty = 0.0
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        viol = inside_violation(verts, outer_verts)
        total_penalty += (max(0.0, viol)) ** 2
    
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += (max(0.0, pd)) ** 2
            
    return s + 100.0 * total_penalty

def construct():
    # Phase 1: Global Search with DE
    bounds = []
    for _ in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
        bounds.append((0.0, 180.0))
    bounds.append((-2.0, 2.0)) # cx
    bounds.append((-2.0, 2.0)) # cy
    bounds.append((2.5, 5.0))  # s
    bounds.append((0.0, 60.0)) # psi
    
    de_result = differential_evolution(
        objective_smooth, 
        bounds, 
        seed=42, 
        maxiter=50, 
        popsize=10,
        tol=1e-5,
        polish=False
    )
    
    x0 = de_result.x
    
    # Phase 2: Local Refinement with L-BFGS-B
    result = minimize(objective_smooth, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    
    params = result.x
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy, s, psi = params[3*N:3*N+4]
    
    return inner, (float(cx), float(cy)), float(s), float(psi)
