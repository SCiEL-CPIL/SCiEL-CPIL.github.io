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

def inside_violation(inner_verts, outer_verts):
    # inner must be inside outer
    outer_normals = get_normals(outer_verts)
    max_viol = 0.0
    for normal in outer_normals:
        # normal points outward from outer
        inner_max = (inner_verts @ normal).max()
        outer_max = (outer_verts @ normal).max()
        viol = inner_max - outer_max
        if viol > max_viol:
            max_viol = viol
    return max_viol

def objective(params):
    # params: [x1,y1,a1, ..., x12,y12,a12, cx, cy, s, psi]
    inner_params = params[:3 * N]
    cx, cy, s, psi = params[3 * N:3 * N + 4]
    
    if s <= 0.5:  # ensure valid
        return 1e6
    
    outer_verts = hex_vertices(cx, cy, s, psi)
    
    total_penalty = 0.0
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        # inside violation
        viol = inside_violation(verts, outer_verts)
        total_penalty += max(0.0, viol)
    
    # pairwise overlap
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += max(0.0, pd)
    
    return s + 10.0 * total_penalty

def initial_guess():
    # Hexagonal lattice initial guess
    # Place 12 hexagons in a hexagonal cluster
    # Use triangular lattice with spacing = sqrt(3) (center distance for adjacent hexagons)
    spacing = np.sqrt(3)
    positions = []
    # Generate a hexagonal cluster of 12 hexagons
    # Center at (0,0), then rings
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexes at distance spacing
    # Ring 2: we need 5 more to make 12
    positions.append((0.0, 0.0))
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    # 5 more around ring 2
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
    
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    # Outer hexagon: center at (0,0), side large enough, rotation 0
    params.extend([0.0, 0.0, 5.0, 0.0])
    return np.array(params)

def construct():
    x0 = initial_guess()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
    params = result.x
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy, s, psi = params[3*N:3*N+4]
    return inner, (float(cx), float(cy)), float(s), float(psi)
