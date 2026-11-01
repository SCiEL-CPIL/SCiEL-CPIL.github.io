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

def compute_outer_side(inner_verts_list, cx, cy):
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj = -np.inf
    for verts in inner_verts_list:
        rel_verts = verts - np.array([cx, cy])
        projs = rel_verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj * 2.0 / np.sqrt(3)

def generate_symmetric_hexagons(params):
    # params: [x1, y1, a1, x2, y2, a2]
    # Generate 12 hexagons using 6-fold symmetry
    base_hexes = []
    for i in range(2):
        x, y, a = params[3*i:3*i+3]
        base_hexes.append((x, y, a))
        
    all_hexes = []
    for k in range(6):
        angle_rad = k * np.pi / 3
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        for (bx, by, ba) in base_hexes:
            # Rotate position
            rx = bx * cos_a - by * sin_a
            ry = bx * sin_a + by * cos_a
            # Rotate angle
            ra = ba + k * 60.0
            all_hexes.append((rx, ry, ra))
            
    return all_hexes

def objective_symmetric(params):
    hexes = generate_symmetric_hexagons(params)
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in hexes]
    
    s = compute_outer_side(inner_verts_list, 0.0, 0.0)
    
    total_penalty = 0.0
    # Pairwise overlap
    # Optimization: Only check pairs that are likely to collide? 
    # With 12 hexes, 66 pairs. It's fast enough.
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            if pd > 0:
                total_penalty += pd**2 # Squared penalty for smoothness
    
    return s + 100.0 * total_penalty

def construct():
    # Bounds for differential evolution
    # x, y in [-5, 5], angle in [0, 60]
    # We restrict angle to [0, 60] because of symmetry? 
    # Actually, relative angle matters. 
    # Let's just bound angle to [0, 360] and let symmetry handle the rest?
    # No, if we rotate the base hex, the pattern rotates.
    # So we can fix the "orientation" of the pattern by restricting the first hex's angle to [0, 60]?
    # Not necessarily. The inner hexes can be rotated independently relative to the radial direction.
    # But due to 6-fold symmetry, the relative orientation is preserved.
    
    bounds = [(-5, 5), (-5, 5), (0, 60), (-5, 5), (-5, 5), (0, 60)]
    
    # Initial guess: 
    # Hex 1: near center? No, 12 hexes. 
    # Maybe one at (0,0) and one at (sqrt(3), 0)?
    # If we use 2 base hexes, and 6-fold symmetry, we get 12.
    # If Hex 1 is at (0,0), its rotations are all (0,0). That's 6 overlapping hexes. Bad.
    # So Hex 1 and Hex 2 must be distinct from origin.
    # Let's try:
    # Hex 1: (1.5, 0)
    # Hex 2: (0.75, 1.3) approx?
    
    x0 = [1.5, 0.0, 0.0, 0.75, 1.3, 0.0]
    
    # Use differential evolution for global search
    result_de = differential_evolution(objective_symmetric, bounds, seed=42, maxiter=100, tol=1e-6, polish=True)
    
    # Refine with L-BFGS-B
    result = minimize(objective_symmetric, result_de.x, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    params = result.x
    hexes = generate_symmetric_hexagons(params)
    
    inner = []
    for x, y, a in hexes:
        inner.append([float(x), float(y), float(a)])
        
    # Calculate final s
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side(inner_verts_list, 0.0, 0.0)
    
    return inner, (0.0, 0.0), float(s), 0.0

