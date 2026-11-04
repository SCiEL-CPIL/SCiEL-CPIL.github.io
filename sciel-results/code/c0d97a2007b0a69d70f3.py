import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

def hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base = np.array([[1.0, 0.0],
                     [0.5, SQRT3/2],
                     [-0.5, SQRT3/2],
                     [-1.0, 0.0],
                     [-0.5, -SQRT3/2],
                     [0.5, -SQRT3/2]])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = base @ R.T + np.array([cx, cy])
    return verts

def required_side_length(positions, angles_deg, container_angle):
    """Compute the minimum side length of a regular hexagon (centered at origin, rotated by container_angle)
    that contains all vertices of the inner hexagons."""
    ca = np.radians(container_angle)
    cos_ca, sin_ca = np.cos(ca), np.sin(ca)
    
    # The container hexagon has 6 edges. For a flat-to-flat hexagon of side s centered at origin
    # with rotation theta, the 6 edge normals are at angles theta, theta+60, ..., theta+300.
    # The constraint is: for each vertex v, v . n_k <= s * cos(30°) = s * sqrt(3)/2 for each normal n_k.
    # Wait, let me think more carefully.
    
    # A regular hexagon of side s, centered at origin, with first vertex at angle 0:
    # Vertices: (s*cos(k*60°), s*sin(k*60°)) for k=0..5
    # Edge from vertex k to vertex k+1. The outward normal of edge k points at angle k*60°+30°.
    # The distance from center to each edge (apothem) is s * sqrt(3)/2.
    # So the constraint for each point p is: p · n_k <= s*sqrt(3)/2 for all 6 normals n_k.
    
    # With container rotation theta, normals are at angles theta + 30° + k*60°.
    max_proj = 0.0
    for k in range(6):
        normal_angle = ca + np.radians(30.0 + k * 60.0)
        n = np.array([np.cos(normal_angle), np.sin(normal_angle)])
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i])
            for v in verts:
                proj = v[0] * n[0] + v[1] * n[1]
                if proj > max_proj:
                    max_proj = proj
    
    s = max_proj / (SQRT3 / 2.0)
    return s

def objective(params):
    """Objective function for Nelder-Mead. Params: [66 edge weights, 12 angles, 1 container angle]"""
    N_local = 12
    w_upper = params[:66]
    angles = params[66:78]
    container_angle = params[78]
    
    # Build symmetric weight matrix
    W = np.zeros((N_local, N_local))
    idx = 0
    for i in range(N_local):
        for j in range(i + 1, N_local):
            W[i, j] = w_upper[idx]
            W[j, i] = w_upper[idx]
            idx += 1
    
    # Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    # Eigen decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return 1e6
    
    # Use 2nd and 3rd eigenvectors (skip trivial first one)
    x_coords = eigenvectors[:, 1]
    y_coords = eigenvectors[:, 2]
    positions = np.column_stack([x_coords, y_coords])
    
    # Scale so that max pairwise distance is reasonable (avoid degenerate)
    if N_local > 1:
        dists = []
        for i in range(N_local):
            for j in range(i + 1, N_local):
                d = np.linalg.norm(positions[i] - positions[j])
                dists.append(d)
        max_d = max(dists)
        if max_d > 1e-10:
            scale = 8.0 / max_d
        else:
            return 1e6
        positions = positions * scale
    
    # Add a penalty if hexagons overlap (minimum distance should be >= 2)
    penalty = 0.0
    for i in range(N_local):
        for j in range(i + 1, N_local):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0:
                penalty += (2.0 - d) ** 2
    
    s = required_side_length(positions, angles, container_angle)
    
    return s + 10.0 * penalty

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    # Initial edge weights: encourage a hexagonal lattice-like structure
    # Use a graph where nearby nodes in a 3x4 grid are connected
    init_weights = np.zeros(66)
    idx = 0
    # Connect adjacent nodes in a 3-row, 4-col grid
    for row in range(3):
        for col in range(4):
            i = row * 4 + col
            # Right neighbor
            if col < 3:
                j = row * 4 + (col + 1)
                init_weights[idx] = 1.0
                idx += 1
            # Down neighbor
            if row < 2:
                j = (row + 1) * 4 + col
                init_weights[idx] = 1.0
                idx += 1
            # Diagonal down-right
            if row < 2 and col < 3:
                j = (row + 1) * 4 + (col + 1)
                init_weights[idx] = 0.5
                idx += 1
            # Diagonal down-left
            if row < 2 and col > 0:
                j = (row + 1) * 4 + (col - 1)
                init_weights[idx] = 0.5
                idx += 1
    
    # Fill remaining with small values
    for k in range(idx, 66):
        init_weights[k] = 0.01
    
    angles0 = np.zeros(12)
    container0 = np.array([0.0])
    
    x0 = np.concatenate([init_weights, angles0, container0])
    
    # Run Nelder-Mead
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                               'adaptive': True})
    
    x_opt = result.x
    w_opt_upper = x_opt[:66]
    angles_opt = x_opt[66:78]
    container_angle_opt = x_opt[78]
    
    # Reconstruct positions
    W = np.zeros((12, 12))
    idx = 0
    for i in range(12):
        for j in range(i + 1, 12):
            W[i, j] = w_opt_upper[idx]
            W[j, i] = w_opt_upper[idx]
            idx += 1
    
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple grid
        positions = np.zeros((12, 2))
        for row in range(3):
            for col in range(4):
                positions[row*4+col, 0] = (col - 1.5) * 2.0
                positions[row*4+col, 1] = (row - 1) * SQRT3
        angles_opt = np.zeros(12)
        container_angle_opt = 0.0
    else:
        x_coords = eigenvectors[:, 1]
        y_coords = eigenvectors[:, 2]
        positions = np.column_stack([x_coords, y_coords])
        
        dists = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                dists.append(d)
        max_d = max(dists)
        if max_d > 1e-10:
            scale = 8.0 / max_d
        else:
            scale = 1.0
        positions = positions * scale
    
    s = required_side_length(positions, angles_opt, container_angle_opt)
    
    inner = []
    for i in range(12):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles_opt[i])])
    
    return inner, (0.0, 0.0), float(s), float(container_angle_opt)
