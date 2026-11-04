import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def get_hex_verts(cx, cy, theta):
        angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + theta
        return np.column_stack([np.cos(angles), np.sin(angles)]) + np.array([cx, cy])

    def polygons_overlap_fast(v1, v2, c1, c2):
        # Pre-filter: distance between centers
        dist = np.linalg.norm(c1 - c2)
        if dist > 2.0:
            return False
            
        # SAT
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        
        all_normals = np.vstack([normals1, normals2])
        norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        all_normals = all_normals / norms
        
        for normal in all_normals:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return False
        return True

    def calc_enclosing_hex_side(vertices):
        thetas = np.linspace(0, np.pi / 3, 50)
        min_max_proj = float('inf')
        
        for theta in thetas:
            max_proj = 0
            for k in range(3):
                alpha = theta + k * np.pi / 3
                proj = vertices[:, 0] * np.cos(alpha) + vertices[:, 1] * np.sin(alpha)
                max_proj = max(max_proj, np.max(np.abs(proj)))
            if max_proj < min_max_proj:
                min_max_proj = max_proj
        
        S = min_max_proj * 2 / np.sqrt(3)
        return S

    def objective(params):
        if time.time() > deadline:
            return 1000000
            
        positions = params[:N*2].reshape(N, 2)
        rotations = params[N*2:]
        
        all_verts_list = []
        for i in range(N):
            v = get_hex_verts(positions[i, 0], positions[i, 1], rotations[i])
            all_verts_list.append(v)
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap_fast(all_verts_list[i], all_verts_list[j], positions[i], positions[j]):
                    return 1000000 + i + j
        
        all_vertices = np.vstack(all_verts_list)
        side = calc_enclosing_hex_side(all_vertices)
        return side

    bounds = [(-10, 10)] * (N * 2) + [(0, 2 * np.pi)] * N
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=30, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=False
        )
    except Exception:
        result = None

    if time.time() > deadline:
        # Fallback
        centers = []
        for i in range(N):
            x = (i % 3) * 2.2
            y = (i // 3) * 1.9
            centers.append([x, y, 0.0])
        verts_list = [get_hex_verts(c[0], c[1], c[2]) for c in centers]
        all_v = np.vstack(verts_list)
        S = calc_enclosing_hex_side(all_v)
        return centers, (0.0, 0.0), S, 0.0

    if result is not None and result.fun < 1000000:
        best_params = result.x
    else:
        # Fallback to simple lattice if DE failed
        centers = []
        for i in range(N):
            x = (i % 3) * 2.2
            y = (i // 3) * 1.9
            centers.append([x, y, 0.0])
        verts_list = [get_hex_verts(c[0], c[1], c[2]) for c in centers]
        all_v = np.vstack(verts_list)
        S = calc_enclosing_hex_side(all_v)
        return centers, (0.0, 0.0), S, 0.0

    positions = best_params[:N*2].reshape(N, 2)
    rotations = best_params[N*2:]
    
    all_verts_list = []
    for i in range(N):
        v = get_hex_verts(positions[i, 0], positions[i, 1], rotations[i])
        all_verts_list.append(v)
    all_vertices = np.vstack(all_verts_list)
    centroid = np.mean(all_vertices, axis=0)
    
    shifted_positions = positions - centroid
    
    shifted_verts_list = []
    for i in range(N):
        v = get_hex_verts(shifted_positions[i, 0], shifted_positions[i, 1], rotations[i])
        shifted_verts_list.append(v)
    shifted_vertices = np.vstack(shifted_verts_list)
    
    S = calc_enclosing_hex_side(shifted_vertices)
    
    inner = []
    for i in range(N):
        inner.append([shifted_positions[i, 0], shifted_positions[i, 1], np.rad2deg(rotations[i])])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
