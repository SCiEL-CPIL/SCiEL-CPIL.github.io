import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # Safety margin below 150s

def get_hex_vertices_batch(xys_rots):
    """
    Batch compute vertices for all hexagons. xys_rots is (N, 3) array.
    Returns (N, 6, 2) array.
    """
    N_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    
    return np.stack((x, y), axis=-1)

def sat_overlap_batch(vertices):
    """
    Vectorized SAT overlap check for all pairs of hexagons.
    vertices: (N, 6, 2)
    Returns penalty (sum of squared penetration depths) and has_overlap boolean.
    """
    N_pts = len(vertices)
    if N_pts < 2:
        return 0.0, False
        
    v_next = np.roll(vertices, -1, axis=1)
    edges = v_next - vertices 
    normals = np.stack((-edges[:,:,1], edges[:,:,0]), axis=-1) 
    norms = np.linalg.norm(normals, axis=2, keepdims=True) 
    norms[norms == 0] = 1e-9
    normals = normals / norms 
    
    has_overlap = False
    penalty = 0.0
    
    for i in range(N_pts):
        v1 = vertices[i]
        n1 = normals[i]
        
        for j in range(i+1, N_pts):
            v2 = vertices[j]
            n2 = normals[j]
            
            all_axes = np.vstack((n1, n2))
            proj1 = v1 @ all_axes.T
            proj2 = v2 @ all_axes.T
            
            min1 = np.min(proj1, axis=1)
            max1 = np.max(proj1, axis=1)
            min2 = np.min(proj2, axis=1)
            max2 = np.max(proj2, axis=1)
            
            sep1 = max1 < min2
            sep2 = max2 < min1
            separated = np.any(sep1 | sep2)
            
            if separated:
                continue
            
            overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
            min_overlap = np.min(overlaps)
            
            if min_overlap > 1e-9:
                has_overlap = True
                penalty += min_overlap ** 2
                
    return penalty, has_overlap

def outer_hex_side_batch(vertices, outer_params):
    """
    Calculate the minimum side length of a regular hexagon enclosing all vertices.
    outer_params: [cx, cy, rot_deg]
    """
    cx, cy, rot_deg = outer_params
    rot_rad = np.radians(rot_deg)
    
    base_normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normal_angles = base_normal_angles + rot_rad
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    pts = vertices.reshape(-1, 2)
    pts_rel = pts - np.array([cx, cy])
    
    proj = pts_rel @ normals.T
    max_proj = np.max(np.abs(proj))
    
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    def objective(params):
        if time.time() - start_time > DEADLINE - 2:
            return 1e6
            
        inner_params = params[:3*N].reshape(N, 3)
        outer_params = params[3*N:3*N+3]
        
        vertices = get_hex_vertices_batch(inner_params)
        
        penalty, has_overlap = sat_overlap_batch(vertices)
        
        L = outer_hex_side_batch(vertices, outer_params)
        
        return L + 1000.0 * penalty

    def make_lattice_config(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_ring_config(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    best_L = float('inf')
    best_params = None

    # Initial guess for DE bounds
    # Positions can be roughly within [-5, 5], angles [0, 360]
    bounds = []
    for _ in range(N):
        bounds.append([(-5, 5), (-5, 5), (0, 360)])
    bounds.append([(-1, 1), (-1, 1), (0, 360)]) # Outer center and rotation

    # 1. Global Search with Differential Evolution
    de_start_time = time.time()
    de_time_limit = 60.0 # Give 60s to DE
    
    try:
        # Use a seed for reproducibility if needed, but random is fine
        # strategy='best1bin' is standard
        res_de = differential_evolution(
            objective, 
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False, # We will polish manually
            init='latinhypercube',
            # Callback to check time
            callback=lambda xk, convergence: (time.time() - de_start_time) > de_time_limit
        )
        
        if res_de.success or res_de.fun < best_L:
            # Check validity of DE result
            inner_params = res_de.x[:3*N].reshape(N, 3)
            outer_params = res_de.x[3*N:3*N+3]
            vertices = get_hex_vertices_batch(inner_params)
            penalty, has_overlap = sat_overlap_batch(vertices)
            
            if not has_overlap:
                L = outer_hex_side_batch(vertices, outer_params)
                if L < best_L:
                    best_L = L
                    best_params = res_de.x.copy()
            else:
                # Even if overlapping, keep as a starting point for local refinement
                # But we need a valid best_params for return if all else fails
                # So we store it separately or just use it for refinement
                pass

    except Exception:
        pass

    # If DE didn't find a valid solution, try a few local refinements from initial configs
    if best_params is None:
        configs = [
            make_lattice_config(4, 3, gap=0.1),
            make_lattice_config(3, 4, gap=0.1),
            make_ring_config(gap=0.1)
        ]
        for cfg in configs:
            if time_left() < 10:
                break
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
            try:
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True}
                )
                inner_params = res.x[:3*N].reshape(N, 3)
                outer_params = res.x[3*N:3*N+3]
                vertices = get_hex_vertices_batch(inner_params)
                penalty, has_overlap = sat_overlap_batch(vertices)
                
                if not has_overlap:
                    L = outer_hex_side_batch(vertices, outer_params)
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
            except:
                pass

    # 2. Local Refinement with Nelder-Mead
    # We want to polish the best candidate found so far
    # If DE found a candidate (even if slightly overlapping due to numerical issues), use it.
    # If DE found a valid one, use it.
    # If neither, and best_params is still None, we are in trouble, but we have fallbacks.
    
    # Let's assume best_params is set to a good candidate from DE or local search.
    # If best_params is None, we must generate a fallback.
    if best_params is None:
        # Fallback: simple grid
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        inner_arr = np.array(inner)
        vertices = get_hex_vertices_batch(inner_arr)
        L_val = outer_hex_side_batch(vertices, [0.0, 0.0, 0.0])
        return inner, [0.0, 0.0], float(L_val), 0.0

    # Refine the best_params
    if time_left() > 15:
        try:
            # Use a smaller step size for refinement
            res_refine = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            inner_params = res_refine.x[:3*N].reshape(N, 3)
            outer_params = res_refine.x[3*N:3*N+3]
            vertices = get_hex_vertices_batch(inner_params)
            penalty, has_overlap = sat_overlap_batch(vertices)
            
            if not has_overlap:
                L = outer_hex_side_batch(vertices, outer_params)
                if L < best_L:
                    best_L = L
                    best_params = res_refine.x.copy()
        except Exception:
            pass

    # Final check and extraction
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_params = best_params[3*N:3*N+3]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [float(outer_params[0]), float(outer_params[1])]
    side = float(best_L)
    angle = float(outer_params[2])
    
    return inner, center, side, angle
