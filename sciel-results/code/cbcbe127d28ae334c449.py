import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
_TIME_DEADLINE = START_TIME + TIME_LIMIT - SAFETY_MARGIN

def _get_hex_vertices(cx, cy, s, rot_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(rot_deg)
    vs = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        vs.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return np.array(vs)

def construct():
    if time.time() > _TIME_DEADLINE:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: Hexagonal lattice
    # 11 hexagons: 1 center, 6 ring, 4 outer
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    init_positions = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y, 0.0]) # x, y, angle_rad
        
    init_positions = np.array(init_positions)
    
    # Variables: [outer_rot, x1, y1, a1, x2, y2, a2, ..., x11, y11, a11]
    # We fix outer center at (0,0)
    
    def objective(params):
        if time.time() > _TIME_DEADLINE:
            return 1e10
        
        outer_rot = params[0]
        # Outer normals
        angles = outer_rot + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
        
        # Collect all vertices of all inner hexagons
        # To be efficient, we can compute the max projection directly
        # For each inner hex i, and each vertex j, and each normal k:
        # proj = normals[k] dot (vertex_ij - 0)
        
        # We want max of all these projections.
        # Let's collect all vertices into a single array (66, 2)
        all_vertices = np.empty((N * 6, 2))
        
        for i in range(N):
            x = params[1 + i*3]
            y = params[2 + i*3]
            a = params[3 + i*3]
            
            # Vertices of hex i
            v_angles = a + np.arange(6) * math.pi / 3.0
            vx = x + np.cos(v_angles)
            vy = y + np.sin(v_angles)
            
            all_vertices[i*6:(i+1)*6, 0] = vx
            all_vertices[i*6:(i+1)*6, 1] = vy
            
        # Projections: (66, 6)
        projs = all_vertices @ normals.T
        
        # We need max of projs.
        # Use log-sum-exp for smoothness
        temp = 0.1
        max_val = np.max(projs)
        # If max_val is huge, we are far out
        shifted = (projs - max_val) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        max_proj_smooth = temp * log_sum
        
        # S = (D + APOTHEM) / APOTHEM  where D is the max distance from center to face?
        # No, D is the max projection. The apothem of the outer hex is A_out.
        # Condition: max_proj <= A_out
        # A_out = S * APOTHEM
        # So S = max_proj / APOTHEM
        # Wait, the formula in the previous code was S = (D + APOTHEM) / APOTHEM.
        # Let's re-verify.
        # If the inner hexagon is centered at 0, its vertices are at distance 1 from center.
        # The max projection of a unit hexagon on a normal is 1 (if aligned with vertex) or sqrt(3)/2 (if aligned with face).
        # Actually, the "radius" of a hexagon in direction n is the max projection of its vertices.
        # For a unit hexagon, the max projection is 1.
        # If we place it at 0, the outer hexagon must have apothem >= 1.
        # So S = 1 / APOTHEM = 2/sqrt(3) ~ 1.154.
        # The previous code had S = (D + APOTHEM) / APOTHEM.
        # If D is the max projection of the *centers*? No, D was max projection of points.
        # In the previous code, `pts` were the centers.
        # `projs = pts_centered @ normals.T`
        # `max_proj = np.max(np.abs(projs))`
        # `S = (max_proj + APOTHEM) / APOTHEM`
        # This implies that the distance from center to the *face* of the inner hexagon is considered?
        # No, if `pts` are centers, the distance from outer center to inner center is `r`.
        # The inner hexagon extends by `APOTHEM` in the direction of the normal?
        # No, the inner hexagon extends by `1` (vertex) or `APOTHEM` (face).
        # The worst case extension is `1` (if a vertex points outward).
        # So the required apothem of the outer hex is `max_dist_center_to_vertex`.
        # If we use centers, we are underestimating the required size if we just add APOTHEM.
        # Actually, if the inner hexagon is rotated, its "radius" in direction n varies.
        # The previous code assumed all inner hexagons are parallel and used a bounding box of centers + APOTHEM.
        # This is only correct if the hexagons are "flat-top" aligned such that the max extension is APOTHEM?
        # No, a hexagon of side 1 has max distance from center to vertex 1.
        # If the normal points to a vertex, the extension is 1.
        # If the normal points to a face midpoint, the extension is APOTHEM.
        # So the required outer apothem is `max_i ( dist(C, c_i) + r_i(n) )`?
        # No, it is `max_i ( max_{v in H_i} n . (v - C) )`.
        
        # So my new objective `S = max_proj_smooth / APOTHEM` is correct for the vertices.
        # Let's stick with this.
        
        S = max_proj_smooth / APOTHEM
        
        return S

    # Initial params
    init_params = np.zeros(1 + N*3)
    init_params[0] = 0.0 # outer rot
    for i in range(N):
        init_params[1 + i*3] = init_positions[i, 0]
        init_params[2 + i*3] = init_positions[i, 1]
        init_params[3 + i*3] = 0.0 # inner rot

    best_S = float('inf')
    best_params = init_params.copy()

    # Optimization
    try:
        # Use Nelder-Mead for robustness with non-smooth objective (even with LSE, it's still tricky)
        # Or use L-BFGS-B with the smooth objective
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass

    # Try L-BFGS-B with a smoother approximation (higher temperature)
    def smooth_objective(params, temp=0.5):
        if time.time() > _TIME_DEADLINE:
            return 1e10
        outer_rot = params[0]
        angles = outer_rot + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        all_vertices = np.empty((N * 6, 2))
        for i in range(N):
            x = params[1 + i*3]
            y = params[2 + i*3]
            a = params[3 + i*3]
            v_angles = a + np.arange(6) * math.pi / 3.0
            vx = x + np.cos(v_angles)
            vy = y + np.sin(v_angles)
            all_vertices[i*6:(i+1)*6, 0] = vx
            all_vertices[i*6:(i+1)*6, 1] = vy
            
        projs = all_vertices @ normals.T
        max_val = np.max(projs)
        shifted = (projs - max_val) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        max_proj_smooth = temp * log_sum
        S = max_proj_smooth / APOTHEM
        return S

    try:
        if time.time() < _TIME_DEADLINE:
            res2 = minimize(lambda p: smooth_objective(p, 0.5), best_params, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12})
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
    except Exception:
        pass

    # Final polish with Nelder-Mead on the exact (non-smooth) objective
    def exact_objective(params):
        if time.time() > _TIME_DEADLINE:
            return 1e10
        outer_rot = params[0]
        angles = outer_rot + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = -1e10
        for i in range(N):
            x = params[1 + i*3]
            y = params[2 + i*3]
            a = params[3 + i*3]
            v_angles = a + np.arange(6) * math.pi / 3.0
            vx = x + np.cos(v_angles)
            vy = y + np.sin(v_angles)
            
            # Project these 6 vertices onto 6 normals
            # (6, 1) dot (1, 6) -> (6, 6)
            v_matrix = np.column_stack([vx, vy])
            projs = v_matrix @ normals.T
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
                
        S = max_proj / APOTHEM
        return S

    try:
        if time.time() < _TIME_DEADLINE:
            res3 = minimize(exact_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res3.fun < best_S:
                best_S = res3.fun
                best_params = res3.x
    except Exception:
        pass

    # Extract results
    outer_rot_rad = best_params[0]
    inner = []
    for i in range(N):
        x = best_params[1 + i*3]
        y = best_params[2 + i*3]
        a = best_params[3 + i*3]
        inner.append([float(x), float(y), float(math.degrees(a))])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = float(math.degrees(outer_rot_rad))
    
    return inner, center, side, rotation
