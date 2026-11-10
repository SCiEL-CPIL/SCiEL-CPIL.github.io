import numpy as np
from scipy.optimize import minimize
import itertools

def hexagon_vertices(cx, cy, side, angle_deg):
    """
    Calculate vertices of a regular hexagon.
    Vertices are at cx + side * cos(angle + 60*i), cy + side * sin(angle + 60*i)
    """
    angles = np.deg2rad(angle_deg + np.arange(6) * 60)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def sat_gap(A, B, normals):
    """
    Calculate the maximum gap between two polygons A and B along the given normals.
    If max gap > 0, they are separated.
    If max gap < 0, they overlap.
    Returns the max gap.
    """
    proj_A = np.min(A @ normals.T, axis=0)
    proj_B = np.max(B @ normals.T, axis=0)
    gaps = proj_A - proj_B
    return np.max(gaps)

def get_normals(vertices):
    """
    Get outward edge normals for a convex polygon given in CCW order.
    """
    edges = np.roll(vertices, -1, axis=0) - vertices
    dx = edges[:, 0]
    dy = edges[:, 1]
    normals = np.column_stack((dy, -dx))
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    return normals

def calculate_penalty(params, s_outer):
    """
    params: flat array of 36 floats (x, y, angle) * 12
    s_outer: side length of outer hexagon
    """
    n_inner = 12
    penalty = 0.0
    
    # Reshape params
    inner_params = params.reshape(n_inner, 3)
    
    # Generate vertices and normals for inner hexagons
    inner_verts = []
    inner_normals = []
    for i in range(n_inner):
        x, y, ang = inner_params[i]
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
        inner_normals.append(get_normals(v))
        
    # Outer hexagon vertices and normals
    outer_v = hexagon_vertices(0, 0, s_outer, 0)
    outer_normals = get_normals(outer_v)
    
    # 1. Inner-Inner Overlaps
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            A = inner_verts[i]
            B = inner_verts[j]
            # Use normals from both A and B
            normals = np.vstack((inner_normals[i], inner_normals[j]))
            gap = sat_gap(A, B, normals)
            if gap < 0:
                # Overlap depth is -gap
                penalty += (-gap) ** 2
                
    # 2. Inner-Outer Overlaps
    # Check if any inner vertex is outside the outer hexagon
    for i in range(n_inner):
        v = inner_verts[i]
        for k in range(6):
            pt = v[k]
            for m in range(6):
                p1 = outer_v[m]
                n = outer_normals[m]
                dist = np.dot(pt - p1, n)
                if dist > 0:
                    penalty += dist ** 2
                    
    return penalty

def hexagon_packing_12():
    n = 12
    
    # Initialization: Flower pattern (7 hexes) + 5 more
    # Center: (0,0)
    # Ring 1: 6 hexes at distance sqrt(3)
    # Ring 2: 5 hexes placed in the "valleys" of Ring 1, further out.
    # Distance for Ring 2 centers: 2 * sqrt(3) approx?
    # Let's use a scaled version of a larger cluster.
    
    # Base flower
    d = np.sqrt(3)
    init_params = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [-d, 0, 0],
        [0, d*1.5, 0], # 60 deg rotated? No, hexagon grid coords
        [0, -d*1.5, 0],
        [d/2, d*1.5, 0],
        [-d/2, d*1.5, 0],
        # Add 5 more. Place them at the next ring positions that are "empty"
        # In a full ring 2, there are 12 positions. We need 5.
        # Positions at distance 2*d?
        # Let's just place them roughly and let the optimizer fix it.
        [2*d, 0, 0],
        [0, 2*d*1.5, 0],
        [-2*d, 0, 0],
        [d, 2*d*1.5, 0],
        [-d, 2*d*1.5, 0]
    ]).flatten()

    # Binary search for s
    s_low = np.sqrt(12) * 1.01
    s_high = 8.0
    
    best_s = s_high
    best_params = init_params
    best_penalty = calculate_penalty(best_params, best_s)
    
    def try_pack(s_target, num_restarts=5):
        best_local_params = None
        min_pen = float('inf')
        
        # Multiple initializations
        inits = [
            init_params.copy(),
            init_params + np.random.normal(0, 0.1, size=init_params.shape),
            # Rotated init
            np.array([[x, y, (a+30)%360] for x,y,a in init_params.reshape(12,3)]).flatten(),
            # Scaled init (tighter)
            np.array([[x*0.9, y*0.9, a] for x,y,a in init_params.reshape(12,3)]).flatten(),
            # Scaled init (looser)
            np.array([[x*1.1, y*1.1, a] for x,y,a in init_params.reshape(12,3)]).flatten()
        ]
        
        for r in range(num_restarts):
            p0 = inits[r % len(inits)]
            # Add small jitter to avoid symmetry breaking issues
            p0 = p0 + np.random.normal(0, 0.01, size=p0.shape)
            
            bounds = []
            for i in range(n):
                bounds.append((-s_target, s_target))
                bounds.append((-s_target, s_target))
                bounds.append((0, 360))
                
            res = minimize(calculate_penalty, p0, args=(s_target,), method='Powell', 
                           bounds=bounds, options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < min_pen:
                min_pen = res.fun
                best_local_params = res.x
                
        if min_pen < 1e-6:
            return True, best_local_params
        else:
            return False, best_local_params

    # Binary search
    for _ in range(10): # 10 iterations of binary search
        s_mid = (s_low + s_high) / 2
        valid, params = try_pack(s_mid, num_restarts=3)
        if valid:
            best_s = s_mid
            best_params = params
            s_low = s_mid
        else:
            s_high = s_mid
            
    # Final optimization with best_s
    valid, best_params = try_pack(best_s, num_restarts=10)
    if not valid:
        # Fallback
        best_s = 8.0
        best_params = init_params

    # Construct return values
    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
