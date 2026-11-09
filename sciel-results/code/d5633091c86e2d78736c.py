import numpy as np
import time
from scipy.optimize import minimize

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    """Generate vertices of a regular hexagon."""
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _min_dist_point_to_hex(p, hex_verts):
    """
    Vectorized minimum distance from point p to hex_verts.
    Returns negative if p is inside.
    """
    # p: (2,), hex_verts: (6, 2)
    edges = np.roll(hex_verts, -1, axis=0) - hex_verts
    ap = p - hex_verts
    cross = edges[:, 0] * ap[:, 1] - edges[:, 1] * ap[:, 0]
    
    # Check if inside
    if np.all(cross >= -1e-9):
        # Inside: return -min distance to boundary
        ab_lens_sq = np.sum(edges**2, axis=1)
        t = np.sum(ap * edges, axis=1) / ab_lens_sq
        t = np.clip(t, 0, 1)
        closest = hex_verts + t[:, np.newaxis] * edges
        dists = np.linalg.norm(p - closest, axis=1)
        return -np.min(dists)
    else:
        # Outside: return min distance to boundary
        ab_lens_sq = np.sum(edges**2, axis=1)
        t = np.sum(ap * edges, axis=1) / ab_lens_sq
        t = np.clip(t, 0, 1)
        closest = hex_verts + t[:, np.newaxis] * edges
        dists = np.linalg.norm(p - closest, axis=1)
        return np.min(dists)

def _min_dist_between_hexes(v1, v2):
    """
    Minimum distance between two hexagons (vertices arrays).
    Returns negative if overlapping.
    """
    min_d = np.inf
    for p in v1:
        d = _min_dist_point_to_hex(p, v2)
        if d < min_d:
            min_d = d
            if min_d < 0:
                return min_d
    for p in v2:
        d = _min_dist_point_to_hex(p, v1)
        if d < min_d:
            min_d = d
            if min_d < 0:
                return min_d
    return min_d

def _packing_clearance(positions, angles, S):
    """
    Calculate minimum clearance. Negative if invalid.
    """
    min_clear = np.inf
    outer_verts = _hex_vertices(0.0, 0.0, 0.0, S)
    
    for i in range(N):
        v_i = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        
        # Check against outer
        for p in v_i:
            d = _min_dist_point_to_hex(p, outer_verts)
            if d < min_clear:
                min_clear = d
        
        # Check against others
        for j in range(i + 1, N):
            v_j = _hex_vertices(positions[j, 0], positions[j, 1], angles[j], 1.0)
            d_pair = _min_dist_between_hexes(v_i, v_j)
            if d_pair < min_clear:
                min_clear = d_pair
                
    return min_clear

def _merit(params):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., x11, y11, a11, S]
    """
    positions = params[:3*N].reshape(N, 3)
    S = params[-1]
    if S < 3.0: S = 3.0
    
    clear = _packing_clearance(positions[:, :2], positions[:, 2], S)
    if clear < 0:
        return S + 1000.0 * (clear * clear)
    else:
        return S

def construct():
    start_time = time.time()
    best_S = np.inf
    best_params = None
    
    # Initial guesses
    # 1. Hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    positions = np.zeros((N, 2))
    for i, (q, r) in enumerate(axial_coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        positions[i, 0] = x
        positions[i, 1] = y
    
    max_r = np.max(np.linalg.norm(positions, axis=1))
    S_init = (max_r + 1.5) * 2.0 / np.sqrt(3.0)
    
    # Prepare initial params
    init_params = np.zeros(3*N + 1)
    init_params[0:3*N:3] = positions[:, 0]
    init_params[1:3*N:3] = positions[:, 1]
    init_params[2:3*N:3] = 0.0
    init_params[-1] = S_init
    
    # Run Nelder-Mead from this start
    # And maybe a few random restarts
    
    rng = np.random.default_rng(42)
    
    starts = [init_params]
    # Add random starts
    for _ in range(5):
        p = init_params.copy()
        # Random perturb positions
        p[0:3*N:3] += rng.normal(0, 0.5, N)
        p[1:3*N:3] += rng.normal(0, 0.5, N)
        p[2:3*N:3] = rng.uniform(0, 2*np.pi, N)
        # Random S
        p[-1] = S_init * rng.uniform(0.8, 1.2)
        starts.append(p)
        
    for p0 in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            res = minimize(_merit, p0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_S:
                # Check if valid
                pos = res.x[:3*N].reshape(N, 3)
                S_res = res.x[-1]
                clear = _packing_clearance(pos[:, :2], pos[:, 2], S_res)
                if clear >= -1e-5: # Allow tiny numerical error
                    if S_res < best_S:
                        best_S = S_res
                        best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)], (0.0, 0.0), 50.0, 0.0
        
    pos = best_params[:3*N].reshape(N, 3)
    S_final = best_params[-1]
    
    inner = []
    for i in range(N):
        inner.append([pos[i, 0], pos[i, 1], np.degrees(pos[i, 2])])
    
    return inner, (0.0, 0.0), S_final, 0.0
