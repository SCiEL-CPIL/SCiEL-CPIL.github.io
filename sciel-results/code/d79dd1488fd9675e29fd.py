import numpy as np
import itertools
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_separated(v1, v2, tol=1e-6):
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _is_valid(cfg):
    for i in range(11):
        for j in range(i+1, 11):
            v1 = _hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2])
            v2 = _hex_vertices(cfg[j,0], cfg[j,1], 1.0, cfg[j,2])
            if not _hex_separated(v1, v2):
                return False
    return True

def _compute_outer_side_fast(inner_data, outer_angle=0.0):
    a = np.radians(outer_angle)
    all_verts = np.array([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_data])
    
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)

def _check_containment(inner_data, outer_center, outer_angle, side):
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for hex_row in inner_data:
        v = _hex_vertices(hex_row[0], hex_row[1], 1.0, hex_row[2])
        for p in v:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True

def _generate_candidates():
    # Axial coordinates for the 19-hex cluster (side 3)
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes
    
    # Directions in axial coords (q, r)
    # Flat top:
    # E: (1, 0)
    # NE: (0, -1) ? No, let's use standard cube or axial.
    # Axial: x = 1.5 q, y = sqrt(3) (r + 0.5 q)
    # Neighbors:
    # (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    
    hexes = set()
    # Generate all hexes in a 5x5x5 cube (axial q, r, s=-q-r)
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if -2 <= s <= 2:
                hexes.add((q, r))
                
    # Identify Center, Ring 1, Ring 2
    center = (0, 0)
    ring1 = []
    ring2 = []
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    for q, r in hexes:
        dist = (abs(q) + abs(r) + abs(-q-r)) / 2.0
        if dist == 0:
            pass
        elif dist == 1:
            ring1.append((q, r))
        elif dist == 2:
            ring2.append((q, r))
            
    # We want subsets of size 11 containing Center + Ring1 (7 hexes) + 4 from Ring2
    # This assumes the optimal packing is a "7-core" + 4 satellites.
    
    candidates = []
    for combo in itertools.combinations(ring2, 4):
        subset = [center] + ring1 + list(combo)
        # Convert to Cartesian
        verts = []
        for q, r in subset:
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            verts.append((x, y))
        candidates.append(np.array(verts))
        
    return candidates

def _optimize_cluster(cluster_verts, time_limit=5.0):
    # cluster_verts: (11, 2) array of centers
    # We need to place 11 hexagons at these centers.
    # The hexagons have angle 0 relative to the cluster frame.
    # We rotate the whole cluster by theta, translate by t.
    
    # Get all vertices of the 11 hexagons in the cluster frame
    all_verts = []
    for cx, cy in cluster_verts:
        v = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_theta = 0.0
    best_t = np.array([0.0, 0.0])
    
    # Coarse search for theta
    for theta_deg in np.arange(0, 60, 0.5):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rotated_verts = all_verts @ rot.T
        
        # Project onto 6 normals
        # Normals of outer hexagon (fixed at 0 angle)
        # The outer hexagon has vertices at angles 0, 60, ...
        # Its edge normals are at 30, 90, ...
        # Wait, the definition of outer hexagon:
        # vertices (cx + s*cos(a + 60i), ...)
        # If a=0, vertices are at 0, 60, ...
        # Edge between 0 and 60 is at 30 deg. Normal is 30+90=120?
        # Let's check the containment check logic.
        # The check uses cross product with edge vectors.
        # For a hexagon centered at 0 with angle 0:
        # v0=(s,0), v1=(s/2, s*sqrt(3)/2).
        # Edge v0->v1. Vector (-s/2, s*sqrt(3)/2).
        # Normal (perpendicular, inward): (s*sqrt(3)/2, s/2). Angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        normals = []
        for i in range(6):
            ang = np.radians(30 + 60 * i)
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        projs = rotated_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        
        # We want to find t such that max_k max(max_projs[k] + t.n[k], -min_projs[k] - t.n[k]) is minimized.
        # Let f(t) = max_k (max_projs[k] + t.n[k], -min_projs[k] - t.n[k])
        # We can use a simple local search for t.
        
        def objective(t):
            tx, ty = t
            t_vec = np.array([tx, ty])
            shifts = t_vec @ normals.T
            vals = np.maximum(max_projs + shifts, -min_projs - shifts)
            return np.max(vals)
        
        # Initial guess for t: center of mass of projections?
        # Or just 0.
        res = minimize(objective, [0, 0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8})
        
        R = res.fun
        side = 2.0 * R / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_theta = theta_deg
            best_t = res.x
            
    # Refine theta around best
    for theta_deg in np.arange(best_theta - 0.5, best_theta + 0.5, 0.05):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rotated_verts = all_verts @ rot.T
        
        normals = []
        for i in range(6):
            ang = np.radians(30 + 60 * i)
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        projs = rotated_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        min_projs = np.min(projs, axis=0)
        
        def objective(t):
            tx, ty = t
            t_vec = np.array([tx, ty])
            shifts = t_vec @ normals.T
            vals = np.maximum(max_projs + shifts, -min_projs - shifts)
            return np.max(vals)
            
        res = minimize(objective, best_t, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10})
        
        R = res.fun
        side = 2.0 * R / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_theta = theta_deg
            best_t = res.x

    # Construct final config
    theta = np.radians(best_theta)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    
    final_centers = cluster_verts @ rot.T + best_t
    cfg = np.zeros((11, 3))
    cfg[:, :2] = final_centers
    cfg[:, 2] = best_theta # Angle of the hexagons themselves
    
    return cfg, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 615
    
    candidates = _generate_candidates()
    
    best_side = float('inf')
    best_cfg = None
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        cfg, side = _optimize_cluster(cand, time_limit=10.0)
        if side < best_side:
            best_side = side
            best_cfg = cfg
            
    if best_cfg is None:
        # Fallback
        best_cfg = np.zeros((11, 3))
        best_side = 10.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Verify and adjust side if necessary
    if not _check_containment(best_cfg, np.array([0.0, 0.0]), 0.0, best_side):
        # Increase side slightly
        for _ in range(100):
            if _check_containment(best_cfg, np.array([0.0, 0.0]), 0.0, best_side):
                break
            best_side *= 1.001
            
    return best_cfg, outer_hex_data, float(best_side)
