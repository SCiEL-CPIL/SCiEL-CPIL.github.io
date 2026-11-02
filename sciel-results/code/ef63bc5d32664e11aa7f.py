import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

def _hex_vertices(cx, cy, s, rot):
    """Return 6 vertices of a regular hexagon (flat-top convention)."""
    v = []
    for k in range(6):
        a = rot + k * math.pi / 3.0
        v.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return np.array(v)

def _support(pts, nx, ny):
    """Max dot product of pts with normal (nx,ny)."""
    return float(np.max(pts[:, 0] * nx + pts[:, 1] * ny))

def _min_enclosing_hexagon(pts, rot):
    """Find minimal center and apothem D for a hexagon with fixed rotation rot."""
    normals = np.array([
        [math.cos(rot + math.pi/6.0 + k * math.pi/3.0),
         math.sin(rot + math.pi/6.0 + k * math.pi/3.0)]
        for k in range(6)
    ])
    # For each face k, constraint: pts @ n_k <= c_k and -pts @ n_k <= c_k
    # So c_k >= max(pts @ n_k) and c_k >= max(-pts @ n_k)
    # We want to minimize D = max_k c_k by choosing center offset (ox, oy)
    # pts_shifted = pts + (ox, oy)
    # c_k >= max((pts + off) @ n_k) = max(pts @ n_k) + off @ n_k
    # c_k >= max(-(pts + off) @ n_k) = -min(pts @ n_k) - off @ n_k
    # So D >= max_k [max(pts @ n_k) + off @ n_k, -min(pts @ n_k) - off @ n_k]
    
    max_dots = pts @ normals.T  # (n_pts, 6)
    min_dots = pts @ normals.T  # same
    max_vals = max_dots.max(axis=0)  # (6,)
    min_vals = min_dots.min(axis=0)  # (6,)
    
    # D(off) = max_k [max_vals[k] + off @ n_k, -min_vals[k] - off @ n_k]
    # = max_k [max_vals[k] + off @ n_k, -min_vals[k] - off @ n_k]
    
    def D_of_off(off):
        terms1 = max_vals + off @ normals.T  # (6,)
        terms2 = -min_vals - off @ normals.T  # (6,)
        return float(np.max(np.maximum(terms1, terms2)))
    
    # Optimize off = (ox, oy) using Nelder-Mead
    # Initial guess: centroid
    off0 = np.array([0.0, 0.0])
    
    best_D = float('inf')
    best_off = off0.copy()
    
    # Multi-start for the offset optimization
    starts = [
        off0,
        np.array([0.1, 0.1]), np.array([-0.1, 0.1]),
        np.array([0.1, -0.1]), np.array([-0.1, -0.1]),
        np.array([0.2, 0.0]), np.array([-0.2, 0.0]),
        np.array([0.0, 0.2]), np.array([0.0, -0.2]),
    ]
    
    for s in starts:
        try:
            res = minimize(D_of_off, s, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_off = res.x.copy()
        except Exception:
            pass
    
    center = -best_off  # Since we shifted pts by off, center = -off
    return best_D, center

def _full_objective(params, pts):
    """Full objective: optimize center (ox,oy) and rotation rot."""
    ox, oy, rot = params
    # Normalize rotation to [0, pi/3)
    rot = ((rot % (math.pi/3.0)) + math.pi/3.0) % (math.pi/3.0)
    
    normals = np.array([
        [math.cos(rot + math.pi/6.0 + k * math.pi/3.0),
         math.sin(rot + math.pi/6.0 + k * math.pi/3.0)]
        for k in range(6)
    ])
    
    pts_shifted = pts + np.array([ox, oy])
    dots = pts_shifted @ normals.T
    max_vals = dots.max(axis=0)
    min_vals = dots.min(axis=0)
    
    D = float(np.max(np.maximum(max_vals, -min_vals)))
    S = (D + APOTHEM) / APOTHEM
    return S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse candidate configurations using axial coordinates
    # Each config is a list of (q, r) axial positions
    configs = []
    
    # Classic honeycomb cluster (7 hexagons)
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ])
    
    # 11 hexagons: 7 + 4 more in various patterns
    # Pattern 1: extend one direction
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (2,1), (2,2), (2,-1)
    ])
    
    # Pattern 2: diamond extension
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2)
    ])
    
    # Pattern 3: two opposite extensions
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (1,1), (-1,-1)
    ])
    
    # Pattern 4: L-shape
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (2,1), (3,1)
    ])
    
    # Pattern 5: compact 3x3 minus corners
    configs.append([
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (1,-1), (1,3)
    ])
    
    # Pattern 6: cross shape
    configs.append([
        (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2)
    ])
    
    # Pattern 7: elongated
    configs.append([
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (1,2), (2,2), (1,-1)
    ])
    
    # Pattern 8: asymmetric
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,1), (1,2), (-1,2)
    ])
    
    # Pattern 9: another variation
    configs.append([
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2),
        (1,3), (2,3)
    ])
    
    # Pattern 10: star-like
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2)
    ])
    
    # Pattern 11: another
    configs.append([
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (0,3), (1,3)
    ])
    
    # Pattern 12: wide
    configs.append([
        (0,0), (1,0), (2,0), (3,0), (4,0),
        (0,1), (1,1), (2,1), (3,1),
        (1,2), (2,2)
    ])
    
    # Pattern 13: tall
    configs.append([
        (0,0), (1,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2),
        (0,3), (1,3), (2,3), (1,4)
    ])
    
    # Pattern 14: mixed
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (1,1)
    ])
    
    # Pattern 15: another
    configs.append([
        (0,0), (1,0), (2,0), (3,0),
        (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2)
    ])
    
    # Pattern 16: compact
    configs.append([
        (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (1,-2)
    ])
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    best_S = float('inf')
    best_pts = None
    best_center = None
    best_rot = 0.0

    for cfg in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(cfg)
        if len(pts) != N:
            continue
        
        # Optimize over rotation and center
        # First, try a grid of rotations
        best_local_S = float('inf')
        best_local_center = None
        best_local_rot = 0.0
        
        # Coarse grid over rotation
        for i in range(19):
            rot = i * (math.pi / 3.0) / 18.0
            D, center = _min_enclosing_hexagon(pts, rot)
            S = (D + APOTHEM) / APOTHEM
            if S < best_local_S:
                best_local_S = S
                best_local_center = center
                best_local_rot = rot
        
        # Refine with full optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Multi-start full optimization
                starts = [
                    [best_local_center[0], best_local_center[1], best_local_rot],
                    [best_local_center[0], best_local_center[1], best_local_rot + 0.05],
                    [best_local_center[0], best_local_center[1], best_local_rot - 0.05],
                    [best_local_center[0] + 0.1, best_local_center[1], best_local_rot],
                    [best_local_center[0] - 0.1, best_local_center[1], best_local_rot],
                    [best_local_center[0], best_local_center[1] + 0.1, best_local_rot],
                    [best_local_center[0], best_local_center[1] - 0.1, best_local_rot],
                ]
                
                for st in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(_full_objective, st, args=(pts,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                        if res.fun < best_local_S:
                            best_local_S = res.fun
                            best_local_center = (res.x[0], res.x[1])
                            best_local_rot = ((res.x[2] % (math.pi/3.0)) + math.pi/3.0) % (math.pi/3.0)
                    except Exception:
                        pass
        except Exception:
            pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_pts = pts
            best_center = best_local_center
            best_rot = best_local_rot

    # Fallback
    if best_pts is None or best_S == float('inf'):
        pts = axial_to_cartesian(configs[0])
        D, center = _min_enclosing_hexagon(pts, 0.0)
        best_S = (D + APOTHEM) / APOTHEM
        best_pts = pts
        best_center = center
        best_rot = 0.0

    # Compute final side length with a tiny safety margin
    # The hexagons are at best_pts, outer hexagon center at best_center, rotation best_rot
    # Side length S = (D + APOTHEM) / APOTHEM where D is the max projection
    # Add tiny margin to ensure no overlap
    margin = 1e-10
    best_S_final = best_S + margin / APOTHEM

    # Return in the required format
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(math.degrees(best_rot))])
    
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S_final)
    rotation = float(math.degrees(best_rot))
    
    return inner, center, side, rotation
