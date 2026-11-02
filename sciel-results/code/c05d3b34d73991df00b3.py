import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Unit hexagon vertices (flat-top)
def hex_vertices(angle):
    """Vertices of unit hexagon rotated by angle, centered at origin."""
    # Flat-top hexagon: vertices at angles 0, 60, 120, ...
    # Pointy-top would be 30, 90, 150, ...
    # For flat-top: vertices at angle + k*60 degrees
    angles = angle + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def hex_support(cx, cy, angle, normals):
    """Max projection of hexagon (centered at cx,cy, rotated by angle) onto given normals."""
    verts = hex_vertices(angle)
    pts = verts + np.array([cx, cy])
    projs = pts @ normals.T
    return np.max(projs, axis=1)

def hex_support_both(cx, cy, angle, normals):
    """Max of |projection| for hexagon onto normals."""
    verts = hex_vertices(angle)
    pts = verts + np.array([cx, cy])
    projs = pts @ normals.T
    return np.max(np.abs(projs), axis=1)

def outer_hex_normals(beta):
    """Normals to the 6 sides of outer hexagon with rotation beta."""
    # Outer hexagon side normals at beta + k*60 degrees
    angles = beta + np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def exact_S(pts_centers, angles, beta, cx, cy):
    """Exact S: find minimum outer side length containing all hexagons."""
    normals = outer_hex_normals(beta)
    # For each normal, find max projection over all hexagons
    max_projs = np.zeros(6)
    for i in range(N):
        sp = hex_support_both(pts_centers[i], 0, angles[i], normals)
        # Wait, need to account for center offset
        pass
    
    # Recompute properly
    max_projs = np.zeros(6)
    for i in range(N):
        verts = hex_vertices(angles[i])
        pts = verts + np.array([pts_centers[i][0], pts_centers[i][1]])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_projs = np.maximum(max_projs, np.max(np.abs(projs), axis=1))
    
    D = np.max(max_projs)
    S = (D + APOTHEM) / APOTHEM
    return S

def smooth_S(pts_centers, angles, beta, cx, cy, temperature=0.1):
    """Smooth approximation of S using log-sum-exp."""
    normals = outer_hex_normals(beta)
    
    # Collect all projections
    all_projs = []
    for i in range(N):
        verts = hex_vertices(angles[i])
        pts = verts + np.array([pts_centers[i][0], pts_centers[i][1]])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs.append(projs)
    
    all_projs = np.concatenate(all_projs, axis=0)
    
    # Soft max of absolute values
    abs_projs = np.abs(all_projs)
    max_val = np.max(abs_projs)
    if max_val > 1000:
        return float('inf')
    
    shifted = (abs_projs - max_val) / temperature
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    S = (D + APOTHEM) / APOTHEM
    return S

def overlap_penalty(pts_centers, angles, temperature=0.1):
    """Penalty for overlapping hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Distance between centers
            dx = pts_centers[i][0] - pts_centers[j][0]
            dy = pts_centers[i][1] - pts_centers[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Minimum distance for non-overlapping hexagons (rough)
            # Two unit hexagons need at least 2*apothem = sqrt(3) separation
            # when aligned, but less when rotated
            min_dist = 2 * APOTHEM  # Conservative estimate
            
            if dist < min_dist:
                # Smooth penalty
                overlap = min_dist - dist
                penalty += temperature * math.log(1.0 + math.exp(overlap / temperature))
    
    return penalty

def make_initial_config(config_type):
    """Generate initial positions and angles for inner hexagons."""
    centers = []
    angles = []
    
    if config_type == 0:  # Hexagonal cluster (center + 6 neighbors + 4 more)
        # Center
        centers.append((0.0, 0.0))
        angles.append(0.0)
        # 6 neighbors at distance sqrt(3)
        for k in range(6):
            ang = k * math.pi / 3.0
            cx = math.sqrt(3) * math.cos(ang)
            cy = math.sqrt(3) * math.sin(ang)
            centers.append((cx, cy))
            angles.append(0.0)
        # 4 more in second ring
        for k in range(4):
            ang = k * math.pi / 3.0 + math.pi / 6.0
            cx = 3.0 * math.cos(ang)
            cy = 3.0 * math.sin(ang)
            centers.append((cx, cy))
            angles.append(0.0)
            
    elif config_type == 1:  # Ring arrangement
        # Center
        centers.append((0.0, 0.0))
        angles.append(0.0)
        # 10 in a ring
        for k in range(10):
            ang = k * 2 * math.pi / 10.0
            r = 2.5
            cx = r * math.cos(ang)
            cy = r * math.sin(ang)
            centers.append((cx, cy))
            angles.append(0.0)
            
    elif config_type == 2:  # Grid-like
        positions = [
            (0, 0), (1.732, 0), (-1.732, 0),
            (0.866, 1.5), (-0.866, 1.5),
            (0.866, -1.5), (-0.866, -1.5),
            (2.598, 1.5), (-2.598, 1.5),
            (2.598, -1.5), (-2.598, -1.5)
        ]
        for x, y in positions:
            centers.append((x, y))
            angles.append(0.0)
            
    elif config_type == 3:  # Compact triangular
        positions = [
            (0, 0),
            (1.732, 0), (-1.732, 0),
            (0.866, 1.5), (-0.866, 1.5),
            (0.866, -1.5), (-0.866, -1.5),
            (2.598, 0.75), (-2.598, 0.75),
            (2.598, -0.75), (-2.598, -0.75)
        ]
        for x, y in positions:
            centers.append((x, y))
            angles.append(0.0)
            
    elif config_type == 4:  # With some rotation
        positions = [
            (0, 0),
            (1.732, 0), (-1.732, 0),
            (0.866, 1.5), (-0.866, 1.5),
            (0.866, -1.5), (-0.866, -1.5),
            (2.598, 1.5), (-2.598, 1.5),
            (0, 3.0)
        ]
        for x, y in positions:
            centers.append((x, y))
            angles.append(0.0)
        # Add some rotation to outer ones
        angles[1] = math.pi / 6
        angles[2] = -math.pi / 6
        angles[7] = math.pi / 3
        angles[8] = -math.pi / 3
        
    else:  # Random
        np.random.seed(config_type)
        for i in range(N):
            cx = np.random.uniform(-3, 3)
            cy = np.random.uniform(-3, 3)
            centers.append((cx, cy))
            angles.append(np.random.uniform(0, math.pi / 3))
    
    return np.array(centers), np.array(angles)

def optimize_config(config_type):
    """Optimize a single configuration."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), None, None, None
    
    centers, angles = make_initial_config(config_type)
    
    # Parameters: cx, cy, beta, then for each hex: x, y, angle
    # Total: 3 + 3*N = 36 parameters
    n_params = 3 + 3 * N
    
    def unpack(params):
        cx = params[0]
        cy = params[1]
        beta = ((params[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        pts = []
        angs = []
        for i in range(N):
            x = params[3 + 3 * i]
            y = params[3 + 3 * i + 1]
            a = params[3 + 3 * i + 2]
            pts.append((x, y))
            angs.append(a)
        return cx, cy, beta, np.array(pts), np.array(angs)
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        cx, cy, beta, pts, angs = unpack(params)
        
        # Smooth S
        S = smooth_S(pts, angs, beta, cx, cy, temperature=0.1)
        
        # Overlap penalty
        pen = overlap_penalty(pts, angs, temperature=0.1)
        
        return S + 10.0 * pen
    
    # Initial parameter vector
    cx0 = float(np.mean(centers[:, 0]))
    cy0 = float(np.mean(centers[:, 1]))
    beta0 = 0.0
    
    init = [cx0, cy0, beta0]
    for i in range(N):
        init.extend([centers[i][0], centers[i][1], angles[i]])
    
    # Multi-start optimization
    best_S = float('inf')
    best_params = None
    
    # Start 1: Initial
    try:
        res = minimize(objective, init, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_S and res.fun < 100:
            best_S = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Start 2: Perturbed
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        init2 = init.copy()
        for i in range(len(init2)):
            init2[i] += np.random.uniform(-0.1, 0.1)
        try:
            res = minimize(objective, init2, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S and res.fun < 100:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Start 3: Different beta
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        init3 = init.copy()
        init3[2] = math.pi / 12
        try:
            res = minimize(objective, init3, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S and res.fun < 100:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return float('inf'), None, None, None
    
    # Extract best
    cx, cy, beta, pts, angs = unpack(best_params)
    
    # Exact S for this configuration
    S_exact = exact_S(pts, angs, beta, cx, cy)
    
    # Final polish with exact objective
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 2.0:
        def exact_objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')
            cx, cy, beta, pts, angs = unpack(params)
            return exact_S(pts, angs, beta, cx, cy)
        
        try:
            res = minimize(exact_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < S_exact:
                S_exact = res.fun
                best_params = res.x.copy()
                cx, cy, beta, pts, angs = unpack(best_params)
        except Exception:
            pass
    
    return S_exact, pts, angs, (cx, cy, beta)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: single hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_global_S = float('inf')
    best_pts = None
    best_angles = None
    best_center = None
    best_beta = 0.0
    
    # Try multiple configurations
    config_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for ct in config_types:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, pts, angs, center_info = optimize_config(ct)
        
        if S < best_global_S and pts is not None:
            best_global_S = S
            best_pts = pts
            best_angles = angs
            if center_info is not None:
                best_center = (center_info[0], center_info[1])
                best_beta = center_info[2]
    
    # If no good result, use fallback
    if best_pts is None or best_global_S == float('inf'):
        # Simple hexagonal arrangement
        centers, angles = make_initial_config(0)
        cx = float(np.mean(centers[:, 0]))
        cy = float(np.mean(centers[:, 1]))
        beta = 0.0
        best_global_S = exact_S(centers, angles, beta, cx, cy)
        best_pts = centers
        best_angles = angles
        best_center = (cx, cy)
        best_beta = beta
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Build inner hexagons
    inner = []
    for i in range(N):
        x = float(best_pts[i][0])
        y = float(best_pts[i][1])
        a_deg = float(math.degrees(best_angles[i]))
        inner.append([x, y, a_deg])
    
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
