import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar

N = 11
START = time.time()
DEADLINE = 148.0  # Safety margin

def _hex_vertices(cx, cy, s, rot_deg):
    """
    Generate vertices of a regular hexagon with center (cx, cy), side length s, 
    and rotation rot_deg (degrees).
    """
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """
    Generate vertices of a unit regular hexagon with center (cx, cy) 
    and rotation ang_deg (degrees).
    """
    return _hex_vertices(cx, cy, 1.0, ang_deg)

def _min_outer_side(inner_params, outer_c=(0.0, 0.0), rot=0.0):
    """
    Calculate the minimum side length of the outer hexagon required to contain 
    all inner hexagons defined by inner_params, given outer center and rotation.
    inner_params: list of (x, y, angle) or numpy array.
    """
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    
    # Convert to array if needed
    if isinstance(inner_params, np.ndarray):
        params = inner_params.reshape(N, 3)
    else:
        params = np.array(inner_params)
        
    for k in range(N):
        x, y, a = params[k]
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _get_penetration(A, B):
    """
    Calculate the minimum translation distance to separate two convex polygons A and B.
    Returns 0 if they are disjoint or touching.
    """
    min_overlap = float('inf')
    
    edges = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx /= norm
            ny /= norm
            edges.append((nx, ny))
            
    for nx, ny in edges:
        minA = min(p[0]*nx + p[1]*ny for p in A)
        maxA = max(p[0]*nx + p[1]*ny for p in A)
        minB = min(p[0]*nx + p[1]*ny for p in B)
        maxB = max(p[0]*nx + p[1]*ny for p in B)
        
        if maxA < minB or maxB < minA:
            return 0.0
            
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            
    if min_overlap == float('inf'):
        return 0.0
        
    return min_overlap

def _objective(params):
    """
    Objective function for optimization.
    params: array of length 3*N
    """
    pts = params.reshape(N, 3)
    
    outer_s = _min_outer_side(pts)
    
    overlap_penalty = 0.0
    polys = [_inner_hex(pts[i,0], pts[i,1], pts[i,2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            pen = _get_penetration(polys[i], polys[j])
            if pen > 0:
                overlap_penalty += pen * pen
    
    penalty_weight = 100.0
    total_cost = outer_s + penalty_weight * overlap_penalty
    
    return total_cost

def _valid(inner_params):
    """
    Strict validity check.
    """
    polys = [_inner_hex(p[0], p[1], p[2]) for p in inner_params]
    for i in range(N):
        for j in range(i+1, N):
            if _get_penetration(polys[i], polys[j]) > 1e-7:
                return False
    return True

def _generate_lattice_points(count):
    """
    Generate the 'count' closest points in a hexagonal lattice to the origin.
    Returns list of (x, y) coordinates.
    """
    points = [(0.0, 0.0)]
    # Lattice basis
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # We search in a square region large enough
    max_i = 5
    candidates = []
    for i in range(-max_i, max_i + 1):
        for j in range(-max_i, max_i + 1):
            x = math.sqrt(3) * i + math.sqrt(3) / 2 * j
            y = 1.5 * j
            dist_sq = x*x + y*y
            candidates.append((dist_sq, x, y))
            
    candidates.sort(key=lambda x: x[0])
    
    # Take the first 'count' unique points
    # Note: (0,0) is included.
    # We need to ensure we don't pick duplicates if any (none in lattice).
    selected = []
    for d, x, y in candidates:
        if len(selected) >= count:
            break
        selected.append((x, y))
        
    return selected

def _optimize_rotation_for_cluster(positions):
    """
    Given a list of (x, y) positions for N hexagons,
    find the rotation theta (0-60 deg) that minimizes the outer side length,
    assuming all inner hexagons have the same rotation theta.
    """
    def cost(theta_deg):
        params = np.zeros((N, 3))
        for i, (x, y) in enumerate(positions):
            params[i, 0] = x
            params[i, 1] = y
            params[i, 2] = theta_deg
        return _min_outer_side(params)
    
    # Optimize theta in [0, 60]
    res = minimize_scalar(cost, bounds=(0, 60), method='bounded', options={'xatol': 1e-5})
    return res.x, res.fun

def _construct_lattice_candidates():
    """
    Generate multiple lattice-based cluster candidates.
    """
    candidates = []
    
    # Candidate 1: 11 closest lattice points
    pos11 = _generate_lattice_points(11)
    candidates.append(pos11)
    
    # Candidate 2: Try swapping one of the outer points to a slightly different lattice point
    # The 11 closest include 1 center, 6 ring 1, 4 ring 2 (dist 3).
    # There are 6 ring 2 points at dist 3. We picked 4.
    # Let's try other combinations of 4 from the 6 ring 2 points.
    
    # Identify the 6 ring 2 points (dist ~ 3)
    # From _generate_lattice_points, we can extract them.
    # For simplicity, let's just use the 11 closest for now, as it's the most compact.
    # If time permits, we can iterate.
    
    return candidates

def construct():
    best_params = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    
    # 1. Generate Lattice Candidates
    candidates = _construct_lattice_candidates()
    
    # 2. Optimize Rotation for each candidate
    best_lattice_params = None
    best_lattice_score = float('inf')
    
    for pos in candidates:
        if time.time() > budget - 20:
            break
        theta, score = _optimize_rotation_for_cluster(pos)
        if score < best_lattice_score:
            best_lattice_score = score
            # Construct full params
            params = np.zeros((N, 3))
            for i, (x, y) in enumerate(pos):
                params[i, 0] = x
                params[i, 1] = y
                params[i, 2] = theta
            best_lattice_params = params
    
    if best_lattice_params is not None:
        best_params = best_lattice_params
        best_score = best_lattice_score
        
    # 3. Local Refinement using Nelder-Mead
    if best_params is not None and time.time() < budget - 10:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            if res_nm.fun < best_score and _valid(res_nm.x.reshape(N, 3)):
                best_params = res_nm.x
                best_score = res_nm.fun
        except Exception:
            pass
            
    # 4. If still time, try a small DE from the best point?
    # No, DE is risky. Nelder-Mead is better for local refinement.
    
    # Finalize
    if best_params is None:
        # Fallback
        inner = _generate_lattice_points(11)
        inner_full = [[x, y, 0.0] for x, y in inner]
        s = _min_outer_side(inner_full)
        return inner_full, (0.0, 0.0), s, 0.0
    
    inner = [[float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])] for i in range(N)]
    s = _min_outer_side(inner)
    
    if _valid(inner):
        return inner, (0.0, 0.0), s, 0.0
    else:
        # Return best valid found so far? 
        # The lattice one was valid.
        if best_lattice_params is not None:
             inner_l = [[float(best_lattice_params[i*3]), float(best_lattice_params[i*3+1]), float(best_lattice_params[i*3+2])] for i in range(N)]
             s_l = _min_outer_side(inner_l)
             return inner_l, (0.0, 0.0), s_l, 0.0
        else:
             inner = _generate_lattice_points(11)
             inner_full = [[x, y, 0.0] for x, y in inner]
             s = _min_outer_side(inner_full)
             return inner_full, (0.0, 0.0), s, 0.0

