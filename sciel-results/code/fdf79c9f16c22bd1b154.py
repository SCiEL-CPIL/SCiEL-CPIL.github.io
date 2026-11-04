import math
import time
import random
import numpy as np

N = 11
START = time.time()
DEADLINE = 110.0 # Safety margin for 120s limit

def _unit_hex_vertices():
    """Vertices of a unit regular hexagon centered at (0,0), aligned with x-axis."""
    pts = []
    for k in range(6):
        a = math.radians(60 * k)
        pts.append((math.cos(a), math.sin(a)))
    return np.array(pts)

UNIT_HEX = _unit_hex_vertices()

# Precompute normals for the hexagon edges (outward)
# Angles: 30, 90, 150, 210, 270, 330
NORMALS = np.array([[math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))] for k in range(6)])

# Precompute max projection of UNIT_HEX onto each normal
# This is constant for all hexagons of the same size and orientation
MAX_PROJ_UNITS = np.array([np.max(UNIT_HEX @ NORMALS[k]) for k in range(6)])
MIN_PROJ_UNITS = np.array([np.min(UNIT_HEX @ NORMALS[k]) for k in range(6)])

def _min_outer_side_fast(centers):
    """
    Calculate min side length of outer hexagon (centered at 0,0, rot 0) 
    containing hexagons at given centers.
    centers: (N, 2) array
    """
    # center_projs[j, k] = dot(centers[j], NORMALS[k])
    center_projs = centers @ NORMALS.T  # (N, 6)
    
    # Max projection of any vertex of hex j onto normal k is:
    # center_projs[j, k] + MAX_PROJ_UNITS[k]
    # We need the max over j for each k.
    max_over_j = np.max(center_projs, axis=0) + MAX_PROJ_UNITS
    
    # The inradius r must be at least max_over_j for each direction.
    # Since the hexagon is regular, the distance to each side is r.
    # The outer hexagon is defined by r.
    # s = 2 * r / sqrt(3)
    r_min = np.max(max_over_j)
    
    if r_min < 0:
        return 0.0
        
    s = 2.0 * r_min / math.sqrt(3.0)
    return s

def _sat_penetration_fast(A, B):
    """
    Fast SAT penetration depth for two convex polygons with fixed orientation.
    A, B: (6, 2) arrays of vertices.
    Uses precomputed NORMALS.
    """
    # Project A and B onto the 6 normals
    # projA: (6, 6) where projA[i, k] is dot(A[i], NORMALS[k])
    projA = A @ NORMALS.T
    projB = B @ NORMALS.T
    
    minA = np.min(projA, axis=0)
    maxA = np.max(projA, axis=0)
    minB = np.min(projB, axis=0)
    maxB = np.max(projB, axis=0)
    
    min_overlap = float('inf')
    
    for k in range(6):
        if maxA[k] < minB[k] or maxB[k] < minA[k]:
            return 0.0
        
        overlap = min(maxA[k], maxB[k]) - max(minA[k], minB[k])
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _overlap_penalty(centers):
    """
    Calculate total squared penetration depth between all pairs of hexagons.
    centers: (N, 2) array
    """
    penalty = 0.0
    # Precompute vertices for all hexagons to avoid repeated addition in SAT?
    # A = UNIT_HEX + center[i]
    # We can pass centers directly to a modified SAT or just compute vertices.
    # Computing vertices is N * 6 * 2 additions. Cheap.
    
    for i in range(N):
        A = UNIT_HEX + centers[i]
        for j in range(i + 1, N):
            B = UNIT_HEX + centers[j]
            pen = _sat_penetration_fast(A, B)
            if pen > 0:
                penalty += pen * pen
    return penalty

def _objective(centers):
    """
    Objective function for the fixed-rotation packing.
    centers: (N, 2) array
    """
    s = _min_outer_side_fast(centers)
    pen = _overlap_penalty(centers)
    
    # Weight for penalty
    # If there is overlap, we must resolve it.
    return s + 100.0 * pen

def construct():
    best_centers = None
    best_score = float('inf')
    
    # Check time immediately
    if time.time() > DEADLINE:
         # Fallback immediately
         return _fallback_result()

    # Seed: Hexagonal lattice
    def get_seed():
        d = math.sqrt(3.0) * 1.01
        pts = [[0.0, 0.0]]
        for k in range(6):
            a = math.radians(60 * k)
            pts.append([d * math.cos(a), d * math.sin(a)])
        # Add 4 more. Let's put them at the "corners" of the next shell
        for k in range(4):
            a = math.radians(30 + k * 90) # Spread out
            r = 2.0 * d
            pts.append([r * math.cos(a), r * math.sin(a)])
        return np.array(pts)
    
    initial_seed = get_seed()
    initial_score = _objective(initial_seed)
    
    if initial_score < best_score:
        best_centers = initial_seed.copy()
        best_score = initial_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    cooling = 0.99
    current_centers = initial_seed.copy()
    current_score = initial_score
    
    # Use a tighter loop with time checks
    while time.time() < DEADLINE - 10.0: # Leave 10s for refinement/fallback
        # Perturb
        i = random.randint(0, N - 1)
        delta_x = random.gauss(0, 0.1)
        delta_y = random.gauss(0, 0.1)
        
        new_centers = current_centers.copy()
        new_centers[i][0] += delta_x
        new_centers[i][1] += delta_y
        
        new_score = _objective(new_centers)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_centers = new_centers
            current_score = new_score
            
            if current_score < best_score:
                best_centers = current_centers.copy()
                best_score = current_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat
            T = 0.5
            for k in range(N):
                current_centers[k][0] += random.gauss(0, 0.5)
                current_centers[k][1] += random.gauss(0, 0.5)
            current_score = _objective(current_centers)
            if current_score < best_score:
                best_centers = current_centers.copy()
                best_score = current_score

    # Final refinement with Nelder-Mead on the best centers found
    if best_centers is not None and time.time() < DEADLINE - 5.0:
        try:
            from scipy.optimize import minimize
            # Limit iterations to ensure we don't blow the time
            res = minimize(
                lambda x: _objective(x.reshape(N, 2)),
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_score:
                best_centers = res.x.reshape(N, 2)
                best_score = res.fun
        except Exception:
            pass

    # Convert to list of lists (x, y, angle)
    inner = [[float(best_centers[i][0]), float(best_centers[i][1]), 0.0] for i in range(N)]
    
    # Calculate final outer side
    s = _min_outer_side_fast(best_centers)
    
    # Validity check
    valid = True
    for i in range(N):
        A = UNIT_HEX + best_centers[i]
        for j in range(i+1, N):
            B = UNIT_HEX + best_centers[j]
            if _sat_penetration_fast(A, B) > 1e-7:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback to seed
        seed_inner = [[float(p[0]), float(p[1]), 0.0] for p in get_seed()]
        seed_s = _min_outer_side_fast(np.array(seed_inner))
        return seed_inner, (0.0, 0.0), seed_s, 0.0
        
    return inner, (0.0, 0.0), s, 0.0

def _fallback_result():
    d = math.sqrt(3.0) * 1.01
    pts = [[0.0, 0.0]]
    for k in range(6):
        a = math.radians(60 * k)
        pts.append([d * math.cos(a), d * math.sin(a)])
    for k in range(4):
        a = math.radians(30 + k * 90)
        r = 2.0 * d
        pts.append([r * math.cos(a), r * math.sin(a)])
    
    inner = [[float(p[0]), float(p[1]), 0.0] for p in pts]
    s = _min_outer_side_fast(np.array(pts))
    return inner, (0.0, 0.0), s, 0.0
