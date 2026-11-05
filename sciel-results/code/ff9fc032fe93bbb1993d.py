import math
import time
import numpy as np
from itertools import combinations

N = 11
_START = time.time()
DEADLINE = 150.0
MARGIN = 20.0

def _get_base_centers():
    """
    Returns the center of the main hexagon and the 6 first-ring neighbors.
    Coordinates are in a standard Cartesian system.
    """
    d = math.sqrt(3)
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        centers.append((d * math.cos(a), d * math.sin(a)))
    return centers

def _get_candidate_corners():
    """
    Returns the 6 possible positions for the second-ring corner hexagons.
    These are at distance 3 from the origin, at angles 30, 90, 150, 210, 270, 330.
    """
    d = 3.0
    candidates = []
    for k in range(6):
        a = math.radians(30 + 60 * k)
        candidates.append((d * math.cos(a), d * math.sin(a)))
    return candidates

def _min_enclosing_hex_side(points, rot_deg):
    """
    Calculates the side length of a regular hexagon centered at origin,
    rotated by rot_deg, that encloses all given points.
    """
    th = math.radians(rot_deg)
    # Face normals of the enclosing hexagon
    # Normals are at rot + 30 + 60*k
    normals = []
    for k in range(6):
        a = th + math.radians(30 + 60 * k)
        normals.append([math.cos(a), math.sin(a)])
    normals = np.array(normals)
    
    pts = np.array(points)
    # Project points onto normals
    # proj[i, j] = dot(pts[i], normals[j])
    proj = pts @ normals.T
    max_proj = proj.max()
    
    # Inradius R_in = S * sqrt(3) / 2
    # R_in >= max_proj  => S >= max_proj / (sqrt(3)/2)
    s = max_proj / (math.sqrt(3) / 2.0)
    return s

def construct():
    global _START, DEADLINE, MARGIN
    
    if time.time() - _START > DEADLINE - MARGIN:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 100.0, 0.0

    base_centers = _get_base_centers()
    candidate_corners = _get_candidate_corners()
    
    best_s_total = float('inf')
    best_points = None
    best_rot = 0.0
    best_subset_idx = None
    
    # There are 6 candidate corners. We need to pick 4.
    # We iterate over all combinations of 4 indices from 0..5
    for subset in combinations(range(6), 4):
        # Construct the full set of 11 points
        pts = base_centers[:]
        for idx in subset:
            pts.append(candidate_corners[idx])
        
        # Scan rotation to find minimal enclosing hexagon for these points
        # The rotation of the enclosing hexagon is what we vary.
        # Step 0.5 degrees is sufficient for high precision
        for rot in range(0, 60, 1): # 1 deg steps for speed, refine later if needed
            if time.time() - _START > DEADLINE - MARGIN:
                return inner, (0.0, 0.0), 100.0, 0.0
            
            s_centers = _min_enclosing_hex_side(pts, rot)
            s_total = s_centers + 1.0 # Add 1 for the size of the inner hexagons
            
            if s_total < best_s_total:
                best_s_total = s_total
                best_points = pts
                best_rot = float(rot)
                best_subset_idx = subset
    
    # Refine the best rotation with finer steps if time permits
    if best_s_total < float('inf'):
        # Use the best points found
        pts = best_points
        # Scan 0.1 degree steps around the best integer rotation
        start_rot = best_rot - 1.0
        end_rot = best_rot + 1.0
        for rot in np.arange(start_rot, end_rot, 0.1):
            if time.time() - _START > DEADLINE - MARGIN:
                break
            s_centers = _min_enclosing_hex_side(pts, rot)
            s_total = s_centers + 1.0
            if s_total < best_s_total:
                best_s_total = s_total
                best_rot = float(rot)
                best_points = pts # points don't change, but we keep consistency

    # Construct the final inner list
    # The inner hexagons should be rotated to match the outer hexagon
    # to minimize the "added" radius (aligning faces).
    inner = []
    for p in best_points:
        inner.append([float(p[0]), float(p[1]), float(best_rot)])
    
    outer_center = (0.0, 0.0)
    final_side = float(best_s_total)
    final_rot = float(best_rot)
    
    return inner, outer_center, final_side, final_rot
