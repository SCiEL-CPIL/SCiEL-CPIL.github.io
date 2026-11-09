import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations, product

def get_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Vectorized for speed.
    """
    N = len(points)
    if N < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # We need to compute area for all triples (i, j, k)
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    # To avoid O(N^3) memory blowup for large N, we can loop over i 
    # but vectorize j, k. For N=13, N^3 is small (2197), so full vectorization is fine.
    
    # Create indices
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    
    # Filter valid triples i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    i_sel = i_idx[mask]
    j_sel = j_idx[mask]
    k_sel = k_idx[mask]
    
    p_i = points[i_sel]
    p_j = points[j_sel]
    p_k = points[k_sel]
    
    # Cross product in 2D
    # (p_j - p_i) x (p_k - p_i)
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Strategy:
    # 1. Define a set of candidate points:
    #    - Regular k-gons for k in [3, 4, 5, 6, 7, 8]
    #    - Center point (0,0)
    #    - Points on a smaller inner circle (radius 0.5) at various angles
    # 2. Search for the best combination of N points from these candidates.
    #    Since 13 is small, we can use a heuristic or random sampling of subsets.
    #    However, the space of "which points are on which polygon" is discrete.
    
    # Let's refine the candidate set.
    # The "12-gon + center" topology was mentioned in inspirations.
    # Let's generate candidates:
    # - Center: (0,0)
    # - Regular 12-gon vertices (radius 1)
    # - Regular 6-gon vertices (radius 1)
    # - Regular 4-gon vertices (radius 1)
    # - Points on radius 0.5 at angles of 12-gon
    # - Points on radius 0.5 at angles of 6-gon
    
    candidates = []
    
    # Center
    candidates.append(np.array([0.0, 0.0]))
    
    # Regular 12-gon (radius 1)
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for a in angles_12:
        candidates.append(np.array([np.cos(a), np.sin(a)]))
        
    # Regular 6-gon (radius 1) - subset of 12-gon, but let's include explicitly if we want distinct roles?
    # Actually, 6-gon vertices are a subset of 12-gon vertices.
    # Let's add some intermediate radii to allow "inner" points that aren't just center.
    
    # Inner 12-gon (radius 0.5)
    for a in angles_12:
        candidates.append(0.5 * np.array([np.cos(a), np.sin(a)]))
        
    # Inner 6-gon (radius 0.5)
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for a in angles_6:
        candidates.append(0.5 * np.array([np.cos(a), np.sin(a)]))
        
    # Inner 4-gon (radius 0.5)
    angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    for a in angles_4:
        candidates.append(0.5 * np.array([np.cos(a), np.sin(a)]))
        
    # Maybe some radius 0.7?
    for a in angles_12:
        candidates.append(0.7 * np.array([np.cos(a), np.sin(a)]))

    candidates = np.array(candidates)
    M = len(candidates)
    
    # We need to select N=13 points from M candidates.
    # M is roughly 1 + 12 + 12 + 6 + 4 + 12 = 47.
    # C(47, 13) is huge. We need a smarter search.
    
    # Alternative Strategy:
    # Assume a topology: k points on outer regular polygon, N-k points on inner regular polygon (or center).
    # Let's try combinations of (k_outer, k_inner) where k_outer + k_inner = 13.
    # And vary the rotation of the inner polygon relative to the outer one.
    
    def score_topology(k_outer, k_inner, r_inner, rot_offset):
        """
        Construct points:
        - k_outer points on regular k_outer-gon (radius 1)
        - k_inner points on regular k_inner-gon (radius r_inner) with rotation rot_offset
        - If k_inner == 1, it's the center (or single point on inner circle? No, let's say center if 1)
        Actually, let's keep it general:
        Outer: regular k_outer-gon.
        Inner: regular k_inner-gon.
        Total N = k_outer + k_inner.
        """
        if k_outer == 0 or k_inner == 0:
            return 0.0, None
            
        # Outer points
        if k_outer == 1:
            outer_pts = np.array([[1.0, 0.0]])
        else:
            angles_o = np.linspace(0, 2 * np.pi, k_outer, endpoint=False)
            outer_pts = np.column_stack((np.cos(angles_o), np.sin(angles_o)))
            
        # Inner points
        if k_inner == 1:
            # Center
            inner_pts = np.array([[0.0, 0.0]])
        else:
            angles_i = np.linspace(rot_offset, 2 * np.pi + rot_offset, k_inner, endpoint=False)
            inner_pts = r_inner * np.column_stack((np.cos(angles_i), np.sin(angles_i)))
            
        pts = np.vstack([outer_pts, inner_pts])
        
        # Check for duplicates?
        # If r_inner is very small and k_inner > 1, they might be close but distinct.
        # If r_inner = 0 and k_inner > 1, they are all at center -> invalid hull or bad score.
        # We ensure r_inner > 0 for k_inner > 1.
        
        score = get_score(pts)
        return score, pts

    # Search over topologies
    # k_outer from 3 to 12 (need at least 3 for hull, but if k_inner provides hull points, k_outer can be smaller?
    # The convex hull of the set must have area > 0.
    # If k_outer >= 3, hull is determined by outer points.
    # If k_outer < 3, hull must come from inner points, which requires r_inner > 0 and k_inner >= 3.
    # Given we want to maximize area, outer points should be on the boundary.
    # So let's assume k_outer >= 3.
    
    # Possible splits of 13:
    # (3,10), (4,9), (5,8), (6,7), (7,6), (8,5), (9,4), (10,3), (11,2), (12,1), (13,0 - invalid inner)
    # Also (2,11) etc if inner forms hull? Unlikely to be better than outer on unit circle.
    
    splits = []
    for k_out in range(3, 14):
        k_in = 13 - k_out
        if k_in >= 0:
            splits.append((k_out, k_in))
            
    # Radii for inner points
    # If k_in == 1, r_in is irrelevant (center).
    # If k_in > 1, try several radii: 0.1, 0.3, 0.5, 0.7, 0.9
    radii_to_try = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Rotations for inner points
    # Symmetry suggests we only need to check rotations up to the symmetry of the polygons.
    # For simplicity, try a few key rotations: 0, pi/k_out, etc.
    # Or just sample a few.
    rotations_to_try = [0.0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]
    
    # We will iterate through these configurations.
    # Number of configs: ~10 splits * 5 radii * 5 rotations = 250.
    # Each score takes ~ms. Total < 1s.
    # This allows us to use remaining time for local refinement or random perturbation.
    
    configs_checked = 0
    for k_out, k_in in splits:
        if k_in == 0:
            # All points on outer polygon
            # This is just a regular k_out-gon.
            score, pts = score_topology(k_out, 0, 0, 0) # score_topology handles k_in=0? 
            # My score_topology returns 0 for k_in=0. Let's handle k_in=0 separately or fix function.
            # Let's fix score_topology to handle k_in=0.
            pass
            
        if k_in == 0:
             # Regular k_out-gon
             angles_o = np.linspace(0, 2 * np.pi, k_out, endpoint=False)
             pts = np.column_stack((np.cos(angles_o), np.sin(angles_o)))
             score = get_score(pts)
             if score > best_score:
                 best_score = score
                 best_points = pts.copy()
             continue
             
        for r_in in radii_to_try:
            if k_in == 1:
                # Center
                rot_vals = [0.0]
            else:
                rot_vals = rotations_to_try
                
            for rot in rot_vals:
                if time.time() > deadline:
                    break
                score, pts = score_topology(k_out, k_in, r_in, rot)
                configs_checked += 1
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                
    # After the structured search, use remaining time for random local search around the best point
    # to fine-tune positions.
    if best_points is not None and time.time() < deadline - 5:
        # Simple random perturbation
        # We can't easily map back to parameters, so just jitter the points directly.
        # But we must maintain the "structure" roughly? No, just jitter.
        # Jittering might break the convex hull order or create duplicates.
        # Let's do a more careful local search:
        # Treat each point as a variable. Use Nelder-Mead on the raw coordinates?
        # 26 dimensions. Might be slow.
        
        # Instead, let's try to refine the specific topology found.
        # Identify which topology gave the best score?
        # We didn't store it. Let's re-run the best topology with a finer grid or optimization.
        
        # For now, let's just do a few random restarts of the topology search with slightly different radii/rotations.
        for _ in range(10):
            if time.time() > deadline:
                break
            # Randomize radii and rotations a bit
            k_out, k_in = np.random.choice(splits)
            r_in = np.random.uniform(0.1, 0.9)
            rot = np.random.uniform(0, 2 * np.pi)
            score, pts = score_topology(k_out, k_in, r_in, rot)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    return best_points

# Fix score_topology to handle k_in=0 properly if called, though we handled it in loop.
# Actually, the loop handles k_in=0. The function is only called for k_in>0.
# But wait, I defined score_topology above. It returns 0 for k_in=0.
# The loop checks `if k_in == 0` and continues, so it doesn't call score_topology for k_in=0.
# So it's fine.

# One issue: `score_topology` uses `get_score` which calculates hull.
# If points are collinear or duplicate, it returns 0.
# This is safe.

# Let's double check the `score_topology` logic for k_in=1.
# If k_in=1, inner_pts is [[0,0]].
# If k_out=12, total 13 points.
# This matches the "12-gon + center" inspiration.

# Is there a better strategy?
# The inspiration mentioned "12-gon+center topology ... improve from 39.3 to 49.6".
# My search includes (12, 1) with r_in=0 (center).
# It also includes other splits.
# This should find the 49.6+ solution if it exists in this parametric family.

# To be safe, let's add a final "polish" step using scipy.optimize.minimize on the coordinates
# of the best point found, using the raw score as objective.
# This might help if the discrete grid missed the exact optimal rotation/radius.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # ... (same topology search as above) ...
    
    # Refactor: Put topology search in a helper or inline.
    
    # Inline topology search
    splits = [(k, 13-k) for k in range(3, 14)]
    radii_to_try = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 0.0 is only valid for k_in=1 (center) or if we treat it as degenerate.
    # If k_in > 1 and r=0, all inner points are at center -> duplicates -> score 0.
    
    for k_out, k_in in splits:
        if k_in == 0:
            angles_o = np.linspace(0, 2 * np.pi, k_out, endpoint=False)
            pts = np.column_stack((np.cos(angles_o), np.sin(angles_o)))
            score = get_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
            continue
            
        # Determine valid radii
        if k_in == 1:
            radii_valid = [0.0] # Center
            rots_valid = [0.0]
        else:
            radii_valid = [r for r in radii_to_try if r > 0.01]
            # Rotations: sample 12 values
            rots_valid = np.linspace(0, 2*np.pi, 12, endpoint=False)
            
        for r_in in radii_valid:
            for rot in rots_valid:
                if time.time() > deadline:
                    break
                # Construct points
                angles_o = np.linspace(0, 2 * np.pi, k_out, endpoint=False)
                outer_pts = np.column_stack((np.cos(angles_o), np.sin(angles_o)))
                
                if k_in == 1:
                    inner_pts = np.array([[0.0, 0.0]])
                else:
                    angles_i = np.linspace(rot, 2 * np.pi + rot, k_in, endpoint=False)
                    inner_pts = r_in * np.column_stack((np.cos(angles_i), np.sin(angles_i)))
                    
                pts = np.vstack([outer_pts, inner_pts])
                score = get_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    
    # Local refinement using Nelder-Mead on the coordinates of the best point
    if best_points is not None and time.time() < deadline - 10:
        def obj(params):
            pts = params.reshape(-1, 2)
            return -get_score(pts)
            
        x0 = best_points.flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
            if -res.fun > best_score:
                best_score = -res.fun
                best_points = res.x.reshape(-1, 2)
        except:
            pass
            
    return best_points

# Need to import minimize
from scipy.optimize import minimize

# Final check on imports and structure.
# The function `construct` is defined.
# `get_score` is defined.
# Time guard is present.
# Strategy is structurally different (discrete topology search + local polish vs pure continuous param optimization).
