# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _get_min_triangle_info(points: np.ndarray):
    """
    Returns (min_area, hull_area, min_triangle_indices, all_areas)
    """
    n = points.shape[0]
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0, 1.0, None, None
        
    if A_hull <= 1e-10:
        return 0.0, 1.0, None, None

    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    min_tri = triples_arr[min_idx]
    
    return min_area, A_hull, min_tri, areas

def _generate_initializations():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal packing approximation
    pts = []
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random points
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def _try_expand_triangle(points, tri_idx, step_size, rng):
    """
    Attempts to expand the triangle at tri_idx by moving its vertices.
    Returns a new point set if an improvement is found, otherwise None.
    """
    i, j, k = tri_idx
    p_i = points[i].copy()
    p_j = points[j].copy()
    p_k = points[k].copy()
    
    # Calculate vectors
    v_ji = p_j - p_i
    v_ki = p_k - p_i
    area = 0.5 * np.abs(np.cross(v_ji, v_ki))
    
    if area < 1e-10:
        return None

    # Normal to the triangle plane (in 2D, this is just a 90-degree rotation)
    # We want to move vertices to increase area.
    # Moving p_i away from line jk increases area.
    # Direction from line jk to p_i:
    # Vector jk: p_k - p_j
    # Normal to jk: n_jk
    # If cross(jk, p_i - p_j) > 0, p_i is on one side. We want to move p_i in direction n_jk * sign.
    
    def get_expansion_dir(p_m, p_n, p_o):
        """Get direction to move p_o to increase area of triangle m,n,o"""
        v_mn = p_n - p_m
        v_mo = p_o - p_m
        cross_val = v_mn[0] * v_mo[1] - v_mn[1] * v_mo[0]
        if abs(cross_val) < 1e-12:
            return None
        # Normal to MN
        n_mn = np.array([-v_mn[1], v_mn[0]])
        # Ensure n_mn points away from MN towards O
        # Dot product of n_mn and v_mo should have same sign as cross_val?
        # Actually, just normalize and check.
        n_mn = n_mn / np.linalg.norm(n_mn)
        # If p_o is on the "positive" side, move in n_mn direction.
        # cross_val > 0 means O is to the left of MN. n_mn is 90 deg CCW from MN.
        # So n_mn points towards O if cross_val > 0?
        # Let's verify: MN = (1,0). O = (0,1). Cross = 1. n_MN = (0,1). Points to O. Correct.
        # If MN = (1,0), O = (0,-1). Cross = -1. n_MN = (0,1). Points away from O.
        # So if cross_val < 0, we need -n_mn.
        if cross_val < 0:
            n_mn = -n_mn
        return n_mn

    best_candidate = None
    best_score = _compute_score(points)
    
    # Try moving each vertex
    for idx, p in [(i, p_i), (j, p_j), (k, p_k)]:
        # Determine which two other points form the base
        others = [x for x in [i, j, k] if x != idx]
        p_a = points[others[0]]
        p_b = points[others[1]]
        
        dir_vec = get_expansion_dir(p_a, p_b, p)
        if dir_vec is None:
            continue
            
        # Try a few step sizes
        for step in [step_size * 0.1, step_size * 0.5, step_size, step_size * 2.0]:
            new_p = p + dir_vec * step
            # Keep in bounds
            new_p = np.clip(new_p, 0.01, 0.99)
            
            new_points = points.copy()
            new_points[idx] = new_p
            
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_candidate = new_points
                
    # Also try random small perturbations for all 3 vertices to escape local minima
    for _ in range(5):
        new_points = points.copy()
        for idx in [i, j, k]:
            new_points[idx] = np.clip(new_points[idx] + rng.normal(0, step_size * 0.1, 2), 0.01, 0.99)
        new_score = _compute_score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_candidate = new_points
            
    return best_candidate

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using bottleneck-targeted coordinate ascent.
    """
    start_time = time.time()
    deadline = 600  # seconds
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(123)
    
    for init_pts in inits:
        if time.time() - start_time > deadline - 10:
            break
            
        current_points = init_pts.copy()
        current_score = _compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Optimization loop for this initialization
        stagnation_counter = 0
        step_size = 0.05
        max_iter = 1000
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
                
            min_area, A_hull, min_tri, _ = _get_min_triangle_info(current_points)
            
            if min_tri is None:
                break
                
            # Try to expand the bottleneck triangle
            candidate = _try_expand_triangle(current_points, min_tri, step_size, rng)
            
            if candidate is not None:
                current_points = candidate
                current_score = _compute_score(current_points)
                stagnation_counter = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                stagnation_counter += 1
                # If stuck, try a global shake
                if stagnation_counter % 10 == 0:
                    shake_points = current_points + rng.normal(0, 0.02, (13, 2))
                    shake_points = np.clip(shake_points, 0.01, 0.99)
                    shake_score = _compute_score(shake_points)
                    if shake_score > current_score * 0.9: # Accept if not much worse
                        current_points = shake_points
                        current_score = shake_score
                        step_size *= 0.9 # Shrink step size
                    else:
                        step_size *= 0.9
                        if step_size < 1e-4:
                            break # Local minimum reached

    if best_points is None:
        best_points = inits[0]
        
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
