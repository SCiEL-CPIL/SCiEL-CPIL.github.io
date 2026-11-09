import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Check for finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_tetrahedron_midpoints():
    """
    Generate base template points for a regular tetrahedron.
    Returns:
        vertices: (4, 3) array
        midpoints: (10, 3) array
    """
    # Regular tetrahedron vertices centered at origin, normalized to unit length
    # Using the standard cross-product construction or explicit coordinates
    # Vertices of a regular tetrahedron can be taken as:
    # (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Calculate 10 edge midpoints
    mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            # Normalize midpoints to unit sphere as well? 
            # The strategy says "optimize radial scaling factor of each orbit".
            # This implies the base points are fixed directions, and we scale them.
            # If we normalize the midpoints to unit length, they lie on the sphere.
            # If we don't, they are closer to the origin.
            # The "radial scaling factor" s_i means point = s_i * base_dir.
            # So base_dir should be a unit vector.
            mid_dir = mid / np.linalg.norm(mid)
            mids.append(mid_dir)
            
    mids = np.array(mids)
    return verts, mids

def _build_points(scales):
    """
    Construct 14 points from 5 scale factors.
    scales: [s_v, s_m1, s_m2, s_m3, s_m4]
    Where s_v scales the 4 vertices.
    s_m1..s_m4 scale the 10 midpoints? 
    Wait, the midpoints form orbits under the tetrahedral group T.
    The 10 edges of a tetrahedron form a single orbit under the full tetrahedral group T (order 12) or Td (order 24).
    Actually, the edge set is a single orbit. 
    So all 10 midpoints are equivalent under symmetry.
    Thus, we only need 2 parameters: s_vertex and s_midpoint.
    
    Let's re-read the prompt's strategy hint: "optimize only the single radial scaling factor of each of the 5 symmetry orbits".
    Are there 5 orbits?
    The 14 points are 4 vertices + 10 midpoints.
    Under the symmetry group of the tetrahedron (Td), the 4 vertices form one orbit.
    The 10 midpoints form another orbit.
    So there are only 2 orbits.
    
    Why did the prompt say 5? 
    Maybe it considers a different template? Or maybe it allows breaking symmetry slightly?
    Or maybe it refers to a different decomposition?
    "4 vertices + 10 edge midpoints" is the specific template mentioned.
    If I use only 2 parameters, the search space is 2D.
    
    However, the "5 symmetry orbits" might refer to a different structure, or perhaps the prompt implies that we can vary the scales of subsets of midpoints if we break symmetry? 
    But the prompt says "high-symmetry combinatorial template".
    If we stick to full Td symmetry, we have 2 parameters.
    
    Let's check if 2 parameters is enough.
    The current best is ~49.
    With 2 parameters, we can do a very fine grid search.
    
    Let's consider if we should allow the midpoints to be in multiple orbits?
    If we break symmetry, we lose the benefit of the template.
    
    Let's stick to the 2-parameter Td symmetric case first:
    P_v = s_v * v_i
    P_m = s_m * m_j
    
    We will grid search s_v and s_m.
    Since the metric is scale invariant, the absolute scale doesn't matter, only the ratio s_m / s_v.
    So we can fix s_v = 1 and search s_m.
    This reduces it to 1D!
    
    Is that true?
    (d_min / d_max)^2.
    If we scale all points by k, d_min and d_max both scale by k, so the ratio is invariant.
    So yes, only the ratio matters.
    
    Let's verify the distances.
    Distances within vertices: |s_v v_i - s_v v_j| = s_v * |v_i - v_j|.
    Distances within midpoints: |s_m m_i - s_m m_j| = s_m * |m_i - m_j|.
    Distances between vertex and midpoint: |s_v v_i - s_m m_j|.
    
    Let r = s_m / s_v. Fix s_v = 1.
    Points:
    V_i = v_i (4 points)
    M_j = r * m_j (10 points)
    
    We need to optimize r.
    This is a 1D optimization problem.
    
    Let's implement this 1D search.
    """
    verts, mids = _generate_tetrahedron_midpoints()
    
    s_v = 1.0
    s_m = scales[0] if len(scales) > 0 else 1.0
    
    pts_v = s_v * verts
    pts_m = s_m * mids
    
    points = np.vstack([pts_v, pts_m])
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    verts, mids = _generate_tetrahedron_midpoints()
    
    best_score = -np.inf
    best_r = 1.0
    best_points = None
    
    # Define the objective for a given r (ratio of midpoint scale to vertex scale)
    def _score_r(r):
        if r <= 0:
            return 0.0
        pts_v = verts # s_v = 1
        pts_m = r * mids
        points = np.vstack([pts_v, pts_m])
        return _score(points)
    
    # 1. Coarse Grid Search for r
    # r is the scale of midpoints relative to vertices.
    # If r=1, midpoints are at their natural position (distance 1/sqrt(2) from origin? No, normalized).
    # Let's check the geometry.
    # v = (1,1,1)/sqrt(3). |v|=1.
    # m = (v1+v2)/2. |m| = 1/sqrt(2). Direction is normalized.
    # So if r=1, the midpoints are at distance 1 from origin, same as vertices.
    # The points lie on the unit sphere.
    
    # We expect the optimal r to be somewhere near 1, maybe slightly different.
    # Let's search r in [0.1, 3.0].
    
    r_coarse = np.linspace(0.2, 3.0, 100)
    
    scores = []
    for r in r_coarse:
        if time.time() > deadline - 60:
            break
        s = _score_r(r)
        scores.append(s)
        
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_r_coarse = r_coarse[best_idx]
    best_score = scores[best_idx]
    best_points = _build_points([best_r_coarse])
    
    # 2. Refine using Golden Section Search or Fine Grid
    # The function is likely unimodal or has few local maxima in 1D.
    # Let's do a fine grid around the best coarse candidate.
    
    # Determine a reasonable window for refinement
    # Step size in coarse was (3.0-0.2)/99 ~ 0.028
    # Let's refine in [best_r - 0.1, best_r + 0.1]
    
    lower = max(0.01, best_r_coarse - 0.2)
    upper = best_r_coarse + 0.2
    
    r_fine = np.linspace(lower, upper, 500)
    
    for r in r_fine:
        if time.time() > deadline - 30:
            break
        s = _score_r(r)
        if s > best_score:
            best_score = s
            best_r = r
            best_points = _build_points([r])
            
    # 3. Final Polish with Nelder-Mead on 1D (or just manual bisection if NM is overkill)
    # Since it's 1D, we can just do a very fine grid or use minimize_scalar.
    # Let's use scipy.optimize.minimize_scalar on the negative score.
    
    try:
        res = minimize_scalar(
            lambda x: -_score_r(x),
            bounds=(max(0.01, best_r - 0.05), best_r + 0.05),
            method='bounded',
            options={'xatol': 1e-10}
        )
        if res.success:
            r_opt = res.x
            s_opt = -res.fun
            if s_opt > best_score:
                best_score = s_opt
                best_r = r_opt
                best_points = _build_points([r_opt])
    except Exception:
        pass
    
    # Fallback if something went wrong
    if best_points is None:
        best_points = _build_points([1.0])
        
    # Ensure output shape is (14, 3) and finite
    if best_points.shape != (14, 3):
        # Reconstruct from best_r
        best_points = _build_points([best_r])
        
    return best_points

# EVOLVE-BLOCK-END
