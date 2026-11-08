import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute pair indices for distance objective
_pair_combos = list(combinations(range(N), 2))
_pidx_i = np.array([c[0] for c in _pair_combos])
_pidx_j = np.array([c[1] for c in _pair_combos])


def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Constraints:
    1. y >= 0
    2. y <= sqrt(3) * x  (Line AC)
    3. sqrt(3) * x + y <= sqrt(3) (Line BC)
    """
    pts = np.array(pts, dtype=float)
    # We might need iterative projection or simple clipping.
    # Simple clipping might violate constraints if corners are sharp.
    # For Nelder-Mead, simple projection is often sufficient if steps are small.
    # Let's do a few iterations of clipping to be safe.
    for _ in range(10):
        # y >= 0
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        # x >= 0 (implied by y <= sqrt(3)x if y>=0, but let's enforce x>=0)
        pts[:, 0] = np.maximum(pts[:, 0], 0)
        
        # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
        # If y > sqrt(3) x, project onto y = sqrt(3) x
        # The line is y = sqrt(3)x. Normal vector (-sqrt(3), 1).
        # Actually, just clipping x to be at least y/sqrt(3) is a valid projection onto the half-plane?
        # No, projection onto a half-plane is orthogonal.
        # But for optimization, a simple coordinate-wise fix-up is often robust enough.
        # Let's use the explicit inequalities.
        
        # Constraint 2: y - sqrt(3)x <= 0
        # If violated, we need to move point towards the feasible side.
        # The feasible side is y < sqrt(3)x.
        # Let's just clip x to be large enough? Or y small enough?
        # Orthogonal projection onto line y = sqrt(3)x:
        # This is getting complex. Given the time limit, a simpler "push back" strategy works.
        # If y > sqrt(3)*x, set y = sqrt(3)*x. (This keeps x, reduces y).
        mask2 = pts[:, 1] > _H * 2.0 * pts[:, 0] # sqrt(3) is approx 1.732, _H is 0.866. 2*_H = sqrt(3).
        pts[mask2, 1] = _H * 2.0 * pts[mask2, 0]
        
        # Constraint 3: sqrt(3)*x + y <= sqrt(3)
        # If violated, set y = sqrt(3) - sqrt(3)*x. (This keeps x, reduces y).
        # Note: if x > 1, this makes y negative.
        mask3 = _H * 2.0 * pts[:, 0] + pts[:, 1] > _H * 2.0
        pts[mask3, 1] = _H * 2.0 - _H * 2.0 * pts[mask3, 0]
        
        # Re-check y >= 0 after adjustments
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        
        # If x becomes > 1 due to some weird interaction, clip x?
        # The triangle is within x in [0, 1].
        pts[:, 0] = np.minimum(pts[:, 0], 1.0)
        
        # One more pass for stability
        mask2 = pts[:, 1] > _H * 2.0 * pts[:, 0]
        pts[mask2, 1] = _H * 2.0 * pts[mask2, 0]
        mask3 = _H * 2.0 * pts[:, 0] + pts[:, 1] > _H * 2.0
        pts[mask3, 1] = _H * 2.0 - _H * 2.0 * pts[mask3, 0]
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        
    return pts


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    # Area = 0.5 * |cross product|
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _min_dist_sq(points):
    """Compute minimum squared distance between any pair."""
    pi = points[_pidx_i]
    pj = points[_pidx_j]
    diff = pi - pj
    dists = np.sum(diff * diff, axis=1)
    return np.min(dists)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Objective function for the main optimization (maximize min area)
    def objective_main(flat):
        pts = _project_to_triangle(flat.reshape(N, 2))
        # Return negative area to minimize
        return -_min_triangle_area(pts)
    
    # Objective function for initialization (maximize min distance)
    def objective_dist(flat):
        pts = _project_to_triangle(flat.reshape(N, 2))
        return -_min_dist_sq(pts)

    rng = np.random.default_rng(42)
    
    # Strategy 1: Max-Min Distance Initialization
    # Run a few minimizations on the distance objective to get a good spread
    init_candidates = []
    
    # 1. Random start for distance optimization
    for i in range(3):
        if time.time() > deadline - 20: break
        # Random points in bounding box, then projected
        rand_pts = rng.random((N, 2)) * [1.0, _H]
        rand_pts = _project_to_triangle(rand_pts)
        
        res = minimize(objective_dist, rand_pts.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        init_candidates.append(res.x)
        
    # 2. Geometric heuristic: Place points in a grid-like fashion
    # Approximate a hexagonal packing
    hex_pts = []
    # Center
    hex_pts.append([0.5, _H/3]) 
    # Surrounding
    # This is hard to generate perfectly. Let's just add a few structured starts.
    # Start with vertices and centroid
    verts = np.array([[0,0], [1,0], [0.5, _H]])
    centroid = [0.5, _H/3]
    
    # A simple symmetric start:
    # 3 vertices, 3 midpoints of edges, 3 points on medians, 1 centroid?
    # Let's just rely on the distance optimizer results and randoms.
    
    # Add some random starts for the main optimizer as well
    for i in range(5):
        if time.time() > deadline - 20: break
        rand_pts = rng.random((N, 2)) * [1.0, _H]
        rand_pts = _project_to_triangle(rand_pts)
        init_candidates.append(rand_pts.flatten())

    # Run main optimization on each candidate
    for init_flat in init_candidates:
        if time.time() > deadline - 5: break
        
        try:
            res = minimize(objective_main, init_flat, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            final_pts = _project_to_triangle(res.x.reshape(N, 2))
            score = _min_triangle_area(final_pts)
            
            if score > best_score:
                best_score = score
                best_points = final_pts
                
        except Exception:
            continue

    if best_points is None:
        # Fallback: Random points
        pts = rng.random((N, 2)) * [1.0, _H]
        best_points = _project_to_triangle(pts)
        
    return best_points

# EVOLVE-BLOCK-END
