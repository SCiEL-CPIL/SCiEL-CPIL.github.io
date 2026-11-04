"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Precompute indices for all triplets i < j < k
    # This is done once per call, but N is small.
    # We can cache these indices globally for N=14 to save time.
    # However, for simplicity and safety, we compute them.
    # Actually, let's make a helper that uses vectorized ops.
    
    # For N=14, the number of triples is 364.
    # Vectorized calculation:
    # We need to extract pts[i], pts[j], pts[k] for all combinations.
    
    # Using np.triu_indices or similar is tricky for triples.
    # Let's use a pre-computed list of indices if possible, or just a fast loop.
    # A fast loop in Python for 364 iterations is ~0.1ms.
    # Vectorized numpy overhead might be similar.
    # Let's try a hybrid: compute all pairwise cross products? No.
    
    # Let's stick to the vectorized index approach, it's clean.
    # To avoid regenerating indices every time, we can store them.
    # But the function signature is fixed. We can use a closure or global.
    
    # Let's just use the vectorized index generation inside, it's fast enough for 14.
    # Or better, pre-compute the indices for N=14 globally.
    
    return _min_triangle_area_impl(pts)

# Pre-compute indices for N=14
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area_impl(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -np.inf
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two Rings: 7 inner, 7 outer
    # Inner radius 0.5, Outer radius 1.0
    # Interleaved angles
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = inner_angles + (2 * np.pi / 14)
    
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    
    # Interleave them
    two_ring_pts = np.vstack([inner_pts, outer_pts])
    # Sort by angle to maintain some order? Not strictly necessary for score, but helps optimizer?
    # Let's just stack.
    inits.append(two_ring_pts)
    
    # 3. Grid (4x4 minus 2)
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    inits.append(grid_pts[:14])
    
    # 4. Random (seeded)
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N, 2) * 2 - 1)
    
    # 5. Perturbed Regular Polygon
    rng2 = np.random.RandomState(123)
    perturbed = inits[0] + rng2.normal(0, 0.1, size=(N, 2))
    inits.append(perturbed)
    
    def objective(x):
        return -_score(x.reshape(-1, 2))
        
    for idx_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_score = _score(init_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = init_pts.copy()
            
        # Optimize
        try:
            # Use Nelder-Mead for robustness
            res = minimize(objective, init_pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
            
        # Try Powell as well if time permits?
        if time.time() < deadline - 10.0:
            try:
                res_powell = minimize(objective, init_pts.flatten(), method='Powell', 
                           options={'maxiter': 1000, 'xtol': 1e-9, 'ftol': 1e-11})
                if -res_powell.fun > best_score:
                    best_score = -res_powell.fun
                    best_pts = res_powell.x.reshape(-1, 2)
            except Exception:
                pass

    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
