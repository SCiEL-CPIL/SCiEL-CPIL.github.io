import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian.
    bary shape: (N, 2) where col 0 is u, col 1 is v.
    w0 = 1 - u - v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w0 = 1.0 - u - v
    x = w0 * V0[0] + u * V1[0] + v * V2[0]
    y = w0 * V0[1] + u * V1[1] + v * V2[1]
    return np.column_stack((x, y))

def compute_min_area_bary(bary_params):
    """
    bary_params: flat array of shape (2*N,)
    Returns negative min area (to be minimized).
    """
    # Reshape to (N, 2)
    bary = bary_params.reshape(N, 2)
    
    # Check validity (should be handled by bounds/constraints, but good to check for numerical stability)
    # If any point is outside, return a large penalty
    u = bary[:, 0]
    v = bary[:, 1]
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1 + 1e-9):
        return 1e6
        
    # Construct full barycentric matrix (N, 3)
    # Col 0: 1 - u - v
    # Col 1: u
    # Col 2: v
    B = np.empty((N, 3))
    B[:, 0] = 1.0 - u - v
    B[:, 1] = u
    B[:, 2] = v
    
    # Calculate areas for all triangles
    # tri_idx is (165, 3)
    # We need det of 3x3 matrix formed by rows i, j, k of B
    # B[i], B[j], B[k]
    
    # Vectorized determinant calculation for 3x3 matrices
    # Let M be (165, 3, 3)
    # M[:, 0, :] = B[tri_idx[:, 0]]
    # M[:, 1, :] = B[tri_idx[:, 1]]
    # M[:, 2, :] = B[tri_idx[:, 2]]
    
    row0 = B[tri_idx[:, 0]]
    row1 = B[tri_idx[:, 1]]
    row2 = B[tri_idx[:, 2]]
    
    # Determinant of [[a,b,c], [d,e,f], [g,h,i]] = a(ei-fh) - b(di-fg) + c(dh-eg)
    # Here rows are vectors.
    # a = row0[:, 0], b = row0[:, 1], c = row0[:, 2]
    # d = row1[:, 0], e = row1[:, 1], f = row1[:, 2]
    # g = row2[:, 0], h = row2[:, 1], i = row2[:, 2]
    
    a, b, c = row0[:, 0], row0[:, 1], row0[:, 2]
    d, e, f = row1[:, 0], row1[:, 1], row1[:, 2]
    g, h, i = row2[:, 0], row2[:, 1], row2[:, 2]
    
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    
    areas = np.abs(det)
    
    min_area = np.min(areas)
    
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Objective function wrapper for minimize
    def objective(x):
        return compute_min_area_bary(x)
    
    # Define bounds for barycentric parameters (u, v)
    # u >= 0, v >= 0, u+v <= 1
    # We will use simple bounds for u and v in [0, 1] and rely on the penalty
    # or better, use a transformation?
    # Actually, Nelder-Mead doesn't support bounds natively in older SciPy versions,
    # but we can use the penalty method or just clip/penalize.
    # The penalty method is robust here.
    
    best_val = -1e6
    best_params = np.zeros(2 * N)
    
    # Generate initial points
    # We will use a combination of random and structured starts.
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. A few random starts
    # 2. A structured grid start (e.g., center of triangle)
    # 3. Refinement of the best found so far
    
    num_starts = 20  # We will adjust this based on time
    
    # Pre-generate starts
    starts = []
    
    # Center of triangle in barycentric: (1/3, 1/3)
    center = np.tile([1/3, 1/3], N)
    starts.append(center)
    
    # Random starts
    for _ in range(num_starts - 1):
        # Sample u, v such that u>=0, v>=0, u+v<=1
        # Method: sample u, v from uniform [0,1], if u+v>1, reject or transform.
        # Better: sample t1, t2 from uniform [0,1].
        # u = t1 * (1 - t2)
        # v = t2 * (1 - t1) ? No.
        # Standard method: 
        # r1, r2 ~ U(0,1)
        # if r1 + r2 > 1: r1, r2 = 1-r1, 1-r2
        # This maps to the triangle.
        
        u_v = np.zeros((N, 2))
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            u_v[i, 0] = r1
            u_v[i, 1] = r2
            
        starts.append(u_v.flatten())
        
    # Add a "grid-like" start: distribute points roughly evenly
    # Approximate optimal distribution for 11 points is often a hexagonal packing or similar.
    # Let's just add a perturbed version of the center.
    for i in range(5):
        perturbation = rng.normal(0, 0.1, (N, 2))
        p = np.tile([1/3, 1/3], N) + perturbation.flatten()
        # Clamp to valid region roughly
        u = p[0::2]
        v = p[1::2]
        # Simple rejection/clamping for validity in barycentric
        # This is just an initial guess, penalty handles the rest
        starts.append(p)
        
    # Shuffle starts
    rng.shuffle(starts)
    
    # Run optimization
    for i, start in enumerate(starts):
        if time.time() > deadline - 10: # Leave 10s for final check/return
            break
            
        try:
            result = minimize(
                objective, 
                start, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 500, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            val = -result.fun
            if val > best_val:
                best_val = val
                best_params = result.x
        except Exception:
            pass
            
    # Final polish: If we have time, run another round of NM on the best solution
    # with tighter tolerances or just ensure we return the best valid point.
    
    # Convert best barycentric params to Cartesian
    bary_best = best_params.reshape(N, 2)
    
    # Ensure validity (clip tiny negative values)
    bary_best = np.clip(bary_best, 0, 1)
    # Fix sum > 1
    sums = bary_best[:, 0] + bary_best[:, 1]
    mask = sums > 1.0
    if np.any(mask):
        # Scale down
        scale = 1.0 / sums[mask]
        bary_best[mask, 0] *= scale
        bary_best[mask, 1] *= scale
        
    cart_points = bary_to_cart(bary_best)
    
    # Final check: ensure points are strictly inside or on boundary
    # The barycentric conversion guarantees this if inputs are valid.
    
    return cart_points

# EVOLVE-BLOCK-END
