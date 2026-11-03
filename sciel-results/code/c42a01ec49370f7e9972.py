import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])

# Barycentric coordinates transformation
# A point P = u*V1 + v*V2 + w*V3, where u+v+w=1, u,v,w >= 0
# We can represent a point by (u, v), with w = 1 - u - v
# To ensure validity, we project (u, v) to the simplex u>=0, v>=0, u+v<=1

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    w is implicit: 1 - u - v.
    Input: (N, 2) array of (u, v).
    Output: (N, 2) array of (x, y).
    """
    # P = u*V1 + v*V2 + w*V3
    # V1 = (0,0), V2 = (1,0), V3 = (0.5, sqrt(3)/2)
    # x = 0*u + 1*v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = 0*u + 0*v + (sqrt(3)/2)*w = (sqrt(3)/2)*(1-u-v)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    x = v + 0.5 * w
    y = (SQRT3 / 2.0) * w
    
    return np.column_stack((x, y))

def _project_to_simplex(bary):
    """
    Project barycentric coordinates to the valid simplex region.
    Input: (N, 2) array of (u, v).
    Output: (N, 2) array of projected (u, v).
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Simple projection: clip negatives to 0, then normalize if sum > 1
    # This is a heuristic projection, not exact Euclidean projection, 
    # but sufficient for DE to stay in bounds.
    # More robust: if u<0 or v<0 or u+v>1, reflect or clip.
    # For DE, bounds are usually handled by the algorithm, but we enforce 
    # strict validity for the objective function.
    
    # We will use scipy's bounds in DE, so this function is mainly for 
    # safety if we do manual steps or final polish.
    # However, for the objective function inside DE, we assume bounds are respected.
    
    # If we need a hard projection for Nelder-Mead:
    # If u < 0: u = 0
    # If v < 0: v = 0
    # If u + v > 1: scale down or project to edge.
    
    # Let's implement a simple clip-and-normalize for safety in local search
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    # Ensure u + v <= 1
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        # Scale down to boundary
        factor = 1.0 / s[mask]
        u[mask] *= factor
        v[mask] *= factor
        
    return np.column_stack((u, v))

def _min_triangle_area_cartesian(points):
    """
    Calculate the minimum area of any triangle formed by N points.
    points: (N, 2) array in Cartesian coordinates.
    Returns: float, the minimum area.
    """
    n = len(points)
    min_area = np.inf
    
    # Vectorized calculation for speed
    # Number of triples is C(11, 3) = 165.
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            v_ij = pj - pi
            # Vector from i to all k > j
            pk_set = points[j+1:]
            v_ik_set = pk_set - pi
            # Cross product: v_ij[0]*v_ik[1] - v_ij[1]*v_ik[0]
            crosses = v_ij[0] * v_ik_set[:, 1] - v_ij[1] * v_ik_set[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _objective_de(bary_flat):
    """
    Objective function for Differential Evolution.
    Input: 1D array of length 2*N (u1, v1, u2, v2, ...)
    Returns: -min_area (to be minimized)
    """
    bary = bary_flat.reshape(N, 2)
    # DE respects bounds, but let's be safe
    bary = _project_to_simplex(bary)
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_cartesian(points)
    return -min_area

def _objective_nm(bary_flat):
    """
    Objective function for Nelder-Mead.
    Input: 1D array of length 2*N.
    Returns: -min_area + penalty if out of bounds (though we project, 
             so we might just return -min_area of projected points).
             Actually, for NM, it's better to penalize going out of bounds 
             to guide it back, or just project. Projecting makes the 
             landscape flat outside the boundary, which can confuse NM.
             Let's use a soft penalty.
    """
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for violating simplex constraints
    penalty = 0.0
    # u >= 0, v >= 0, u + v <= 1
    neg_u = np.sum(np.clip(-u, 0, None))
    neg_v = np.sum(np.clip(-v, 0, None))
    over_sum = np.sum(np.clip(u + v - 1.0, 0, None))
    
    penalty = 100.0 * (neg_u + neg_v + over_sum)
    
    # Project for evaluation
    bary_proj = _project_to_simplex(bary)
    points = _bary_to_cart(bary_proj)
    min_area = _min_triangle_area_cartesian(points)
    
    return -min_area + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Bounds for barycentric coordinates (u, v)
    # u in [0, 1], v in [0, 1]
    # The constraint u+v<=1 is handled by projection/penalty, 
    # but DE bounds are rectangular. 
    # To help DE, we could restrict v to [0, 1-u], but DE doesn't support 
    # correlated bounds directly. We rely on the objective function 
    # (penalty or projection) and the fact that the valid region is 
    # a large part of the unit square.
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_score = -np.inf
    best_bary = None
    
    # Phase 1: Differential Evolution
    # DE is stochastic and global.
    # Strategy: 'best1bin' or 'best1exp'
    # Maxiter and popsize need to be tuned for time budget.
    
    # We will run DE with a time limit check inside a wrapper if possible,
    # or just set maxiter conservatively.
    # 11 points -> 22 dimensions.
    # Popsize 15 (default is 15*dim = 330, which is too many for 22 dim? 
    # Actually popsize=15 means 15*22=330 individuals. 
    # Each eval is ~165 triangle area calcs. 330 * 165 * 22 iterations?
    # Let's estimate cost. 
    # One eval: ~1000 flops. 
    # 330 evals * 50 iterations = 16500 evals.
    # 16500 * 1000 flops = 1.65e7 flops. Very fast.
    # We can afford more iterations or larger population.
    
    try:
        # Use DE to find a good starting point
        # seed=42 for reproducibility
        de_result = differential_evolution(
            _objective_de, 
            bounds, 
            strategy='best1bin', 
            maxiter=100, 
            popsize=10, # 10*22 = 220 individuals
            tol=1e-6,
            seed=42,
            init='latinhypercube'
        )
        
        if time.time() < deadline - 20.0:
            current_bary = de_result.x
            current_score = -de_result.fun
            
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
                
    except Exception as e:
        # Fallback to random if DE fails
        pass
    
    # If DE didn't run or found nothing, initialize randomly
    if best_bary is None:
        # Random valid points
        rng = np.random.default_rng(42)
        best_bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                if u + v <= 1.0:
                    best_bary[i] = [u, v]
                    break
        best_points = _bary_to_cart(best_bary)
        best_score = _min_triangle_area_cartesian(best_points)

    # Phase 2: Local Polish with Nelder-Mead
    # NM is derivative-free and good for non-smooth functions if started close.
    # We use the best point from DE as the initial simplex center.
    
    x0 = best_bary.flatten()
    
    # We can try multiple NM restarts from slightly perturbed versions of the best DE solution
    # or just one strong NM run.
    # Given the time budget, one strong NM run is safer.
    
    try:
        nm_result = minimize(
            _objective_nm, 
            x0, 
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if time.time() < deadline - 5.0:
            new_bary = nm_result.x
            # Project to ensure validity
            new_bary = _project_to_simplex(new_bary.reshape(N, 2)).flatten()
            
            # Evaluate
            new_points = _bary_to_cart(new_bary.reshape(N, 2))
            new_score = _min_triangle_area_cartesian(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary.copy()
                
    except Exception:
        pass

    # Final verification and conversion
    final_bary = _project_to_simplex(best_bary.reshape(N, 2))
    final_points = _bary_to_cart(final_bary)
    
    # Double check validity
    # Points should be inside the triangle.
    # Barycentric coords are u, v, w=1-u-v.
    # If u,v,w >= 0, point is inside.
    # _project_to_simplex ensures u>=0, v>=0, u+v<=1 => w>=0.
    
    return final_points

# Quick self-test
if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    area = _min_triangle_area_cartesian(pts)
    print(f"Time: {elapsed:.2f}s, Min Area: {area:.6f}")
    print(f"Points:\n{pts}")
