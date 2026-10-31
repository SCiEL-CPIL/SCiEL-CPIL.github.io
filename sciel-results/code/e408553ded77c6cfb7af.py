"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
SQRT3 = np.sqrt(3)
TRI_AREA = 0.5 * 1.0 * (SQRT3 / 2) # 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4

# Precompute indices for all triangles
COMBOS = list(combinations(range(N), 3))
COMBO_INDICES = np.array(COMBOS) # shape (165, 3)

def _clamp_to_triangle_vectorized(pts):
    """
    Clamp an array of points to the equilateral triangle.
    Uses barycentric coordinate projection onto the simplex.
    pts: shape (n, 2)
    """
    x, y = pts[:, 0], pts[:, 1]
    
    # Calculate barycentric coordinates
    # p = a*V0 + b*V1 + c*V2
    # x = b + 0.5*c
    # y = (sqrt(3)/2)*c  => c = 2*y/sqrt(3)
    c = 2 * y / SQRT3
    b = x - 0.5 * c
    a = 1 - b - c
    
    # Project (a, b, c) onto the probability simplex a,b,c >= 0, a+b+c=1
    # Algorithm for 3D:
    # 1. Sort v in decreasing order
    # 2. Find rho such that sum_{i<=rho} (v_i - tau) = 1
    # 3. v_i = max(0, v_i - tau)
    
    # We do this for each point. Since n is small (11), we can loop, 
    # but vectorizing the projection is also possible. 
    # For simplicity and robustness, we loop over points here as n=11 is tiny.
    
    clamped_pts = np.zeros_like(pts)
    
    for i in range(len(pts)):
        v = np.array([a[i], b[i], c[i]])
        
        # Sort v in decreasing order
        sorted_v = np.sort(v)[::-1]
        
        # Find rho
        # rho is the largest index k such that sum(sorted_v[:k]) - k + 1 > 0
        # Actually, the condition is: (sum(sorted_v[:k]) - 1) / k < sorted_v[k]
        # Let's just try k=3, 2, 1
        
        tau = 0.0
        proj = np.zeros(3)
        
        for k in range(3, 0, -1):
            tau_candidate = (np.sum(sorted_v[:k]) - 1.0) / k
            # Check if the k-th element is greater than tau
            if sorted_v[k-1] > tau_candidate:
                tau = tau_candidate
                # Apply projection to original v
                proj = np.maximum(0, v - tau)
                break
        
        # If no k worked (shouldn't happen for valid inputs), fallback
        if np.all(proj == 0):
             proj = np.array([1/3, 1/3, 1/3])
        
        # Convert back to Cartesian
        # P = a*V0 + b*V1 + c*V2
        # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
        # x = b + 0.5*c
        # y = sqrt(3)/2 * c
        
        # proj is [a, b, c]
        aa, bb, cc = proj
        clamped_pts[i, 0] = bb + 0.5 * cc
        clamped_pts[i, 1] = (SQRT3 / 2) * cc
        
    return clamped_pts

def _min_area_vectorized(pts):
    """
    Compute the minimum area of any triangle formed by three of the points.
    pts: shape (n, 2)
    """
    # Get points for all combinations
    # COMBO_INDICES shape (165, 3)
    A = pts[COMBO_INDICES[:, 0]]
    B = pts[COMBO_INDICES[:, 1]]
    C = pts[COMBO_INDICES[:, 2]]
    
    # Calculate areas
    # Area = 0.5 * | (B_x - A_x)(C_y - A_y) - (B_y - A_y)(C_x - A_x) |
    cross = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x):
    """
    Objective function for optimization.
    x: flat array of 2*N coordinates.
    Returns negative min area (to be minimized).
    """
    pts = x.reshape(N, 2)
    # Clamp to triangle
    pts_clamped = _clamp_to_triangle_vectorized(pts)
    
    # Calculate min area
    min_a = _min_area_vectorized(pts_clamped)
    
    # We want to maximize min_a, so minimize -min_a
    return -min_a

def _initial_placement_random(rng):
    """Generate a random valid initial placement."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u * 1.0 + v * 0.5
        y = v * SQRT3 / 2
        pts.append([x, y])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1
    
    # Define bounds for Differential Evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, SQRT3 / 2)) # y
    
    # Run Differential Evolution
    # DE is stochastic and global. 
    # maxiter: number of generations. 
    # popsize: population size multiplier.
    # We monitor time inside the objective? No, DE doesn't support easy time limits directly.
    # We will run a fixed number of iterations and check time.
    
    # Strategy: Run DE for a fixed time budget, then polish.
    # Since we can't easily interrupt DE, we estimate iterations.
    # Each objective call is fast (vectorized). 
    # Let's try maxiter=200, popsize=15. 
    # If it takes too long, we might timeout. 
    # To be safe, we can run DE with a smaller maxiter and multiple restarts? 
    # Or just one long run. 
    # Let's try maxiter=100 first.
    
    try:
        # Use 'best1bin' strategy
        res_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=150, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True, # DE has a built-in polish step (Nelder-Mead)
            init='latinhypercube'
        )
        
        if res_de.success or res_de.fun < -1e-10: # Check if it found something
            final_pts_de = res_de.x.reshape(N, 2)
            final_pts_de = _clamp_to_triangle_vectorized(final_pts_de)
            score_de = _min_area_vectorized(final_pts_de)
            
            if score_de > best_score:
                best_score = score_de
                best_points = final_pts_de.copy()
                
    except Exception as e:
        pass # Fallback to random if DE fails
    
    # Check time
    if time.time() > deadline - 20:
        if best_points is not None:
            return best_points
        else:
            return _initial_placement_random(rng)
    
    # Local Polish with Nelder-Mead on the best result from DE
    if best_points is not None:
        current_pts = best_points.flatten()
        current_score = best_score
        
        try:
            res_nm = minimize(
                _objective, 
                current_pts, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            final_pts_nm = res_nm.x.reshape(N, 2)
            final_pts_nm = _clamp_to_triangle_vectorized(final_pts_nm)
            score_nm = _min_area_vectorized(final_pts_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = final_pts_nm.copy()
        except Exception:
            pass
        
    # If DE failed or didn't run, do a few random restarts with NM
    if best_points is None:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            init_pts = _initial_placement_random(rng)
            try:
                res = minimize(
                    _objective, 
                    init_pts.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                final_pts = res.x.reshape(N, 2)
                final_pts = _clamp_to_triangle_vectorized(final_pts)
                score = _min_area_vectorized(final_pts)
                if score > best_score:
                    best_score = score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        return _initial_placement_random(rng)
    
    return best_points

# EVOLVE-BLOCK-END
