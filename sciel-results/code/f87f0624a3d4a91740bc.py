import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
# i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside(points):
    """
    Check if all points are inside the equilateral triangle (0,0), (1,0), (0.5, H).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Check y >= 0
    if np.any(y < -1e-9):
        return False
    
    # Check y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
        
    # Check y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective(params):
    """
    Objective function for minimization.
    Returns -min_area if valid, penalty if invalid.
    """
    points = params.reshape(N, 2)
    
    if not is_inside(points):
        return 1.0  # Penalty: larger than any possible area
        
    area = min_triangle_area_vec(points)
    return -area

def refine_local(points, deadline):
    """
    Perform local refinement using Nelder-Mead on the best point set.
    Returns the refined point set.
    """
    if time.time() > deadline:
        return points
    
    best_points = points.copy()
    best_score = min_triangle_area_vec(best_points)
    
    # Try a few local optimizations with different initial perturbations
    # This helps escape shallow local minima found by DE
    rng = np.random.default_rng(42)
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        # Create a slightly perturbed version to start from
        if attempt == 0:
            x0 = best_points.flatten()
        else:
            # Small random perturbation
            # Scale perturbation with remaining time? No, keep it simple but vary magnitude
            scale = 0.005 * (attempt + 1)
            perturbation = rng.normal(0, scale, size=(N, 2))
            x0 = (best_points + perturbation).flatten()
            
            # Ensure validity by checking bounds roughly
            # If invalid, skip this perturbation
            if not is_inside(x0.reshape(N, 2)):
                continue
        
        try:
            # Use adaptive Nelder-Mead with more iterations as per observed improvement
            # maxiter=1000 and adaptive=True were noted to help escape local minima
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            
            if time.time() > deadline:
                break
                
            new_points = res.x.reshape(N, 2)
            if is_inside(new_points):
                new_score = min_triangle_area_vec(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
        except Exception:
            continue
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Bounds for Cartesian coordinates
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search
    try:
        # Tune maxiter and popsize. 
        # The previous program used maxiter=200, popsize=15.
        # Inspiration 1 used maxiter=1000, popsize=20 but had a longer timeout or less local refinement.
        # We need to balance DE exploration with local refinement.
        # Let's give DE a moderate budget, then use the remaining time for aggressive local refinement.
        # 145s total. 
        # Let's try maxiter=300, popsize=15. This should take ~30-60s depending on speed.
        result = differential_evolution(
            objective,
            bounds,
            maxiter=300, 
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,  # We will do our own refined polish
            init='sobol' 
        )
        
        best_params = result.x
        candidate_points = best_params.reshape(N, 2)
        
        if is_inside(candidate_points):
            candidate_score = min_triangle_area_vec(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
    except Exception:
        pass
        
    # If DE didn't find a valid point or failed, try a few random restarts
    if best_points is None:
        # Fallback to a simple valid configuration
        rng = np.random.default_rng(42)
        for _ in range(100):
            # Sample in bounding box
            pts = rng.random((N, 2))
            pts[:, 0] *= 1.0
            pts[:, 1] *= H
            if is_inside(pts):
                best_points = pts
                best_score = min_triangle_area_vec(best_points)
                break
        if best_points is None:
            # Last resort: grid points
            # Create a valid grid
            xs = np.linspace(0.1, 0.9, 3)
            ys = np.linspace(0.1, 0.5, 4)
            candidates = []
            for x in xs:
                for y in ys:
                    if is_inside(np.array([[x, y]])):
                        candidates.append([x, y])
            # Take first 11
            if len(candidates) >= N:
                best_points = np.array(candidates[:N])
                best_score = min_triangle_area_vec(best_points)
            else:
                # Generate random valid points
                while len(candidates) < N:
                    p = rng.random(2)
                    p[0] *= 1.0
                    p[1] *= H
                    if is_inside(p.reshape(1, 2)):
                        candidates.append(p)
                best_points = np.array(candidates[:N])
                best_score = min_triangle_area_vec(best_points)
    
    # Check time before local refinement
    if time.time() < deadline - 5:
        # Perform local refinement to squeeze out more performance
        # This is where the structural change happens: more aggressive, adaptive local search
        best_points = refine_local(best_points, deadline)
    
    return best_points
