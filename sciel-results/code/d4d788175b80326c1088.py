"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11
S3 = np.sqrt(3.0)
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])

# Precompute all triangle index combinations
TRIS = list(itertools.combinations(range(N), 3))
NTRIS = len(TRIS)  # 165

def barycentric_to_cartesian(params):
    """Convert 2*N barycentric coordinates (last coord = 1 - sum of first two) to Cartesian."""
    # params shape: (2*N,) -> for each point, (u, v) where w = 1-u-v
    pts = np.zeros((N, 2))
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        w = 1.0 - u - v
        pts[i] = u * V1 + v * V2  # since V0 is origin
    return pts

def compute_min_area(pts):
    """Compute minimum triangle area for given points."""
    min_a = np.inf
    for (i, j, k) in TRIS:
        cross = (pts[j][0]-pts[i][0])*(pts[k][1]-pts[i][1]) - (pts[j][1]-pts[i][1])*(pts[k][0]-pts[i][0])
        area = 0.5 * abs(cross)
        if area < min_a:
            min_a = area
    return min_a

def soft_min_objective(params, tau=0.01):
    """
    Soft-min surrogate: minimize -logsumexp(-areas/tau) which approximates -min(areas).
    Returns the negative soft-min (to minimize).
    """
    pts = barycentric_to_cartesian(params)
    # Check if all points are inside (with margin)
    # In barycentric coords: u>=0, v>=0, u+v<=1
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        if u < -1e-6 or v < -1e-6 or u + v > 1.0 + 1e-6:
            return 1e10  # penalty
    
    areas = np.zeros(NTRIS)
    for idx, (i, j, k) in enumerate(TRIS):
        cross = (pts[j][0]-pts[i][0])*(pts[k][1]-pts[i][1]) - (pts[j][1]-pts[i][1])*(pts[k][0]-pts[i][0])
        areas[idx] = 0.5 * abs(cross)
    
    # Soft min: -logsumexp(-areas/tau)
    neg_areas = -areas / tau
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -log_sum * tau  # This approximates min(areas)
    
    # We want to maximize min area, so minimize -soft_min_val
    return -soft_min_val

def hard_objective(params):
    """Exact min area objective for final polish (minimize negative min area)."""
    pts = barycentric_to_cartesian(params)
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        if u < -1e-9 or v < -1e-9 or u + v > 1.0 + 1e-9:
            return 1e10
    min_a = compute_min_area(pts)
    return -min_a

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Bounds for barycentric coordinates: u in [0,1], v in [0,1], u+v <= 1
    # We use u in [0,1], v in [0, 1-u] but DE needs rectangular bounds
    # Use u in [0,1], v in [0,1] and rely on penalty/clip
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    # Initial best
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution with soft-min surrogate
    # Use multiple restarts with different tau values
    tau_values = [0.1, 0.05, 0.02, 0.01]
    
    for tau in tau_values:
        if time.time() > deadline - 30.0:
            break
        
        def de_objective(params):
            return soft_min_objective(params, tau=tau)
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            
            if result.success or result.fun < 1e9:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: random placement
        best_params = np.random.uniform(0, 1, size=2*N)
        # Clip to valid region
        for i in range(N):
            u = best_params[2*i]
            v = best_params[2*i+1]
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                best_params[2*i] *= scale
                best_params[2*i+1] *= scale
        best_score = -hard_objective(best_params)
    
    # Phase 2: Nelder-Mead polish with exact objective
    if time.time() < deadline - 10.0:
        try:
            nm_result = minimize(
                hard_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_result.fun < 1e9:
                score = -nm_result.fun
                if score > best_score:
                    best_score = score
                    best_params = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional NM restarts from perturbed best
    if time.time() < deadline - 5.0:
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            perturbed = best_params + np.random.normal(0, 0.01, size=2*N)
            # Clip to valid region
            for i in range(N):
                u = perturbed[2*i]
                v = perturbed[2*i+1]
                if u < 0: perturbed[2*i] = 0
                if v < 0: perturbed[2*i+1] = 0
                if u + v > 1.0:
                    scale = 1.0 / (u + v)
                    perturbed[2*i] *= scale
                    perturbed[2*i+1] *= scale
            
            try:
                nm2 = minimize(
                    hard_objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if nm2.fun < 1e9:
                    score = -nm2.fun
                    if score > best_score:
                        best_score = score
                        best_params = nm2.x.copy()
            except Exception:
                pass
    
    # Final: clip and return Cartesian points
    final_params = best_params.copy()
    for i in range(N):
        u = final_params[2*i]
        v = final_params[2*i+1]
        if u < 0: final_params[2*i] = 0
        if v < 0: final_params[2*i+1] = 0
        if u + v > 1.0:
            scale = 1.0 / (u + v)
            final_params[2*i] *= scale
            final_params[2*i+1] *= scale
    
    pts = barycentric_to_cartesian(final_params)
    
    # Ensure points are strictly inside (tiny margin from edges)
    # This is already handled by barycentric coords being in [0,1]
    
    return pts

