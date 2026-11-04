"""
Strategy: Direct coordinate optimization of 14 points with symmetry breaking.
Instead of parametric ring families, we treat all 28 coordinates as free variables.
We use a robust global search (Differential Evolution) followed by a local
polish (L-BFGS-B) to maximize the minimum triangle area relative to the convex hull area.
"""

N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _objective(params):
    """
    Objective function for optimization.
    Returns negative score to maximize.
    """
    pts = params.reshape((N, 2))
    
    # Basic validity check: points should be reasonably spread
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e9
        
    # Check for degenerate hull (collinear points)
    # A quick check: if all points are very close, score is bad
    if np.allclose(pts, pts[0]):
        return 1e9

    score = _score(pts)
    if score <= 0:
        return 1e9
        
    return -score

def _init_regular_polygon(radius=1.0):
    """
    Initialize points as a regular 14-gon.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y]).flatten()

def _optimize_direct(deadline):
    """
    Optimize point coordinates directly.
    """
    # 1. Global Search using Differential Evolution
    # Bounds: keep points within a box of [-1.5, 1.5] to avoid numerical issues
    # while allowing enough space for the convex hull to form.
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    # We use a moderate population size and iterations to balance exploration and time.
    # The objective function is expensive (O(N^3) roughly), so we limit iterations.
    # However, DE is robust.
    
    start_time = time.time()
    
    # We can try multiple random restarts if time permits, but DE itself is stochastic.
    # Let's run one DE instance with a decent budget.
    
    # Time budget allocation:
    # - DE: ~60s
    # - Local Polish: ~30s
    # - Safety: rest
    
    de_time_limit = 60.0
    local_time_limit = 30.0
    
    # We need to be careful with DE's maxiter and popsize.
    # popsize=10, maxiter=50 -> 500 evaluations.
    # Each eval is ~1-5ms? _score is fast for N=14.
    # Let's estimate: 364 triangles. Vectorized loops are fast.
    # Let's try popsize=15, maxiter=40.
    
    try:
        res_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=40, 
            popsize=15, 
            tol=1e-4, 
            seed=42, 
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        best_params = res_de.x
        best_score = -res_de.fun
        
    except Exception as e:
        # Fallback to regular polygon if DE fails
        best_params = _init_regular_polygon()
        best_score = _score(best_params.reshape((N, 2)))
        
    # 2. Local Polish using L-BFGS-B
    # L-BFGS-B is good for smooth-ish problems, but min area is non-smooth.
    # We can use Nelder-Mead or L-BFGS-B with a smoothed objective?
    # Actually, for this specific task, a simple coordinate ascent or Nelder-Mead
    # on the raw points often works well if started from a good global solution.
    
    # Let's use Nelder-Mead for the local polish as it doesn't require gradients
    # and can handle the non-smoothness of the min operation better than L-BFGS-B
    # which might get stuck on plateaus.
    
    if time.time() < deadline - 10.0:
        # Run Nelder-Mead
        # Initial step size needs to be reasonable.
        # The scale of coordinates is ~1.
        
        def neg_score_pts(params):
            pts = params.reshape((N, 2))
            return -_score(pts)
            
        try:
            res_nm = minimize(
                neg_score_pts, 
                best_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True}
            )
            
            if -res_nm.fun > best_score:
                best_params = res_nm.x
                best_score = -res_nm.fun
                
        except Exception:
            pass
            
        # Optional: Another round of NM with smaller step or different start if time permits
        if time.time() < deadline - 5.0:
            # Try a few random perturbations of the best solution and polish
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                
                # Perturb
                noise = np.random.normal(0, 0.05, size=best_params.shape)
                perturbed = best_params + noise
                
                try:
                    res_nm2 = minimize(
                        neg_score_pts, 
                        perturbed, 
                        method='Nelder-Mead', 
                        options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 100, 'adaptive': True}
                    )
                    
                    if -res_nm2.fun > best_score:
                        best_params = res_nm2.x
                        best_score = -res_nm2.fun
                        
                except Exception:
                    pass

    # Final verification
    final_pts = best_params.reshape((N, 2))
    final_score = _score(final_pts)
    
    # Ensure we return the best found
    if final_score < best_score:
        # This shouldn't happen, but just in case
        final_score = best_score
        
    return final_pts, final_score

def construct():
    start_time = time.time()
    # 150s limit. Use 140s deadline to be safe.
    deadline = start_time + 140.0
    
    # Direct coordinate optimization
    pts, score = _optimize_direct(deadline)
    
    # Fallback: If score is very low, try a simple regular polygon
    if score < 1.0:
        reg_pts = _init_regular_polygon().reshape((N, 2))
        reg_score = _score(reg_pts)
        if reg_score > score:
            pts = reg_pts
            score = reg_score
            
    return pts

# EVOLVE-BLOCK-END
