import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# Pre-compute indices for N=14 for fast triangle area calculation
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

def _min_triangle_area(pts):
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

def _soft_min_area(pts, beta):
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
    
    min_a = np.min(areas)
    # Shift to avoid overflow
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            
    # 1. Initial Guesses
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)
    
    # Two-ring symmetric guess (7 inner, 7 outer)
    # Inner radius ~ 0.5, Outer 1.0
    # Angles offset by pi/7
    theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    theta_outer = theta_inner + np.pi / 7
    r_inner = 0.5
    inner_pts = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    two_ring_pts = np.vstack([inner_pts, outer_pts])
    update_best(two_ring_pts)
    
    # 2. Global Optimization with Differential Evolution
    # We optimize the 28 coordinates directly.
    # To keep it fast, we limit maxiter and popsize.
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Optional: Penalize points far outside unit circle to keep hull area bounded?
        # The ratio is scale invariant, so points can fly out. 
        # However, if points fly out, hull area grows quadratically, min area grows quadratically.
        # The ratio stays roughly constant if the shape is preserved.
        # But numerical issues might arise. 
        # Let's constrain to unit disk for stability.
        # Actually, if we constrain to unit disk, we might miss solutions that need to be "flatter" or "longer".
        # But convex region is normalized to unit area. 
        # The evaluator normalizes by the point set convex hull area.
        # So we can place points anywhere. 
        # However, to avoid overflow/underflow, let's scale the points to fit in a reasonable box, say [-10, 10].
        # But DE bounds are important.
        # Let's use bounds [-1, 1] initially. If the optimum is a regular polygon, it fits.
        # If the optimum is elongated, it might be cut off.
        # Let's try bounds [-2, 2].
        
        s = _score(pts)
        return -s

    # Run DE
    if time.time() < deadline - 30.0:
        try:
            # Start from best known point if it's within bounds? No, DE is global.
            # But we can use 'x0' in some implementations? No, DE is population based.
            # We can just run it.
            
            # To speed up, use a smaller population and fewer iterations, but multiple restarts?
            # Or just one big run.
            # N=14 -> 28 dims.
            # popsize=10, maxiter=100 -> 1000 evaluations.
            # Each eval: 364 triangle areas + 1 hull. Fast.
            
            res = differential_evolution(
                objective, 
                bounds=[(-1.0, 1.0) for _ in range(2*N)], 
                seed=42, 
                maxiter=150, 
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol' # Faster initialization than 'latinhypercube' for high dim?
            )
            
            if res.fun < -best_score:
                best_pts = res.x.reshape(-1, 2)
                best_score = -res.fun
                update_best(best_pts)
                
        except Exception:
            pass

    # 3. Local Refinement with L-BFGS-B on Soft Min
    # This helps if DE found a good local optimum but not the exact peak.
    
    def neg_soft_score(x, beta):
        pts = x.reshape(-1, 2)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-9:
            return 1e10
        soft_min = _soft_min_area(pts, beta)
        return -(soft_min / hull_area)

    if best_pts is not None and time.time() < deadline - 10.0:
        # Continuation method for beta
        betas = [20.0, 100.0, 500.0]
        current_pts = best_pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    neg_soft_score,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < -_score(current_pts): # Check if it actually improved the hard score?
                    # Or just check if it improved the soft score?
                    # Better to update if the hard score improves.
                    new_pts = res.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts
                        current_pts = new_pts
                else:
                    # Even if soft score didn't improve hard score immediately, 
                    # it might be closer to the peak. 
                    # Let's just update current_pts to the result of the minimizer 
                    # if it converged, to allow next beta to start better.
                    current_pts = res.x.reshape(-1, 2)
                    
            except Exception:
                pass

    # 4. Final Nelder-Mead Polish
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(
                objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < -best_score:
                best_pts = res.x.reshape(-1, 2)
                best_score = -res.fun
                update_best(best_pts)
        except Exception:
            pass

    if best_pts is None:
        return reg_pts
        
    return best_pts
