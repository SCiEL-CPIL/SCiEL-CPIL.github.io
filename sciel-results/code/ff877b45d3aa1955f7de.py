import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of areas
    # To avoid O(n^3) memory blowup, we can loop over i, j and vectorize k
    # Or simply use itertools if n is small, but n=13 is small enough for vectorization tricks.
    
    min_area = float('inf')
    # Precompute points
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            # Cross product magnitude
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha):
    """
    Smooth approximation of the minimum triangle area.
    Returns: -1/alpha * log(sum(exp(-alpha * areas)))
    This is a lower bound on the min area, becoming tight as alpha -> inf.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    log_sum = 0.0
    # We need to compute log(sum(exp(-alpha * A_k)))
    # To avoid underflow/overflow, use log-sum-exp trick:
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    # Here x_i = -alpha * A_i.
    # max(x_i) = -alpha * min(A_i).
    
    # We iterate to accumulate the log-sum.
    # Since we can't easily vectorize the log-sum over all triples without huge memory,
    # we loop.
    
    min_area_so_far = float('inf')
    # First pass to find min area for stability shift? 
    # Or just accumulate log-sum-exp carefully.
    
    # Let's use a stable accumulation.
    # We want S = sum(exp(-alpha * A_k))
    # log(S) = ?
    
    # Let M = min(A_k). Then exp(-alpha * A_k) = exp(-alpha * M) * exp(-alpha * (A_k - M))
    # S = exp(-alpha * M) * sum(exp(-alpha * (A_k - M)))
    # log(S) = -alpha * M + log(sum(exp(-alpha * (A_k - M))))
    
    # We can compute M and the sum in one pass if we store the terms? 
    # Or do two passes. Two passes is safer for stability.
    
    # Pass 1: Find M
    M = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            local_min = np.min(areas)
            if local_min < M:
                M = local_min
    
    if M == float('inf'):
        return 0.0
        
    # Pass 2: Compute log(sum(exp(-alpha * (A_k - M))))
    log_sum_shifted = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            shifted_areas = areas - M
            # exp(-alpha * shifted_areas)
            # Since shifted_areas >= 0, exp is <= 1. No overflow.
            terms = np.exp(-alpha * shifted_areas)
            log_sum_shifted += np.log(np.sum(terms)) # This is wrong, log(sum) != sum(log)
            
    # Correction: We need log(sum(terms)).
    # Let's just collect all terms in a list? N=13, C(13,3)=286 terms. This is small.
    # We can just build a vector of all areas.
    
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            areas_list.extend(areas)
            
    areas_arr = np.array(areas_list)
    if len(areas_arr) == 0:
        return 0.0
        
    M = np.min(areas_arr)
    shifted = areas_arr - M
    log_sum = -alpha * M + np.log(np.sum(np.exp(-alpha * shifted)))
    
    return -log_sum / alpha

def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    min_area_soft = _soft_min_triangle_area(pts, alpha)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return -(min_area_soft / hull_a)

def _generate_initializations():
    """Generate diverse starting points."""
    inits = []
    rng = np.random.RandomState(123)
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular Polygon with slight jitter
    for jitter in [0.05, 0.1, 0.2]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # 3. Concentric Rings
    # 7 outer, 6 inner
    m1 = 7
    m2 = 6
    ang1 = np.linspace(0, 2 * np.pi, m1, endpoint=False)
    ang2 = np.linspace(0, 2 * np.pi, m2, endpoint=False) + np.pi/m2
    r1 = 1.0
    r2 = 0.5
    ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    inits.append(np.vstack([ring1, ring2]))
    
    # 4. 12-gon + Center
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # 5. Random points in unit disk
    for _ in range(10):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like
    xs = np.linspace(-0.5, 0.5, 4)
    ys = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([xx.flatten(), yy.flatten()])
    # Select 13 points from 16
    for i in range(3):
        idx = rng.choice(16, 13, replace=False)
        inits.append(pts_grid[idx])

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    inits = _generate_initializations()
    
    # Evaluate initializations
    for init_pts in inits:
        if time.time() > deadline - 30.0:
            break
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
            
    # Optimization Loop
    # We will try to refine each initialization (or the best few)
    # Strategy:
    # 1. Soft-min optimization with annealing alpha
    # 2. Exact Nelder-Mead polish
    
    candidates = [best_pts]
    # Add top 5 initializations
    scored_inits = [(_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: -x[0])
    for s, p in scored_inits[:5]:
        candidates.append(p)
        
    # Remove duplicates
    unique_candidates = []
    for c in candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)
            
    for init_pts in unique_candidates:
        if time.time() > deadline - 15.0:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Step 1: Soft-min Annealing
        alphas = [10.0, 50.0, 100.0, 200.0]
        for alpha in alphas:
            if time.time() > deadline - 10.0:
                break
            try:
                # Use L-BFGS-B on the soft score
                # Note: _neg_soft_score needs alpha as arg, but minimize passes only x
                # We use a closure or global var? Closure is cleaner.
                def _neg_soft_score_closure(params, a=alpha):
                    return _neg_soft_score(params, a)
                    
                res = minimize(
                    _neg_soft_score_closure,
                    current_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                soft_pts = res.x.reshape(-1, 2)
                soft_score = _score(soft_pts) # Check true score
                if soft_score > current_score:
                    current_pts = soft_pts
                    current_score = soft_score
            except Exception:
                pass
        
        # Step 2: Exact Polish with Nelder-Mead
        if time.time() < deadline - 5.0:
            try:
                res_exact = minimize(
                    _neg_score,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                exact_pts = res_exact.x.reshape(-1, 2)
                exact_score = _score(exact_pts)
                if exact_score > current_score:
                    current_pts = exact_pts
                    current_score = exact_score
            except Exception:
                pass
                
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    # Step 3: Differential Evolution for Global Search
    if time.time() < deadline - 20.0:
        bounds = [(-2, 2)] * (2 * N)
        try:
            # Use a smaller population and fewer iterations to fit in time
            res_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=True # Runs L-BFGS-B at end
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
            
    # Final Polish
    if time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
