import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

# Triangle vertices in cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) to cartesian. bary shape (n,3) with a+b+c=1."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert cartesian points to barycentric coords."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _normalize_bary(bary):
    """Ensure barycentric coords are non-negative and sum to 1."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    bary = bary / s
    return bary

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from cartesian points."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Exact minimum area from cartesian points."""
    return float(np.min(_all_areas_cart(pts)))

def _min_area_bary(bary):
    """Compute min area from barycentric coordinates."""
    cart = _bary_to_cart(_normalize_bary(bary))
    return _min_area_cart(cart)

def _soft_min_area_bary(bary, beta=100.0):
    """Smooth approximation of min area in barycentric coords."""
    cart = _bary_to_cart(_normalize_bary(bary))
    areas = _all_areas_cart(cart)
    # Use log-sum-exp for numerical stability
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _objective_bary(flat_bary, beta=100.0):
    """Objective to minimize: negative soft-min area, working in barycentric space."""
    bary = flat_bary.reshape(-1, 3)
    return -_soft_min_area_bary(bary, beta)

def _objective_hard_bary(flat_bary):
    """Objective to minimize: negative exact min area, working in barycentric space."""
    bary = flat_bary.reshape(-1, 3)
    return -_min_area_bary(bary)

def _objective_de_cart(flat_cart):
    """
    Objective for Differential Evolution.
    Takes flat cartesian coordinates (2*N).
    Clamps to triangle, computes min area, returns negative.
    """
    cart = flat_cart.reshape(N, 2)
    # Convert to barycentric to clamp
    bary = _cart_to_bary(cart)
    bary = _normalize_bary(bary)
    # Convert back to cartesian for area calculation (ensures validity)
    cart_clamped = _bary_to_cart(bary)
    areas = _all_areas_cart(cart_clamped)
    return -np.min(areas)

def _init_structured_bary():
    """
    Structured initialization in barycentric coordinates.
    """
    bary_pts = []
    # Vertices
    bary_pts.append(np.array([1.0, 0.0, 0.0]))
    bary_pts.append(np.array([0.0, 1.0, 0.0]))
    bary_pts.append(np.array([0.0, 0.0, 1.0]))
    
    # Edge midpoints
    bary_pts.append(np.array([0.5, 0.5, 0.0]))
    bary_pts.append(np.array([0.0, 0.5, 0.5]))
    bary_pts.append(np.array([0.5, 0.0, 0.5]))
    
    # Centroid
    bary_pts.append(np.array([1/3, 1/3, 1/3]))
    
    # 4 more points in symmetric positions around centroid
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.15
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        da = r * np.cos(angle)
        db = r * np.sin(angle)
        dc = -da - db
        pt = np.array([cx + da, cy + db, cz + dc])
        pt = _normalize_bary(pt.reshape(1, 3))[0]
        bary_pts.append(pt)
    
    return np.array(bary_pts)

def _simulated_annealing(bary_init, time_left_fn, rng):
    """
    Simulated annealing to escape local minima.
    """
    current = bary_init.copy()
    current_score = _min_area_bary(current)
    best = current.copy()
    best_score = current_score
    
    T_start = 0.05
    T_end = 1e-8
    n_iter = 2000
    cooling = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    
    T = T_start
    for i in range(n_iter):
        if time_left_fn() < 5.0:
            break
        
        step_scale = 0.02 + 0.08 * (T / T_start)
        perturbation = rng.normal(0, step_scale, size=current.shape)
        candidate = current + perturbation
        candidate = _normalize_bary(candidate)
        
        candidate_score = _min_area_bary(candidate)
        delta = candidate_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        T *= cooling
    
    return best, best_score

def _optimize_bary(bary_init, time_left_fn, rng):
    """
    Optimize from a given barycentric initialization using multi-stage refinement.
    Returns (best_bary, best_score).
    """
    best_bary = bary_init.copy()
    best_score = _min_area_bary(best_bary)
    
    # Stage 1: L-BFGS-B with increasing beta
    betas = [50.0, 150.0, 400.0]
    current = best_bary.copy()
    
    for beta in betas:
        if time_left_fn() < 10.0:
            break
        
        try:
            res = minimize(
                _objective_bary,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            opt_bary = _normalize_bary(res.x.reshape(-1, 3))
            score = _min_area_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary.copy()
                current = opt_bary.copy()
        except Exception:
            pass
    
    # Stage 2: Simulated Annealing
    if time_left_fn() > 15.0:
        try:
            sa_bary, sa_score = _simulated_annealing(best_bary, time_left_fn, rng)
            if sa_score > best_score:
                best_score = sa_score
                best_bary = sa_bary.copy()
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard objective
    if time_left_fn() > 5.0:
        try:
            res = minimize(
                _objective_hard_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_bary = _normalize_bary(res.x.reshape(-1, 3))
            nm_score = _min_area_bary(nm_bary)
            
            if nm_score > best_score:
                best_score = nm_score
                best_bary = nm_bary.copy()
        except Exception:
            pass
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin

    def time_left():
        return deadline - time.time()
    
    rng = np.random.default_rng(12345)
    
    best_bary = None
    best_score = -1.0
    
    # Phase 1: Global Search with Differential Evolution
    # We use DE on Cartesian coordinates for simplicity of bounds, 
    # but the objective clamps to the triangle.
    # Bounds: x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    de_results = []
    
    if time_left() > 60.0:
        try:
            # Run DE for a limited time/iterations
            # popsize=15, maxiter=50 is usually enough for a good start
            # We use a callback to stop if time is running out, but DE doesn't support 
            # direct time-based stopping easily in older scipy. We rely on maxiter.
            # 11 points * 2 coords = 22 dimensions.
            de_res = differential_evolution(
                _objective_de_cart, 
                bounds, 
                maxiter=30, 
                popsize=20, 
                tol=1e-5, 
                seed=42, 
                polish=False, # We do our own polish
                init='sobol'
            )
            
            if de_res.x is not None:
                cart = de_res.x.reshape(N, 2)
                bary = _cart_to_bary(cart)
                bary = _normalize_bary(bary)
                score = _min_area_bary(bary)
                de_results.append((bary, score))
                
                # Also keep the best intermediate results if possible? 
                # DE doesn't expose them easily without callback. 
                # We'll just take the final result.
                
        except Exception:
            pass
    
    # Phase 2: Structured Initialization
    # Always try the structured init as a baseline
    struct_bary = _init_structured_bary()
    struct_score = _min_area_bary(struct_bary)
    de_results.append((struct_bary, struct_score))
    
    # Phase 3: Random Initializations (fewer, since DE covers global space)
    for seed in [1, 2, 3]:
        if time_left() < 20.0:
            break
        try:
            # Random barycentric
            bary_pts = []
            for _ in range(N):
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = np.array([1 - r1 - r2, r1, r2])
                bary_pts.append(pt)
            rand_bary = np.array(bary_pts)
            rand_score = _min_area_bary(rand_bary)
            de_results.append((rand_bary, rand_score))
        except Exception:
            pass
            
    # Sort by score descending
    de_results.sort(key=lambda x: x[1], reverse=True)
    
    # Phase 4: Local Refinement on Top Candidates
    # We have ~40-60 seconds left.
    # Refine the top 3 candidates.
    candidates_to_refine = de_results[:3]
    
    for bary_init, score_init in candidates_to_refine:
        if time_left() < 15.0:
            break
            
        try:
            bary_opt, score_opt = _optimize_bary(bary_init, time_left, rng)
            if score_opt > best_score:
                best_score = score_opt
                best_bary = bary_opt.copy()
        except Exception:
            pass

    # Fallback
    if best_bary is None:
        best_bary = _init_structured_bary()
        best_score = _min_area_bary(best_bary)
    
    # Final Polish: High precision Nelder-Mead on the best solution found
    if time_left() > 5.0:
        try:
            res = minimize(
                _objective_hard_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            final_bary = _normalize_bary(res.x.reshape(-1, 3))
            final_score = _min_area_bary(final_bary)
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary.copy()
        except Exception:
            pass

    # Convert to cartesian for output
    best_cart = _bary_to_cart(_normalize_bary(best_bary))
    
    return best_cart

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_cart(pts)}")
