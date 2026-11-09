# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _smooth_score(radii, angles):
    """
    Smoothed surrogate for the min-vertex-score.
    Uses soft-min to make the objective continuous for optimization.
    """
    n = len(radii)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    pn, _, _ = _normalize(pts)
    if pn is None:
        return -100.0
        
    if not _check_strictly_convex(pn):
        return -100.0
        
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(pn, i)
    
    # Soft min
    beta = 50.0
    soft_min = -np.log(np.sum(np.exp(-beta * scores)) / n) / beta
    return soft_min

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()
            return True
        return False

    # Initial seeds
    # 1. Regular
    r_current = np.ones(n)
    try_candidate(r_current)
    
    # 2. Alternating
    r_alt = 1.0 + 0.1 * np.array([1, -1] * 10)
    try_candidate(r_alt)
    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        r_rand = 1.0 + 0.2 * rng.normal(0, 1, n)
        r_rand = np.clip(r_rand, 0.5, 2.0)
        try_candidate(r_rand)

    # Strategy: Use Differential Evolution to find a good starting point
    # in the radius space, then polish with Nelder-Mead on the smooth score.
    
    # Bounds for radii
    bounds = [(0.5, 2.5) for _ in range(n)]
    
    # Objective function for DE (maximize smooth score)
    def objective_de(r):
        return -_smooth_score(r, angles)
    
    # Run DE for a limited number of iterations to get a good initial guess
    # DE is stochastic, so we set a seed for reproducibility
    # maxiter is kept low to ensure we finish in time, as DE is expensive
    # We'll run it in chunks if needed, but one run should be enough.
    
    try:
        # DE can be slow, so we limit it. 
        # If we have time, we can run it.
        de_start = time.time()
        # Use a small population and few iterations to get a decent start
        # Then refine.
        result_de = differential_evolution(
            objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=0.01,
            polish=False, # We will polish separately
            init='sobol'
        )
        
        if result_de.success or result_de.fun < 0:
            r_de = result_de.x
            try_candidate(r_de)
            
            # Now refine with Nelder-Mead on the smooth score
            # This is faster and more precise for local optimization
            def objective_nm(r):
                return -_smooth_score(r, angles)
            
            # Start from DE result or best so far
            r_start = r_de if result_de.fun < 0 else best_radii
            
            # Nelder-Mead is local, so we might need multiple starts
            # But DE gives us a good global start.
            
            nm_options = {
                'maxiter': 200,
                'xatol': 1e-4,
                'fatol': 1e-4
            }
            
            result_nm = minimize(
                objective_nm, 
                r_start, 
                method='Nelder-Mead', 
                options=nm_options
            )
            
            if result_nm.success:
                try_candidate(result_nm.x)
                
    except Exception:
        pass # Fallback to local search if DE fails or times out
    
    # Final polish with local search on the actual (non-smooth) score
    # This ensures we squeeze out the last bit of score
    if best_score > 0 and time.time() < deadline:
        current = best_radii.copy()
        step = 0.01
        # Reduced iterations to fit in time budget
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    new_r = current.copy()
                    new_r[i] += delta
                    if new_r[i] < 0.5 or new_r[i] > 2.5:
                        continue
                    s_new = _score_polygon(np.column_stack([new_r * np.cos(angles), new_r * np.sin(angles)]))
                    if s_new > best_score:
                        best_score = s_new
                        best_radii = new_r.copy()
                        current = new_r
                        improved = True
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break

    return np.column_stack([best_radii * np.cos(angles), best_radii * np.sin(angles)])

# EVOLVE-BLOCK-END
