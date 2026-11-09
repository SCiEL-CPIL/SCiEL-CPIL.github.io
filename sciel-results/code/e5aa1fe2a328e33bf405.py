# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def smooth_min_max_ratio(points, beta=50.0):
    """
    Smooth surrogate for (d_min/d_max)^2 using log-sum-exp.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Soft min via log-sum-exp
    neg_dists = -dists / beta
    max_neg = np.max(neg_dists)
    soft_min = -beta * (max_neg + np.log(np.sum(np.exp(neg_dists - max_neg))))
    
    # Soft max via log-sum-exp  
    pos_dists = dists / beta
    max_pos = np.max(pos_dists)
    soft_max = beta * (max_pos + np.log(np.sum(np.exp(pos_dists - max_pos))))
    
    if soft_max <= 0:
        return 0.0
    
    ratio = soft_min / soft_max
    return ratio * ratio

def nelder_mead_polish(points, time_limit, beta=50.0, seed=0):
    """
    Nelder-Mead optimization on the smooth surrogate, then verify with exact score.
    """
    n = points.shape[0]
    flat = points.ravel().copy()
    dim = flat.shape[0]
    
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initial simplex
    simplex = [flat.copy()]
    for i in range(dim):
        pt = flat.copy()
        pt[i] += 0.1 * (1.0 + abs(pt[i]))
        simplex.append(pt)
    
    # Evaluate initial simplex
    scores = [smooth_min_max_ratio(np.array(s).reshape(n, 2), beta) for s in simplex]
    
    alpha, gamma, rho = 0.5, 2.0, -0.5
    
    best_flat = flat.copy()
    best_score = max(scores)
    
    while time.time() - start_time < time_limit:
        # Sort by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Track best
        if scores[-1] > best_score:
            best_score = scores[-1]
            best_flat = simplex[-1].copy()
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = smooth_min_max_ratio(np.array(x_ref).reshape(n, 2), beta)
        
        if f_ref >= scores[0] and f_ref < scores[-2]:
            simplex[-1] = x_ref
            scores[-1] = f_ref
        elif f_ref >= scores[-2]:
            # Expansion
            x_exp = centroid + gamma * (centroid - simplex[-1])
            f_exp = smooth_min_max_ratio(np.array(x_exp).reshape(n, 2), beta)
            if f_exp > f_ref:
                simplex[-1] = x_exp
                scores[-1] = f_exp
            else:
                simplex[-1] = x_ref
                scores[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + 0.5 * (simplex[-1] - centroid)
            f_con = smooth_min_max_ratio(np.array(x_con).reshape(n, 2), beta)
            if f_con > scores[-1]:
                simplex[-1] = x_con
                scores[-1] = f_con
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    scores[i] = smooth_min_max_ratio(np.array(simplex[i]).reshape(n, 2), beta)
    
    # Verify with exact score and return
    result_points = best_flat.reshape(n, 2)
    exact_score = calculate_score(result_points)
    
    return result_points, exact_score

def local_search_smooth(points, time_limit, beta=50.0, seed=0):
    """
    Local search using smooth surrogate for faster evaluation.
    """
    n = points.shape[0]
    current = points.copy()
    current_score = smooth_min_max_ratio(current, beta)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    temp = 0.5
    cooling = 0.99995
    min_temp = 1e-6
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(n)
        step_size = 0.05 * temp + 0.005
        perturbation = rng.randn(2) * step_size
        
        new_point = current[idx].copy() + perturbation
        trial = current.copy()
        trial[idx] = new_point
        
        trial_score = smooth_min_max_ratio(trial, beta)
        
        if trial_score > current_score or rng.rand() < np.exp((trial_score - current_score) / max(temp, 1e-10)):
            current = trial
            current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
    
    return best, best_score

def optimize_points(points, time_limit, seed=0):
    """
    Optimize points using smooth surrogate + local search + Nelder-Mead polish.
    """
    n = points.shape[0]
    start_time = time.time()
    
    # Phase 1: Local search with smooth surrogate (70% of time)
    phase1_time = time_limit * 0.7
    ls_points, ls_score = local_search_smooth(points, phase1_time, beta=50.0, seed=seed)
    
    # Phase 2: Nelder-Mead polish on smooth surrogate (30% of time)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        nm_points, nm_score = nelder_mead_polish(ls_points, remaining, beta=50.0, seed=seed + 1)
        
        # Compare with exact scores
        ls_exact = calculate_score(ls_points)
        nm_exact = calculate_score(nm_points)
        
        if nm_exact > ls_exact:
            return nm_points, nm_exact
        else:
            return ls_points, ls_exact
    else:
        ls_exact = calculate_score(ls_points)
        return ls_points, ls_exact

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 620  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Hexagonal lattice (best known packing structure)
    hex_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3) / 2 * r
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    # Select 16 points closest to center
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    candidates.append(hex_points[idx])
    
    # 2. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    candidates.append(np.column_stack((xx.ravel(), yy.ravel())))
    
    # 3. Circle with two rings
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi / 8, 2 * np.pi, 8, endpoint=False)
    r1, r2 = 1.0, 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    candidates.append(np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])]))
    
    # 4. Random points in a disk
    rng_init = np.random.RandomState(42)
    angles = rng_init.rand(n) * 2 * np.pi
    radii = np.sqrt(rng_init.rand(n)) * 1.5
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    candidates.append(np.column_stack((x, y)))
    
    # 5. Random points in a square
    rng_init = np.random.RandomState(123)
    candidates.append(rng_init.rand(n, 2) * 3 - 1.5)
    
    # 6. Perturbed circular distribution
    rng_init = np.random.RandomState(777)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng_init.randn(n) * 0.1
    radii = np.sqrt(rng_init.rand(n)) * 1.2
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    candidates.append(np.column_stack((x, y)))
    
    # Optimize each candidate
    best_points = None
    best_score = -1
    
    # Allocate time: ~80s per candidate, leaving time for final polish
    time_per_candidate = 80
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 30:
            break
        
        cand = cand.astype(float)
        remaining_for_this = min(time_per_candidate, global_deadline - 30 - time.time())
        if remaining_for_this < 10:
            break
            
        res_points, res_score = optimize_points(cand, remaining_for_this, seed=1000 + i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # If time remains, do a final high-precision polish on the best candidate
    if best_points is not None and time.time() < global_deadline - 10:
        remaining = global_deadline - 10 - time.time()
        if remaining > 15:
            # Nelder-Mead with higher beta for smoother landscape
            nm_points, nm_score = nelder_mead_polish(best_points, remaining, beta=100.0, seed=9999)
            nm_exact = calculate_score(nm_points)
            
            if nm_exact > best_score:
                best_points = nm_points
                best_score = nm_exact
    
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points

# EVOLVE-BLOCK-END
