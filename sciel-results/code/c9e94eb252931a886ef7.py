"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRIPLES = _IDX_TRIPLES.shape[0]

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian.
    Point = u*V1 + v*V2 + (1-u-v)*V0 = u*V1 + v*V2
    """
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _compute_areas(points):
    """Compute areas of all 165 triangles formed by the points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    
    d1 = pj - pi
    d2 = pk - pi
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Exact minimum triangle area."""
    return np.min(_compute_areas(points))

def _soft_min_objective(bary_flat, beta):
    """
    Objective for optimization: negative soft-min of triangle areas.
    Soft min approximates the min function smoothly.
    S = -1/beta * log(sum(exp(-beta * areas)))
    As beta -> inf, S -> -min(areas).
    We minimize -S, so we maximize S.
    """
    bary = bary_flat.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _compute_areas(points)
    
    # Shift areas to prevent underflow in exp
    # max area is ~ 0.4, min is 0.
    # To be safe, shift by max area? No, standard soft min is fine if beta isn't huge.
    # However, for numerical stability:
    # log(sum(exp(-beta * a))) = -beta * max(a) + log(sum(exp(-beta * (a - max(a)))))
    # This is more stable.
    max_a = np.max(areas)
    # If all areas are 0, return 0
    if max_a == 0:
        return 0.0
        
    # We want to maximize min(areas).
    # Soft min M = -1/beta * log(sum(exp(-beta * a)))
    # To maximize M, we minimize -M.
    # -M = 1/beta * log(sum(exp(-beta * a)))
    
    # Stable computation of log(sum(exp(x))) where x = -beta * a
    x = -beta * areas
    x_max = np.max(x)
    if x_max == -np.inf:
         return 0.0
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    val = (1.0 / beta) * log_sum_exp
    return val

def _simulated_annealing(bary_init, time_limit, rng):
    """
    Run simulated annealing on barycentric coordinates.
    Returns best bary found.
    """
    bary = bary_init.copy()
    bary = _clamp_bary(bary)
    
    # Parameters
    T_start = 0.05
    T_end = 0.001
    beta_start = 10.0
    beta_end = 1000.0
    
    # We will run for a fixed number of iterations or until time limit
    # Estimate iterations: 
    # Each iteration involves computing 165 areas. Very fast.
    # Let's say 5000 iterations per SA run.
    n_iter = 5000
    
    best_bary = bary.copy()
    best_score = _min_triangle_area(_bary_to_cart(bary))
    current_score = best_score
    
    for i in range(n_iter):
        # Check time
        if time.time() > time_limit:
            break
            
        # Annealing schedule
        alpha = i / n_iter
        T = T_start * (T_end / T_start) ** alpha
        beta = beta_start * (beta_end / beta_start) ** alpha
        
        # Perturbation
        # Step size should be related to T or fixed small value
        step_size = 0.05 * (1.0 - 0.5 * alpha) 
        perturbation = rng.normal(0, step_size, size=bary.shape)
        new_bary = bary + perturbation
        new_bary = _clamp_bary(new_bary)
        
        # Evaluate
        # For SA, we can use the soft min or hard min.
        # Using soft min with current beta makes the landscape smoother.
        # However, for acceptance criterion, it's often better to use the actual objective (hard min)
        # or a consistent soft min.
        # Let's use the hard min for the score comparison to ensure we are tracking the true objective.
        # But the "energy" difference for acceptance can be smoothed.
        
        new_points = _bary_to_cart(new_bary)
        new_areas = _compute_areas(new_points)
        new_min = np.min(new_areas)
        
        # Acceptance
        if new_min > current_score:
            bary = new_bary
            current_score = new_min
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
        else:
            # Delta E = new_score - old_score (negative here)
            delta = new_min - current_score
            # Accept with probability exp(delta / T)
            # Since delta is negative, delta/T is negative.
            if rng.uniform() < np.exp(delta / T):
                bary = new_bary
                current_score = new_min
                
    return best_bary, best_score

def _polish(bary_init, time_limit):
    """
    Polish the solution using SLSQP on the exact objective.
    """
    def objective(x):
        bary = x.reshape(-1, 2)
        bary = _clamp_bary(bary)
        points = _bary_to_cart(bary)
        return -_min_triangle_area(points)
    
    # Constraints for SLSQP:
    # u >= 0, v >= 0, u+v <= 1
    constraints = []
    bounds = []
    for i in range(N):
        u_idx = 2 * i
        v_idx = 2 * i + 1
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        
        # u_i >= 0
        A_u = np.zeros(2 * N)
        A_u[u_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_u: A @ x})
        
        # v_i >= 0
        A_v = np.zeros(2 * N)
        A_v[v_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_v: A @ x})
        
        # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
        A_sum = np.zeros(2 * N)
        A_sum[u_idx] = -1.0
        A_sum[v_idx] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_sum: 1.0 + A @ x})
        
    x0 = bary_init.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        bary_opt = _clamp_bary(result.x.reshape(-1, 2))
        score_opt = -result.fun
        
        # Check time
        if time.time() > time_limit:
            return bary_init, _min_triangle_area(_bary_to_cart(bary_init))
            
        return bary_opt, score_opt
        
    except Exception:
        return bary_init, _min_triangle_area(_bary_to_cart(bary_init))

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Structured: Vertices, Midpoints, Centroid
    # (0,0), (1,0), (0,1) in barycentric
    # (0.5, 0), (0, 0.5), (0.5, 0.5)
    # (1/3, 1/3)
    # (0.25, 0), (0.75, 0), (0.25, 0.25), (0.5, 0.25)
    bary_struct = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1.0/3, 1.0/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
    ])
    inits.append(bary_struct)
    
    # 2. Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary_r = np.zeros((N, 2))
        for _ in range(N):
            while True:
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                if u + v <= 1:
                    bary_r[_] = [u, v]
                    break
        inits.append(bary_r)
        
    # 3. Perturbed Structured
    for scale in [0.05, 0.1, 0.2]:
        bary_pert = bary_struct + rng.normal(0, scale, size=bary_struct.shape)
        bary_pert = _clamp_bary(bary_pert)
        inits.append(bary_pert)
        
    return inits

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 140s
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    inits = _generate_initializations()
    
    # Phase 1: Simulated Annealing from each initialization
    rng_sa = np.random.RandomState(0)
    
    for i, bary0 in enumerate(inits):
        if time_left() < 10.0:
            break
            
        # Run SA
        # We can run multiple SA runs from the same init with different seeds
        # But to save time, we'll run one SA per init first, then polish the best ones.
        
        bary_sa, score_sa = _simulated_annealing(bary0, deadline - 5.0, rng_sa)
        
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
            
    # Phase 2: Polish the best candidates
    # We will keep a list of top candidates to polish
    # For simplicity, we just polish the current best.
    # To be more robust, we could keep top 3.
    
    # Let's also try polishing the initial structured one directly, just in case SA messed up
    candidates = []
    if best_bary is not None:
        candidates.append(best_bary)
    candidates.append(inits[0]) # Structured
    
    for bary_c in candidates:
        if time_left() < 15.0:
            break
            
        bary_pol, score_pol = _polish(bary_c, deadline - 5.0)
        
        if score_pol > best_score:
            best_score = score_pol
            best_bary = bary_pol.copy()
            
    # Phase 3: If time remains, do more SA runs with different seeds from the best solution
    if time_left() > 20.0 and best_bary is not None:
        for seed in range(10):
            if time_left() < 10.0:
                break
            rng_run = np.random.RandomState(1000 + seed)
            # Perturb best solution
            bary_pert = best_bary + rng_run.normal(0, 0.05, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            
            bary_sa, score_sa = _simulated_annealing(bary_pert, deadline - 5.0, rng_run)
            
            if score_sa > best_score:
                best_score = score_sa
                best_bary = bary_sa.copy()
                
            # Polish this new candidate
            if time_left() > 10.0:
                bary_pol, score_pol = _polish(bary_sa, deadline - 5.0)
                if score_pol > best_score:
                    best_score = score_pol
                    best_bary = bary_pol.copy()
                    
    # Final check and return
    if best_bary is None:
        best_bary = inits[0]
        
    best_points = _bary_to_cart(best_bary)
    best_points = _clip_to_triangle(best_points)
    
    return best_points

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points
