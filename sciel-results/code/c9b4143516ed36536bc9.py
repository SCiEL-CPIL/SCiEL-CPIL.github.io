import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triple indices
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _project_to_bary_simplex(bary):
    """Project onto the barycentric simplex: u>=0, v>=0, u+v<=1."""
    bary = np.array(bary, dtype=float)
    bary = np.maximum(bary, 0.0)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _all_dets_from_bary(bary):
    """Compute all triple determinants (2*area) from barycentric coords."""
    idx = _IDX_TRIPLES
    ui = bary[idx[:, 0], 0]
    vi = bary[idx[:, 0], 1]
    uj = bary[idx[:, 1], 0]
    vj = bary[idx[:, 1], 1]
    uk = bary[idx[:, 2], 0]
    vk = bary[idx[:, 2], 1]
    
    du_ji = uj - ui
    dv_ji = vj - vi
    du_ki = uk - ui
    dv_ki = vk - vi
    
    dets = np.sqrt(3) / 2.0 * (du_ji * dv_ki - dv_ji * du_ki)
    return dets

def _min_area_ratio_from_bary(bary):
    """Compute the minimum area (divided by triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    min_det = np.min(np.abs(dets))
    return 0.5 * min_det / TRI_AREA

def _objective_exact(x):
    """Objective for minimization: -min_area_ratio."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    score = _min_area_ratio_from_bary(bary)
    return -score

def _objective_smooth(x, beta=50.0):
    """Smooth objective: approximates -min area using log-sum-exp."""
    bary = _project_to_bary_simplex(x.reshape(-1, 2))
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    max_det = np.max(abs_dets)
    if max_det < 1e-15:
        return 0.0
    
    neg_beta_dets = -beta * abs_dets
    max_nb = np.max(neg_beta_dets)
    lse = max_nb + np.log(np.sum(np.exp(neg_beta_dets - max_nb)))
    return lse / beta

def _structured_init_fps():
    """Farthest-point-sampling structured initialization in barycentric coords."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    points.append(np.array([1.0/3, 1.0/3]))
    
    rng = np.random.RandomState(789)
    while len(points) < N:
        candidates = []
        for _ in range(2000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v <= 1.0:
                candidates.append(np.array([u, v]))
        candidates = np.array(candidates)
        
        best_candidate = None
        best_min_dist = -1
        for cand in candidates:
            dists = [np.linalg.norm(cand - p) for p in points]
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        
        if best_candidate is not None:
            points.append(best_candidate)
    
    return np.array(points)

def _local_optimize_slsqp(x0, maxiter=200):
    """SLSQP with simplex constraints."""
    n = len(x0) // 2
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(
            _objective_exact, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_nm(x0, maxiter=500, xatol=1e-9, fatol=1e-9):
    """Nelder-Mead local optimization on smooth objective."""
    try:
        res = minimize(
            _objective_smooth, x0, method='Nelder-Mead',
            args=(50.0,),
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _ga_fitness(bary, rng):
    """Calculate fitness for a GA individual.
    Returns the score. We also store the barycentric coords."""
    bary_clamped = _project_to_bary_simplex(bary)
    return _min_area_ratio_from_bary(bary_clamped)

def _ga_evolve(population, scores, rng, max_iter=50, pop_size=30):
    """
    Evolve the population.
    population: list of barycentric arrays (N, 2)
    scores: list of float scores
    """
    best_idx = np.argmax(scores)
    best_bary = population[best_idx].copy()
    best_score = scores[best_idx]
    
    for it in range(max_iter):
        # Elitism: keep the best
        new_population = [best_bary.copy()]
        new_scores = [best_score]
        
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = rng.randint(0, len(population))
            idx2 = rng.randint(0, len(population))
            parent1 = population[idx1] if scores[idx1] > scores[idx2] else population[idx2]
            
            idx3 = rng.randint(0, len(population))
            idx4 = rng.randint(0, len(population))
            parent2 = population[idx3] if scores[idx3] > scores[idx4] else population[idx4]
            
            # Crossover: blend
            alpha = rng.uniform(0.3, 0.7)
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # Mutation: Gaussian jitter
            mutation_rate = 0.05
            if rng.random() < 0.8:
                child = child + rng.normal(0, mutation_rate, size=child.shape)
            
            # Swap mutation: swap two points
            if rng.random() < 0.3:
                i, j = rng.choice(N, 2, replace=False)
                child[i], child[j] = child[j].copy(), child[i].copy()
            
            child = _project_to_bary_simplex(child)
            
            # Local refinement for the child (optional, expensive)
            # We do a quick NM step to help the GA
            x_flat = child.flatten()
            x_opt = _local_optimize_nm(x_flat, maxiter=50, xatol=1e-7, fatol=1e-7)
            child_opt = _project_to_bary_simplex(x_opt.reshape(-1, 2))
            
            new_population.append(child_opt)
            new_scores.append(_min_area_ratio_from_bary(child_opt))
            
            # Update best if child is better
            if new_scores[-1] > best_score:
                best_score = new_scores[-1]
                best_bary = child_opt.copy()
        
        population = new_population
        scores = new_scores
        
        # Early stop if no improvement for a while? 
        # Keep it simple: run all iterations if time permits.
        
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # 1. Generate Initial Population
    # Mix of structured and random
    init_barys = []
    
    # FPS
    init_barys.append(_structured_init_fps())
    
    # Hexagonal-ish
    bary_hex = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3], [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75],
    ])
    init_barys.append(_project_to_bary_simplex(bary_hex.copy()))
    
    # Random fills
    for _ in range(20):
        bary_rand = []
        for _ in range(N):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                # Project to simplex
                s = u + v
                u /= s
                v /= s
            bary_rand.append([u, v])
        init_barys.append(_project_to_bary_simplex(np.array(bary_rand)))
    
    # Evaluate initial population
    population = []
    scores = []
    for b in init_barys:
        b_clamped = _project_to_bary_simplex(b)
        score = _min_area_ratio_from_bary(b_clamped)
        population.append(b_clamped)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_bary = b_clamped.copy()
            best_points = _bary_to_cart(b_clamped).copy()
    
    # 2. Genetic Algorithm Phase
    # We have ~23 individuals. Let's expand to 30 with more randoms if needed, or just run GA on this.
    # Let's run GA for a fixed number of iterations, checking time.
    max_ga_iters = 40
    
    for it in range(max_ga_iters):
        if time_left() < 20.0:
            break
            
        # Perform one GA iteration manually to control time
        # Select parents, create children, evaluate
        new_population = []
        new_scores = []
        
        # Elitism
        best_idx = np.argmax(scores)
        new_population.append(population[best_idx].copy())
        new_scores.append(scores[best_idx])
        
        target_pop_size = 30
        while len(new_population) < target_pop_size:
            # Tournament
            i1, i2 = rng.choice(len(population), 2, replace=False)
            p1 = population[i1] if scores[i1] > scores[i2] else population[i2]
            i3, i4 = rng.choice(len(population), 2, replace=False)
            p2 = population[i3] if scores[i3] > scores[i4] else population[i4]
            
            # Crossover
            alpha = rng.uniform(0.3, 0.7)
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation
            if rng.random() < 0.9:
                # Jitter
                child = child + rng.normal(0, 0.04, size=child.shape)
            if rng.random() < 0.2:
                # Swap
                i, j = rng.choice(N, 2, replace=False)
                child[i], child[j] = child[j].copy(), child[i].copy()
                
            child = _project_to_bary_simplex(child)
            
            # Quick local opt (NM) to improve child
            x_flat = child.flatten()
            x_opt = _local_optimize_nm(x_flat, maxiter=50, xatol=1e-6, fatol=1e-6)
            child_opt = _project_to_bary_simplex(x_opt.reshape(-1, 2))
            
            new_population.append(child_opt)
            new_scores.append(_min_area_ratio_from_bary(child_opt))
            
            if new_scores[-1] > best_score:
                best_score = new_scores[-1]
                best_bary = child_opt.copy()
                best_points = _bary_to_cart(child_opt).copy()
                
        population = new_population
        scores = new_scores
        
    # 3. Final Polish Phase
    # Take the top 5 individuals from the final GA population and polish them heavily
    top_indices = np.argsort(scores)[-5:]
    for idx in top_indices:
        if time_left() < 10.0:
            break
            
        bary_start = population[idx].copy()
        
        # SLSQP on exact objective
        x_sl = _local_optimize_slsqp(bary_start.flatten(), maxiter=200)
        bary_sl = _project_to_bary_simplex(x_sl.reshape(-1, 2))
        score_sl = _min_area_ratio_from_bary(bary_sl)
        
        if score_sl > best_score:
            best_score = score_sl
            best_bary = bary_sl.copy()
            best_points = _bary_to_cart(bary_sl).copy()
            
        # NM on smooth objective (high precision)
        x_nm = _local_optimize_nm(bary_sl.flatten(), maxiter=1000, xatol=1e-10, fatol=1e-10)
        bary_nm = _project_to_bary_simplex(x_nm.reshape(-1, 2))
        score_nm = _min_area_ratio_from_bary(bary_nm)
        
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
            best_points = _bary_to_cart(bary_nm).copy()
            
    # Fallback
    if best_points is None:
        bary_fb = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75],
        ])
        best_points = _bary_to_cart(_project_to_bary_simplex(bary_fb))
        
    return best_points
