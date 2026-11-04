# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _objective_smooth(x, beta, penalty_lambda=50.0):
    """
    Smooth objective for L-BFGS-B using log-sum-exp soft-min.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    # = m - (1/beta) * log(sum(exp(-beta * shifted)))
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _ga_fitness(bary, beta=200.0):
    """
    Fitness for genetic algorithm: soft-min of areas.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    m = np.min(areas)
    shifted = areas - m
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    return m - (1.0 / beta) * log_sum_exp

def _ga_evolve(pop, pop_size=30, n_gen=50, elite_frac=0.2, crossover_rate=0.8, mutation_rate=0.3, sigma=0.05, time_budget=30.0):
    """
    Simple genetic algorithm to explore the space.
    """
    start = time.time()
    pop = np.array([_project_to_bary_simplex(p.reshape(N, 2)) for p in pop])
    pop = pop.reshape(-1, 2*N)
    
    best_bary = None
    best_fit = -float('inf')
    
    for gen in range(n_gen):
        if time.time() - start > time_budget:
            break
        
        # Evaluate fitness
        fits = np.array([_ga_fitness(p.reshape(N, 2)) for p in pop])
        
        # Track best
        best_idx = np.argmax(fits)
        if fits[best_idx] > best_fit:
            best_fit = fits[best_idx]
            best_bary = pop[best_idx].reshape(N, 2)
        
        # Selection: keep elites
        n_elite = max(2, int(pop_size * elite_frac))
        elite_idx = np.argsort(fits)[::-1][:n_elite]
        new_pop = [pop[i].copy() for i in elite_idx]
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            parent1 = pop[idx1] if fits[idx1] > fits[idx2] else pop[idx2]
            
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            parent2 = pop[idx1] if fits[idx1] > fits[idx2] else pop[idx2]
            
            # Crossover
            if np.random.random() < crossover_rate:
                mask = np.random.random(2*N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                child += np.random.normal(0, sigma, size=2*N)
            
            # Project to simplex
            child_bary = child.reshape(N, 2)
            child_bary = _project_to_bary_simplex(child_bary)
            new_pop.append(child_bary.flatten())
        
        pop = np.array(new_pop)
    
    if best_bary is not None:
        return best_bary, best_fit
    return None, -float('inf')

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _polish_high_precision(x0, bounds, time_budget=15.0):
    """
    High-precision polish: multiple passes with tightening tolerances.
    """
    start = time.time()
    current = x0.copy()
    
    # Pass 1: SLSQP with tight tolerance
    if time.time() - start < time_budget:
        try:
            res = minimize(
                _objective_slsqp, current, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            current = _project_to_bary_simplex(res.x.reshape(N, 2)).flatten()
        except Exception:
            pass
    
    # Pass 2: L-BFGS-B with high beta
    if time.time() - start < time_budget:
        try:
            res = minimize(
                _objective_smooth, current, args=(1000.0,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current = _project_to_bary_simplex(res.x.reshape(N, 2)).flatten()
        except Exception:
            pass
    
    # Pass 3: Nelder-Mead with very tight tolerance
    if time.time() - start < time_budget:
        try:
            res = minimize(
                _objective_slsqp, current, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            current = _project_to_bary_simplex(res.x.reshape(N, 2)).flatten()
        except Exception:
            pass
            
    return current

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - selected, axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 3. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    bary5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.0, 0.3],
        [0.0, 0.7],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary5).flatten())
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 1: Local Optimization from Heuristic Inits
    for init in inits:
        if time_left() < 10:
            break
            
        try:
            init_bary = init.reshape(N, 2)
            init_bary = _project_to_bary_simplex(init_bary)
            
            best_local_x = init_bary
            best_local_score = _min_area_ratio_from_bary(init_bary)
            
            # Strategy: Multi-method refinement
            # 1. SLSQP
            if time_left() > 10:
                res = minimize(
                    _objective_slsqp, init_bary.flatten(), method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                bary_slsqp = _project_to_bary_simplex(res.x.reshape(N, 2))
                score_slsqp = _min_area_ratio_from_bary(bary_slsqp)
                if score_slsqp > best_local_score:
                    best_local_score = score_slsqp
                    best_local_x = bary_slsqp
                    init_bary = bary_slsqp

            # 2. Nelder-Mead
            if time_left() > 10:
                res = minimize(
                    _objective_slsqp, init_bary.flatten(), method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                bary_nm = _project_to_bary_simplex(res.x.reshape(N, 2))
                score_nm = _min_area_ratio_from_bary(bary_nm)
                if score_nm > best_local_score:
                    best_local_score = score_nm
                    best_local_x = bary_nm
                    init_bary = bary_nm

            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
                
        except Exception:
            pass

    # Phase 2: Genetic Algorithm for Global Exploration
    if time_left() > 30:
        # Create a diverse population
        pop = []
        # Add best found so far
        if best_x is not None:
            pop.append(best_x.flatten())
            # Add neighbors of best
            rng_ga = np.random.RandomState(123)
            for _ in range(10):
                perturbed = best_x.flatten() + rng_ga.normal(0, 0.05, size=2*N)
                pop.append(_project_to_bary_simplex(perturbed.reshape(N, 2)).flatten())
        
        # Add random points
        for _ in range(20):
            pts = np.zeros((N, 2))
            for i in range(N):
                u = rng_ga.random()
                v = rng_ga.random()
                while u + v > 1.0:
                    u = rng_ga.random()
                    v = rng_ga.random()
                pts[i, 0] = u
                pts[i, 1] = v
            pop.append(pts.flatten())
            
        # Add heuristic inits
        for init in inits:
            pop.append(init)
            
        # Run GA
        ga_best_bary, ga_best_fit = _ga_evolve(pop, pop_size=30, n_gen=50, time_budget=min(30, time_left() - 10))
        
        if ga_best_bary is not None:
            # Polish the GA result
            polished = _polish_high_precision(ga_best_bary.flatten(), bounds, time_budget=min(15, time_left() - 5))
            score = _min_area_ratio_from_bary(polished.reshape(N, 2))
            
            if score > best_score:
                best_score = score
                best_x = polished.reshape(N, 2)
                
    # Phase 3: Final Polish on the Best Candidate
    if best_x is not None and time_left() > 5:
        polished = _polish_high_precision(best_x.flatten(), bounds, time_budget=time_left() - 2)
        score = _min_area_ratio_from_bary(polished.reshape(N, 2))
        
        if score > best_score:
            best_score = score
            best_x = polished.reshape(N, 2)

    if best_x is None:
        best_x = _structured_init_fps()
        
    cart_points = _bary_to_cart(best_x[:, 0], best_x[:, 1])
    
    return cart_points
# EVOLVE-BLOCK-END
