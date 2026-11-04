"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Combines Genetic Algorithm global search (inspiration 1) with robust local refinement 
(L-BFGS soft-min, Beta Annealing, SLSQP) from the current program."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _bary_to_cart(bary2d):
    """Convert 2D barycentric (u, v) to Cartesian. w = 1-u-v. P = u*V0 + v*V1 + w*V2."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack([x, y])

def _all_dets_from_bary(bary2d):
    """Compute all determinants (twice the area in barycentric coords, scaled by TRI_AREA)."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    Pi = P[_IDX_I]
    Pj = P[_IDX_J]
    Pk = P[_IDX_K]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det

def _min_area_ratio(bary2d):
    """Returns the minimum |det| (proportional to area / TRI_AREA)."""
    dets = _all_dets_from_bary(bary2d)
    return np.min(np.abs(dets))

def _clamp_bary(bary2d):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary2d[:, 0])
    v = np.maximum(0.0, bary2d[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _score_bary(bary2d):
    """Score: minimum triangle area (in actual units)."""
    bary2d = _clamp_bary(bary2d)
    return _min_area_ratio(bary2d) * TRI_AREA

def _softmin_objective(x, beta=120.0):
    """Smooth surrogate: negative soft-min of |dets| using log-sum-exp, with floor penalty."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # Numerically stable log-sum-exp
    max_shift = np.max(shifted)
    log_sum = max_shift + np.log(np.sum(np.exp(-beta * (shifted - max_shift))))
    soft_min = min_a - log_sum / beta
    
    # Penalize any area that is extremely small (near-degenerate)
    floor_penalty = 1000.0 * np.sum(np.maximum(0.0, 1e-6 - areas) ** 2)
    
    return -soft_min + floor_penalty

def _hard_objective(x):
    """Hard objective: negative min |det|."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    return -np.min(np.abs(dets))

def _structured_init_bary(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Hexagonal-ish
    pB = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [0.33, 0.33],
        [0.15, 0.15], [0.65, 0.15], [0.15, 0.65], [0.5, 0.2]
    ])
    patterns.append(pB)
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Concentric hexagons
    pD = np.array([
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
    patterns.append(pD)
    
    # Pattern E: Magic init
    pE = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    patterns.append(pE)
    
    # Pattern F: Hexagonal lattice
    pF = np.array([
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
    patterns.append(pF)
    
    idx = seed % len(patterns)
    bary2d = patterns[idx].copy()
    jitter = rng.normal(0, 0.015, size=bary2d.shape)
    bary2d = bary2d + jitter
    bary2d = _clamp_bary(bary2d)
    return bary2d

def _random_init_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    bary2d = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.01, 0.99)
            v = rng.uniform(0.01, 0.99)
            if u + v < 0.98:
                bary2d[i] = [u, v]
                break
    return bary2d

def _l_bfgs_softmin(bary2d_0, maxiter=400, beta=120.0):
    """L-BFGS-B optimization using soft-min surrogate for smooth gradients."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        def obj(x):
            return _softmin_objective(x, beta=beta)
        
        res = minimize(
            obj, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _l_bfgs_hard(bary2d_0, maxiter=200):
    """L-BFGS-B optimization using hard min."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        res = minimize(
            _hard_objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _slsqp_refine(bary2d_0, maxiter=200):
    """SLSQP refinement with explicit constraint u+v<=1."""
    try:
        x0 = bary2d_0.flatten()
        bounds = [(0, 1)] * (2 * N)
        
        cons = []
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.concatenate([np.zeros(2*i), [-1.0, -1.0], np.zeros(2*(N-i-1))])
            })
        
        res = minimize(
            _hard_objective, x0, method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _nelder_mead_refine(bary2d_0, maxiter=500):
    """Nelder-Mead refinement."""
    try:
        x0 = bary2d_0.flatten()
        
        res = minimize(
            _hard_objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary2d_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary2d_opt
    except Exception:
        return bary2d_0.copy()

def _beta_anneal_refine(bary2d_0, betas=(40.0, 80.0, 160.0, 320.0, 640.0)):
    """Refine with increasing beta values."""
    current = bary2d_0.copy()
    for beta in betas:
        current = _l_bfgs_softmin(current, beta=beta, maxiter=200)
    current = _l_bfgs_hard(current, maxiter=300)
    return current

def _genetic_algorithm(rng, pop_size=40, n_gen=50, deadline=None):
    """
    Genetic Algorithm for global search.
    Operates on barycentric coordinates (N, 2).
    Uses soft-min objective for fitness.
    """
    # Initialize population
    # Mix of structured and random
    pop = []
    for i in range(pop_size // 2):
        pop.append(_structured_init_bary(seed=100 + i))
    for i in range(pop_size - pop_size // 2):
        pop.append(_random_init_bary(seed=200 + i))
    
    # Evaluate initial fitness
    # We use a moderate beta for GA to keep it smooth but responsive
    ga_beta = 50.0
    
    def get_fitness(bary2d):
        return _softmin_objective(bary2d.flatten(), beta=ga_beta)
    
    fitness = np.array([get_fitness(p) for p in pop])
    
    for gen in range(n_gen):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort by fitness (descending, since we are minimizing negative score? 
        # Wait, _softmin_objective returns -soft_min + penalty. So lower is better.
        # But usually GA maximizes. Let's maximize the "score" which is soft_min.
        # So fitness = - _softmin_objective ... 
        # Actually, let's just stick to minimizing the objective function value.
        # Lower value of _softmin_objective is better.
        
        order = np.argsort(fitness)
        sorted_pop = [pop[i] for i in order]
        sorted_fit = fitness[order]
        
        # Elitism
        elite_frac = 0.2
        elite_n = max(1, int(pop_size * elite_frac))
        new_pop = [sorted_pop[i].copy() for i in range(elite_n)]
        
        # Fill rest
        while len(new_pop) < pop_size:
            if deadline is not None and time.time() > deadline:
                break
                
            # Tournament selection
            t1 = rng.randint(0, pop_size)
            t2 = rng.randint(0, pop_size)
            p1 = pop[t1] if fitness[t1] < fitness[t2] else pop[t2]
            
            t1 = rng.randint(0, pop_size)
            t2 = rng.randint(0, pop_size)
            p2 = pop[t1] if fitness[t1] < fitness[t2] else pop[t2]
            
            # Crossover (Blend)
            if rng.random() < 0.8:
                alpha = rng.uniform(0.3, 0.7)
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            
            # Mutation
            if rng.random() < 0.3:
                # Gaussian mutation
                child += rng.normal(0, 0.05, size=child.shape)
            
            child = _clamp_bary(child)
            new_pop.append(child)
        
        pop = new_pop
        fitness = np.array([get_fitness(p) for p in pop])
        
        # If we found a very good solution, we might stop early? 
        # No, let it run to use the time budget.
        
    # Return best individual
    best_idx = np.argmin(fitness)
    return pop[best_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary2d = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Run Genetic Algorithm for global search
    # Budget: ~60-80 seconds for GA
    if time_left() > 60.0:
        try:
            ga_deadline = start_time + 70.0
            ga_result = _genetic_algorithm(rng, pop_size=30, n_gen=100, deadline=ga_deadline)
            if ga_result is not None:
                score = _score_bary(ga_result)
                if score > best_score:
                    best_score = score
                    best_bary2d = ga_result.copy()
        except Exception:
            pass
    
    # 2. If GA didn't find anything good or timed out, fall back to structured inits
    if best_bary2d is None or best_score < 0.01:
         # Quick check of a few structured inits
         for i in range(6):
             if time_left() < 10.0: break
             bary2d = _structured_init_bary(seed=42 + i)
             score = _score_bary(bary2d)
             if score > best_score:
                 best_score = score
                 best_bary2d = bary2d.copy()
    
    # 3. Refine the best candidate found
    if best_bary2d is not None and time_left() > 20.0:
        try:
            # Step 3a: Soft-min annealing
            refined = _beta_anneal_refine(best_bary2d)
            score = _score_bary(refined)
            if score > best_score:
                best_score = score
                best_bary2d = refined.copy()
        except Exception:
            pass
        
        # Step 3b: SLSQP refinement
        if time_left() > 15.0:
            try:
                refined = _slsqp_refine(best_bary2d, maxiter=300)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
            except Exception:
                pass
        
        # Step 3c: Nelder-Mead refinement
        if time_left() > 10.0:
            try:
                refined = _nelder_mead_refine(best_bary2d, maxiter=500)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
            except Exception:
                pass
        
        # Step 3d: Final L-BFGS hard polish
        if time_left() > 5.0:
            try:
                refined = _l_bfgs_hard(best_bary2d, maxiter=300)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
            except Exception:
                pass
    
    # 4. Local search from perturbed best solution (if time remains)
    if best_bary2d is not None and time_left() > 15.0:
        for i in range(5):
            if time_left() < 5.0:
                break
            try:
                noise = rng.normal(0, 0.005, size=2 * N)
                perturbed = best_bary2d + noise.reshape(N, 2)
                perturbed = _clamp_bary(perturbed)
                
                refined = _l_bfgs_softmin(perturbed, beta=120.0, maxiter=150)
                score = _score_bary(refined)
                if score > best_score:
                    best_score = score
                    best_bary2d = refined.copy()
                    
                    # Quick polish on new best
                    if time_left() > 5.0:
                        refined = _l_bfgs_hard(refined, maxiter=100)
                        score2 = _score_bary(refined)
                        if score2 > best_score:
                            best_score = score2
                            best_bary2d = refined.copy()
            except Exception:
                pass

    # Convert best barycentric to Cartesian
    if best_bary2d is None:
        # Fallback: use structured init
        bary2d = _structured_init_bary(seed=42)
    else:
        bary2d = _clamp_bary(best_bary2d)
    
    pts = _bary_to_cart(bary2d)
    return np.array(pts)
