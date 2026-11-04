"""
Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Strategy:
1. Genetic algorithm (GA) to explore the solution space broadly and escape local minima.
2. Soft-min (log-sum-exp) surrogate for smooth gradients within the GA's evaluation.
3. Final polishing with L-BFGS-B and Nelder-Mead on the raw min-area objective.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# Number of triples is C(11, 3) = 165
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]


def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _all_triangle_areas(points):
    """
    Compute all triangle areas, vectorized.
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    Point = lambda_1 * V1 + lambda_2 * V2 (since V0 is origin)
    """
    s = params[0::2]
    t = params[1::2]
    
    # Clip to avoid numerical issues at boundaries
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])


def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Inverse barycentric transformation
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params


def _soft_min_objective_and_grad(params, beta):
    """
    Objective: negative soft-min of triangle areas.
    Gradient: computed via chain rule.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Numerical stability for log-sum-exp
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Objective value
    obj = -(min_a - np.log(sum_exp) / beta)
    
    # Gradient
    weights = exp_shifted / sum_exp
    
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    d_cross_pi = np.column_stack([
        -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1]),
        -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    ])
    d_cross_pj = np.column_stack([
        pk[:, 1] - pi[:, 1],
        -(pk[:, 0] - pi[:, 0])
    ])
    d_cross_pk = np.column_stack([
        -(pj[:, 1] - pi[:, 1]),
        pj[:, 0] - pi[:, 0]
    ])
    
    grad_points = np.zeros((N, 2))
    w = weights * sign_cross * 0.5
    
    np.add.at(grad_points, _IDX_I, w[:, np.newaxis] * d_cross_pi)
    np.add.at(grad_points, _IDX_J, w[:, np.newaxis] * d_cross_pj)
    np.add.at(grad_points, _IDX_K, w[:, np.newaxis] * d_cross_pk)
    
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return obj, -grad_params

def _raw_objective(params):
    """Objective for final polishing: negative min area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _structured_init():
    """Generate a good structured initialization."""
    pts = np.zeros((N, 2))
    pts[0] = V0 + 0.05 * (V1 + V2 - 2*V0)
    pts[1] = V1 + 0.05 * (V0 + V2 - 2*V1)
    pts[2] = V2 + 0.05 * (V0 + V1 - 2*V2)
    pts[3] = (V0 + V1) / 2.0
    pts[4] = (V1 + V2) / 2.0
    pts[5] = (V2 + V0) / 2.0
    centroid = (V0 + V1 + V2) / 3.0
    pts[6] = centroid
    pts[7] = centroid + 0.15 * (V0 - centroid)
    pts[8] = centroid + 0.15 * (V1 - centroid)
    pts[9] = centroid + 0.15 * (V2 - centroid)
    pts[10] = centroid + 0.08 * (V0 - centroid)
    return _points_to_param(pts)

def _polish(params, deadline):
    """
    Final polishing with L-BFGS-B and Nelder-Mead.
    """
    if time.time() > deadline - 5.0:
        return params
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # L-BFGS-B polish
    try:
        res = minimize(
            _raw_objective, params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None:
            params = res.x
    except Exception:
        pass
    
    # Nelder-Mead polish
    if time.time() > deadline - 2.0:
        return params
    
    try:
        res = minimize(
            _raw_objective, params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.x is not None:
            params = np.clip(res.x, 0.0, 1.0)
    except Exception:
        pass
    
    return params

def _ga_step(population, rng, elite_size=10, mutation_rate=0.15, crossover_rate=0.8):
    """
    One step of genetic algorithm.
    population: list of params arrays (2N each)
    """
    n = len(population)
    
    # Evaluate fitness (min area)
    fitness = np.array([_min_triangle_area_vectorized(_param_to_points(p)) for p in population])
    
    # Sort by fitness (descending - we want max min area)
    sorted_idx = np.argsort(fitness)[::-1]
    sorted_pop = [population[i] for i in sorted_idx]
    sorted_fit = fitness[sorted_idx]
    
    # Elitism: keep top elite_size
    new_pop = [sorted_pop[i].copy() for i in range(min(elite_size, n))]
    
    # Tournament selection + crossover + mutation
    while len(new_pop) < n:
        # Select two parents via tournament
        idx1 = rng.choice(n, size=elite_size)
        idx2 = rng.choice(n, size=elite_size)
        parent1 = sorted_pop[idx1[np.argmax(fitness[idx1])]]
        parent2 = sorted_pop[idx2[np.argmax(fitness[idx2])]]
        
        # Crossover
        if rng.random() < crossover_rate:
            # Uniform crossover
            mask = rng.random(2 * N) > 0.5
            child = np.where(mask, parent1, parent2)
        else:
            child = parent1.copy()
        
        # Mutation
        if rng.random() < mutation_rate:
            # Gaussian mutation
            sigma = 0.05
            child = child + sigma * rng.randn(2 * N)
            child = np.clip(child, 0.0, 1.0)
        
        new_pop.append(child)
    
    return new_pop

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # ===== Phase 1: Genetic Algorithm =====
    # Generate diverse initial population
    population = []
    
    # Structured init
    population.append(_structured_init())
    
    # FPS inits
    for i in range(5):
        pts = _generate_fps_initialization(rng)
        population.append(_points_to_param(pts))
    
    # Random inits
    for i in range(5):
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        population.append(params)
    
    # Run GA for several generations
    n_generations = 30
    for gen in range(n_generations):
        if time_left() < 10.0:
            break
        
        population = _ga_step(population, rng, elite_size=10, mutation_rate=0.15, crossover_rate=0.8)
        
        # Track best
        for p in population:
            score = _min_triangle_area_vectorized(_param_to_points(p))
            if score > best_score:
                best_score = score
                best_params = p.copy()
        
        # Periodic polish of the best to help guide the GA
        if gen % 5 == 4 and best_params is not None:
            if time_left() > 5.0:
                polished = _polish(best_params, deadline)
                score = _min_triangle_area_vectorized(_param_to_points(polished))
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
                    # Inject polished solution into population
                    population[0] = best_params.copy()
    
    # ===== Phase 2: Local optimization from GA best =====
    if time_left() > 15.0 and best_params is not None:
        # L-BFGS-B with soft-min annealing
        bounds = [(0.0, 1.0)] * (2 * N)
        betas = [10.0, 30.0, 100.0]
        
        current = best_params.copy()
        for beta in betas:
            if time_left() < 5.0:
                break
            
            def obj_grad(params):
                return _soft_min_objective_and_grad(params, beta)
            
            try:
                res = minimize(
                    obj_grad, current,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    current = res.x
            except Exception:
                pass
        
        # Final polish
        if time_left() > 3.0:
            current = _polish(current, deadline)
        
        score = _min_triangle_area_vectorized(_param_to_points(current))
        if score > best_score:
            best_score = score
            best_params = current.copy()
    
    # ===== Phase 3: Additional local starts from GA population =====
    if time_left() > 20.0:
        # Take top 5 from GA population and polish each
        fitness = np.array([_min_triangle_area_vectorized(_param_to_points(p)) for p in population])
        top_idx = np.argsort(fitness)[::-1][:5]
        
        for idx in top_idx:
            if time_left() < 5.0:
                break
            
            current = population[idx].copy()
            
            # Quick L-BFGS-B
            bounds = [(0.0, 1.0)] * (2 * N)
            try:
                res = minimize(
                    _raw_objective, current,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'ftol': 1e-12}
                )
                if res.x is not None:
                    current = res.x
            except Exception:
                pass
            
            # Polish
            if time_left() > 3.0:
                current = _polish(current, deadline)
            
            score = _min_triangle_area_vectorized(_param_to_points(current))
            if score > best_score:
                best_score = score
                best_params = current.copy()
    
    if best_params is None:
        # Fallback
        return np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/8], [0.25, np.sqrt(3)/8], [0.75, np.sqrt(3)/8],
            [0.5, 3*np.sqrt(3)/8], [0.5, np.sqrt(3)/4]
        ])
        
    best_points = _param_to_points(best_params)
    
    return best_points
