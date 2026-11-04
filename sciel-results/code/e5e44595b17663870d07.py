"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0
TRI_AREA = 0.5 * H

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

# Barycentric coordinate helpers
def _bary_to_cart(bary):
    """Convert barycentric coords (l0, l1, l2) to Cartesian. bary shape (N, 3)."""
    l1 = bary[:, 1:2]
    l2 = bary[:, 2:3]
    x = l1 * V1[0] + l2 * V2[0]
    y = l1 * V1[1] + l2 * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords. Returns (N, 3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    l2 = 2.0 * y / np.sqrt(3)
    l1 = x - 0.5 * l2
    l0 = 1.0 - l1 - l2
    return np.column_stack([l0, l1, l2])

def _min_area_from_bary(bary):
    """Compute min triangle area directly from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _softmin_from_bary(bary, beta=80.0):
    """Soft-min of triangle areas from barycentric coords."""
    pts = _bary_to_cart(bary)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return min_area - log_sum

def _ga_fitness(bary, beta=60.0):
    """Fitness for GA: soft-min area with boundary penalty."""
    # Penalty for barycentric coords outside [0,1]
    penalty = np.sum(np.maximum(0, bary - 1.0)**2) + np.sum(np.maximum(0, -bary)**2)
    fitness = _softmin_from_bary(bary, beta) - 0.1 * penalty
    return fitness

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid range and renormalize."""
    b = np.clip(bary, 1e-12, None)
    s = b.sum(axis=1, keepdims=True)
    b = b / s
    return b

def _ga_evolve(pop, rng, n_gen=30, pop_size=40, elite_frac=0.2, crossover_rate=0.8,
               mutation_rate=0.1, mut_scale=0.05, deadline=None):
    """Genetic algorithm evolving barycentric coordinates."""
    pop_size = len(pop)
    elite_n = max(1, int(pop_size * elite_frac))
    
    for gen in range(n_gen):
        if deadline is not None and time.time() > deadline:
            break
        
        # Evaluate fitness
        fitness = np.array([_ga_fitness(pop[i]) for i in range(pop_size)])
        
        # Sort by fitness (descending)
        order = np.argsort(-fitness)
        sorted_pop = pop[order]
        sorted_fit = fitness[order]
        
        # Elitism: keep top individuals
        new_pop = [sorted_pop[i].copy() for i in range(elite_n)]
        
        # Fill rest via tournament selection + crossover + mutation
        while len(new_pop) < pop_size:
            if deadline is not None and time.time() > deadline:
                break
            
            # Tournament selection
            t1 = rng.randint(0, pop_size)
            t2 = rng.randint(0, pop_size)
            p1 = pop[t1] if fitness[t1] >= fitness[t2] else pop[t2]
            
            t1 = rng.randint(0, pop_size)
            t2 = rng.randint(0, pop_size)
            p2 = pop[t1] if fitness[t1] >= fitness[t2] else pop[t2]
            
            # Crossover (blend)
            if rng.random() < crossover_rate:
                alpha = rng.uniform(0.3, 0.7)
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                mask = rng.random(N) < 0.3
                child[mask, :] += rng.normal(0, mut_scale, size=(mask.sum(), 3))
            
            child = _clamp_bary(child)
            new_pop.append(child)
        
        pop = np.array(new_pop[:pop_size])
    
    # Return best individual
    fitness = np.array([_ga_fitness(pop[i]) for i in range(len(pop))])
    best_idx = np.argmax(fitness)
    return pop[best_idx], fitness[best_idx]

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t such that all triangle areas >= t.
    params = [s_0..s_N, t_0..t_N, t] where t is the epigraph variable.
    Uses soft-min to make it smooth.
    """
    n_pts = 2 * N
    s = params[0::2][:N]
    t_params = params[1::2][:N]
    t_epigraph = params[2*N]
    
    s = np.clip(s, 0.0, 1.0)
    t_params = np.clip(t_params, 0.0, 1.0)
    
    lam1 = s * (1.0 - t_params)
    lam2 = s * t_params
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    points = np.column_stack([x, y])
    
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft penalty for areas < t_epigraph
    deficit = np.maximum(0, t_epigraph - areas)
    penalty = np.sum(deficit**2)
    
    # Also penalize if t_epigraph is too large (we want to maximize it, so minimize -t)
    return -t_epigraph + 10.0 * penalty

def _epigraph_grad(params, beta=50.0):
    """Gradient of the epigraph objective."""
    n_pts = 2 * N
    s = params[0::2][:N]
    t_params = params[1::2][:N]
    t_epigraph = params[2*N]
    
    s = np.clip(s, 0.0, 1.0)
    t_params = np.clip(t_params, 0.0, 1.0)
    
    lam1 = s * (1.0 - t_params)
    lam2 = s * t_params
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    points = np.column_stack([x, y])
    
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    deficit = np.maximum(0, t_epigraph - areas)
    # d(penalty)/d(area) = -2 * deficit (when deficit > 0)
    d_penalty_d_area = -2.0 * deficit
    
    grad_points = np.zeros((N, 2))
    
    d_cross_d_pi = np.column_stack([-dy_k, dx_k])
    contrib_i = d_penalty_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pi
    np.add.at(grad_points, _IDX_I, contrib_i)
    
    d_cross_d_pj = np.column_stack([dy_k, -dx_k])
    contrib_j = d_penalty_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pj
    np.add.at(grad_points, _IDX_J, contrib_j)
    
    d_cross_d_pk = np.column_stack([-dy_i, dx_i])
    contrib_k = d_penalty_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pk
    np.add.at(grad_points, _IDX_K, contrib_k)
    
    # Map to params
    dx_ds = (1.0 - t_params) * V1[0] + t_params * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t_params) * V1[1] + t_params * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_s = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_t = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    grad_params = np.zeros(2 * N + 1)
    grad_params[0::2] = grad_s
    grad_params[1::2] = grad_t
    grad_params[2*N] = -1.0  # d(-t)/dt = -1
    
    return grad_params

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
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
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _softmin_objective(params, beta=100.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _softmin_objective_adaptive(params, beta_start=30.0, beta_end=200.0, progress=0.0):
    """Soft-min with adaptive beta based on optimization progress."""
    beta = beta_start + (beta_end - beta_start) * progress
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _softmin_grad(params, beta=100.0):
    """Gradient of the soft-min objective."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(area_m) = -exp_term[m] / sum_exp
    d_softmin_d_area = -exp_term / sum_exp
    
    # d(area_m)/d(point)
    # area = 0.5 * |cross|
    # sign = sign(cross)
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    # d(cross)/d(pi) = (-(dy_k), dx_k)
    # d(cross)/d(pj) = (dy_k, -dx_k)
    # d(cross)/d(pk) = (-dy_i, dx_i)
    
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_d_pi = np.column_stack([-dy_k, dx_k])
    contrib_i = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pi
    np.add.at(grad_points, _IDX_I, contrib_i)
    
    # Contribution from pj
    d_cross_d_pj = np.column_stack([dy_k, -dx_k])
    contrib_j = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pj
    np.add.at(grad_points, _IDX_J, contrib_j)
    
    # Contribution from pk
    d_cross_d_pk = np.column_stack([-dy_i, dx_i])
    contrib_k = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pk
    np.add.at(grad_points, _IDX_K, contrib_k)
    
    # Now map grad_points to grad_params
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_s = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_t = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_s
    grad_params[1::2] = grad_t
    
    # Objective is -soft_min, so gradient is -grad_softmin
    return -grad_params

def _polish_high_precision(params, deadline):
    """
    High-precision polish using epigraph formulation followed by exact Nelder-Mead.
    Returns improved params.
    """
    best_params = params.copy()
    best_score = -_objective(params)
    
    # Phase 1: Epigraph optimization with TRF
    if time.time() < deadline - 10.0:
        try:
            # Initialize t_epigraph to current min area
            pts = _param_to_points(params)
            t_init = _min_triangle_area_vectorized(pts)
            epigraph_params = np.append(params, t_init)
            
            bounds_ep = [(0.0, 1.0)] * (2*N) + [(0.0, TRI_AREA)]
            
            res = minimize(
                _epigraph_objective, epigraph_params,
                jac=_epigraph_grad,
                method='TRF', bounds=bounds_ep,
                options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            epigraph_pts = _param_to_points(res.x[:2*N])
            epigraph_score = _min_triangle_area_vectorized(epigraph_pts)
            
            if epigraph_score > best_score:
                best_score = epigraph_score
                best_params = res.x[:2*N].copy()
        except Exception:
            pass
    
    # Phase 2: Exact Nelder-Mead polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Micro-polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
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

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: evaluator kills at 150s
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    best_bary = None
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 10
    for i in range(n_fps_inits):
        if time_left() < 60.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 60.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(10):
        if time_left() < 60.0:
            break
        s = rng.uniform(0.05, 0.95, size=N)
        t = rng.uniform(0.05, 0.95, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP on the exact objective
    for start_p in initial_params:
        if time_left() < 30.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Quick Nelder-Mead polish on best
    if best_params is not None and time_left() > 40.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 4. Genetic Algorithm phase to escape local minima
    if time_left() > 40.0:
        try:
            # Build initial population for GA
            ga_pop = []
            
            # Seed with best solution found so far
            if best_params is not None:
                best_pts = _param_to_points(best_params)
                best_bary_init = _cart_to_bary(best_pts)
                ga_pop.append(best_bary_init)
            
            # Add FPS-based inits
            for i in range(6):
                if time_left() < 30.0:
                    break
                pts = _generate_fps_initialization(rng)
                bary = _cart_to_bary(pts)
                ga_pop.append(_clamp_bary(bary))
            
            # Add random inits
            for i in range(12):
                if time_left() < 30.0:
                    break
                bary = np.zeros((N, 3))
                for n_idx in range(N):
                    bary[n_idx] = rng.dirichlet(np.ones(3))
                ga_pop.append(bary)
            
            # Add structured inits
            for pts in _structured_inits(rng):
                if time_left() < 30.0:
                    break
                bary = _cart_to_bary(pts)
                ga_pop.append(_clamp_bary(bary))
            
            # Fill to pop_size
            pop_size = 40
            while len(ga_pop) < pop_size:
                if time_left() < 25.0:
                    break
                bary = np.zeros((N, 3))
                for n_idx in range(N):
                    bary[n_idx] = rng.dirichlet(np.ones(3))
                ga_pop.append(bary)
            
            ga_pop = np.array(ga_pop[:pop_size])
            
            # Run GA with more generations
            ga_deadline = start_time + min(deadline, start_time + time_left() - 15.0)
            ga_best_bary, ga_fitness = _ga_evolve(
                ga_pop, rng, n_gen=35, pop_size=pop_size,
                elite_frac=0.15, crossover_rate=0.75, mutation_rate=0.12,
                mut_scale=0.05, deadline=ga_deadline
            )
            
            # Evaluate GA result with exact objective
            ga_pts = _bary_to_cart(ga_best_bary)
            ga_exact_score = _min_triangle_area_vectorized(ga_pts)
            
            if ga_exact_score > best_score:
                best_score = ga_exact_score
                best_bary = ga_best_bary
                best_params = _points_to_param(ga_pts)
            elif best_bary is None:
                best_bary = ga_best_bary
                best_params = _points_to_param(ga_pts)
                
        except Exception:
            pass

    # 5. Polish GA result (or best) with SLSQP
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='SLSQP', bounds=bounds,
                options={'maxiter': 250, 'ftol': 1e-11}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. Try SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(5):
            if time_left() < 18.0:
                break
            try:
                noise = rng.normal(0, 0.012, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # 7. Trust-Region Reflective on the smooth surrogate (log-sum-exp)
    if best_params is not None and time_left() > 25.0:
        try:
            res = minimize(
                _softmin_objective, 
                best_params, 
                jac=_softmin_grad,
                method='TRF', 
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-9, 'xtol': 1e-11}
            )
            exact_score = -_objective(res.x)
            if exact_score > best_score:
                best_score = exact_score
                best_params = res.x.copy()
        except Exception:
            pass

    # 8. High-precision polish using epigraph + Nelder-Mead
    if best_params is not None and time_left() > 15.0:
        try:
            polished = _polish_high_precision(best_params, deadline)
            polished_score = -_objective(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished
        except Exception:
            pass

    # 9. Final micro-polish
    if best_params is not None and time_left() > 5.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Final fallback
    if best_params is None:
        pts = _generate_fps_initialization(rng)
        best_params = _points_to_param(pts)
    
    best_points = _param_to_points(best_params)
    
    # Safety check: ensure all points are inside the triangle
    for i in range(len(best_points)):
        x, y = best_points[i]
        if y < -1e-10:
            best_points[i, 1] = 1e-10
        if y > H + 1e-10:
            best_points[i, 1] = H - 1e-10
        if x < -1e-10:
            best_points[i, 0] = 1e-10
        if x > 1.0 + 1e-10:
            best_points[i, 0] = 1.0 - 1e-10
        # Check left edge: y <= sqrt(3)*x
        if y > S3 * x + 1e-10:
            # Project onto the edge
            best_points[i, 1] = S3 * x - 1e-10
        # Check right edge: y <= sqrt(3)*(1-x)
        if y > S3 * (1.0 - x) + 1e-10:
            best_points[i, 1] = S3 * (1.0 - x) - 1e-10
    
    return np.array(best_points)
