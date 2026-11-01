"""Heilbronn-type problem: place N points to maximize min triangle area / hull area."""
import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14
DEADLINE = 145.0  # seconds, safety margin before 150s kill

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """Compute negative smooth-min score and gradient for optimization."""
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
    
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _refine_full_space(initial_points, start_time, deadline):
    """Refine points in full 28D space using smooth-min gradient optimization."""
    x0 = initial_points.flatten()
    
    # Multi-stage beta annealing
    betas = [30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
    
    x = x0.copy()
    for beta in betas:
        if time.time() > deadline - 8:
            break
        
        def _obj_and_grad(x_flat, beta_val=beta):
            return _neg_smooth_score_and_grad(x_flat, beta_val)
        
        try:
            res = scipy.optimize.minimize(
                _obj_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 2)

def _simulated_annealing_refine(initial_points, start_time, deadline):
    """Refine using simulated annealing in full 28D space."""
    x = initial_points.flatten()
    current_score = _score(x.reshape(-1, 2))
    best_x = x.copy()
    best_score = current_score
    
    # Start with larger perturbations, cool down
    T_start = 0.1
    T_end = 1e-6
    n_iter = 2000
    
    for i in range(n_iter):
        if time.time() > deadline - 3:
            break
        
        # Linear cooling schedule
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Perturbation size proportional to temperature
        scale = T * 0.5
        x_new = x + scale * np.random.randn(2 * N)
        
        # Keep points within reasonable bounds
        x_new = np.clip(x_new, -2.0, 2.0)
        
        new_score = _score(x_new.reshape(-1, 2))
        
        if new_score > current_score:
            x = x_new
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_x = x_new.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / (T * 0.1 + 1e-10))
                if np.random.random() < prob:
                    x = x_new
                    current_score = new_score
    
    return best_x.reshape(-1, 2)

def _genetic_algorithm_search(start_time, deadline, pop_size=30, n_gen=50):
    """Use a genetic algorithm for global search in the full 28D space."""
    # Initialize population with diverse starting points
    population = []
    
    # Regular polygon
    population.append(_regular_polygon(N).flatten())
    
    # Two-ring configurations
    for n1 in [5, 6, 7]:
        n2 = N - n1
        for r2 in [0.3, 0.5, 0.7]:
            pts1 = _regular_polygon(n1, 1.0, 0.0)
            pts2 = _regular_polygon(n2, r2, np.pi/n2)
            population.append(np.vstack([pts1, pts2]).flatten())
    
    # Random points
    for _ in range(pop_size - len(population)):
        population.append(np.random.uniform(-1.5, 1.5, 2*N))
    
    # Evaluate initial population
    scores = np.array([_score(p.reshape(-1, 2)) for p in population])
    
    best_x = population[np.argmax(scores)].copy()
    best_score = np.max(scores)
    
    for gen in range(n_gen):
        if time.time() > deadline - 5:
            break
        
        # Select parents using tournament selection
        new_population = []
        
        # Elitism: keep top 2
        top_indices = np.argsort(scores)[-2:]
        new_population.append(population[top_indices[0]].copy())
        new_population.append(population[top_indices[1]].copy())
        
        # Crossover and mutation to fill rest of population
        while len(new_population) < pop_size:
            # Tournament selection for two parents
            idx1 = np.random.choice(len(population), 3, replace=False)
            idx1 = idx1[np.argmax(scores[idx1])]
            idx2 = np.random.choice(len(population), 3, replace=False)
            idx2 = idx2[np.argmax(scores[idx2])]
            
            p1 = population[idx1]
            p2 = population[idx2]
            
            # Uniform crossover
            mask = np.random.random(2*N) > 0.5
            child = np.where(mask, p1, p2)
            
            # Gaussian mutation
            mutation_scale = 0.05 * (1 - gen/n_gen) + 0.01
            child += mutation_scale * np.random.randn(2*N)
            child = np.clip(child, -2.0, 2.0)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = np.array([_score(p.reshape(-1, 2)) for p in population])
        
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_x = population[gen_best_idx].copy()
    
    return best_x.reshape(-1, 2), best_score

def _differential_evolution_search(start_time, deadline, pop_size=25):
    """Use differential evolution for global search in the full 28D space."""
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def _de_objective(x_flat):
        points = x_flat.reshape(-1, 2)
        sc = _score(points)
        return -sc
    
    try:
        result = scipy.optimize.differential_evolution(
            _de_objective,
            bounds,
            maxiter=80,
            popsize=pop_size,
            tol=1e-10,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        points = result.x.reshape(-1, 2)
        sc = _score(points)
        return points, sc
    except Exception:
        pass
    
    return None, -1e6

def _local_search_refine(points, start_time, deadline, n_iterations=1000):
    """Local search with adaptive step size."""
    x = points.flatten()
    current_score = _score(x.reshape(-1, 2))
    best_x = x.copy()
    best_score = current_score
    
    step_size = 0.05
    
    for i in range(n_iterations):
        if time.time() > deadline - 3:
            break
        
        improved = False
        for _ in range(10):  # Try 10 random directions per iteration
            direction = np.random.randn(2*N)
            direction /= np.linalg.norm(direction)
            
            x_new = x + step_size * direction
            x_new = np.clip(x_new, -2.0, 2.0)
            
            new_score = _score(x_new.reshape(-1, 2))
            
            if new_score > current_score:
                x = x_new
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_x = x_new.copy()
                break
        
        if not improved:
            step_size *= 0.9  # Reduce step size if no improvement
            if step_size < 1e-4:
                break
    
    return best_x.reshape(-1, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
        return sc
    
    # 1. Start with regular polygon as baseline
    reg_pts = _regular_polygon(N)
    _update_best(reg_pts)
    
    # 2. Differential evolution for global search
    if time.time() < deadline - 50:
        de_pts, de_score = _differential_evolution_search(start_time, deadline - 50)
        if de_pts is not None:
            _update_best(de_pts)
    
    # 3. Genetic algorithm search
    if time.time() < deadline - 40:
        ga_pts, ga_score = _genetic_algorithm_search(start_time, deadline - 40)
        _update_best(ga_pts)
    
    # 4. If we have a good candidate, refine in full space with gradient
    if best_points is not None and time.time() < deadline - 30:
        refined = _refine_full_space(best_points, start_time, deadline - 30)
        _update_best(refined)
    
    # 5. Simulated annealing refinement
    if best_points is not None and time.time() < deadline - 15:
        sa_pts = _simulated_annealing_refine(best_points, start_time, deadline - 15)
        _update_best(sa_pts)
    
    # 6. Local search refinement
    if best_points is not None and time.time() < deadline - 10:
        ls_pts, ls_score = _local_search_refine(best_points, start_time, deadline - 10)
        _update_best(ls_pts)
    
    # 7. Final fine-tuning with small random perturbations
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            pts = best_points + 0.02 * np.random.randn(N, 2)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
    
    # 8. Final fallback: regular polygon if nothing worked
    if best_points is None:
        best_points = _regular_polygon(N)
        best_score = _score(best_points)
    
    return best_points
