"""Rewritten program: max-min optimization with penalty-based approach and gradient descent."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    n = len(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area of triangle using cross product
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_tri_area:
            min_tri_area = area
    
    return min_tri_area / hull_area


def _min_triangle_area_and_indices(points):
    """Return min triangle area and the indices of that triangle."""
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_area:
            min_area = area
            min_idx = (i, j, k)
    return min_area, min_idx


def _fermat_spiral(n, seed=0):
    """Generate points using Fermat's spiral (Vogel's model)."""
    rng = np.random.RandomState(seed)
    i = np.arange(n)
    i_perturbed = i + rng.uniform(-0.1, 0.1, n)
    r = np.sqrt(i_perturbed / n)
    theta = i_perturbed * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points


def _circle_initialization(n, seed=0):
    """Generate points on a circle with small perturbations."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += rng.uniform(-0.05, 0.05, n)
    radii = 1.0 + rng.uniform(-0.02, 0.02, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points


def _random_initialization(n, seed=0):
    """Generate random points in a unit square."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-0.5, 0.5, (n, 2))
    return points


def _penalty_objective(points, target_t, penalty_weight):
    """
    Penalty-based objective: maximize min triangle area / hull_area,
    but penalize triangles with area below target_t * hull_area.
    """
    n = len(points)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return -1e10
    
    if hull_area <= 1e-12:
        return -1e10
    
    min_tri_area = float('inf')
    total_penalty = 0.0
    
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_tri_area:
            min_tri_area = area
        
        # Penalty for triangles below target
        target_area = target_t * hull_area
        if area < target_area:
            deficit = target_area - area
            total_penalty += penalty_weight * deficit ** 2
    
    score = min_tri_area / hull_area
    return score - total_penalty / hull_area


def _gradient_ascent(points, target_t, penalty_weight, lr=0.01, max_iter=500, 
                     time_limit=None, start_time=None):
    """
    Gradient ascent using numerical gradients on the penalty objective.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 30.0
    
    current = points.copy()
    current_obj = _penalty_objective(current, target_t, penalty_weight)
    best = current.copy()
    best_obj = current_obj
    
    eps = 1e-5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute numerical gradient
        grad = np.zeros_like(current)
        for idx in range(len(current)):
            for dim in range(2):
                perturbed = current.copy()
                perturbed[idx, dim] += eps
                obj_plus = _penalty_objective(perturbed, target_t, penalty_weight)
                
                perturbed = current.copy()
                perturbed[idx, dim] -= eps
                obj_minus = _penalty_objective(perturbed, target_t, penalty_weight)
                
                grad[idx, dim] = (obj_plus - obj_minus) / (2 * eps)
        
        # Gradient ascent step
        current = current + lr * grad
        
        new_obj = _penalty_objective(current, target_t, penalty_weight)
        
        if new_obj > best_obj:
            best = current.copy()
            best_obj = new_obj
        
        # Adapt learning rate
        if new_obj < current_obj:
            lr *= 0.9
        elif new_obj > current_obj * 1.01:
            lr = min(lr * 1.05, 0.1)
        
        current_obj = new_obj
    
    return best, best_obj


def _simulated_annealing(points, max_iter=20000, t_start=0.05, t_end=0.001,
                         cooling_rate=0.99995, rng_seed=42, time_limit=40.0,
                         start_time=None):
    """Simulated annealing with 2-point moves."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Cool temperature
        t = t_start * (cooling_rate ** i)
        if t < t_end:
            break
        
        # Pick 2 random points to move
        idx = rng.choice(len(current), size=2, replace=False)
        
        # Generate new positions with small perturbation
        new_points = current.copy()
        for idx_i in idx:
            new_points[idx_i] = current[idx_i] + rng.normal(0, 0.03, 2)
        
        new_score = _compute_score(new_points)
        
        # Accept if better or with probability exp((new - current) / t)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    current = new_points
                    current_score = new_score
    
    return best, best_score


def _local_search(points, max_iter=2000, step_size=0.005, rng_seed=42,
                  time_limit=20.0, start_time=None):
    """Local search with single-point moves."""
    if start_time is None:
        start_time = time.time()
    
    rng = np.random.RandomState(rng_seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, len(current) - 1)
        
        # Try small perturbations
        improved = False
        for _ in range(15):
            new_points = current.copy()
            new_points[idx] = current[idx] + rng.normal(0, step_size, 2)
            new_score = _compute_score(new_points)
            
            if new_score > current_score:
                current = new_points
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                break
        
        if not improved:
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    return best, best_score


def _refine_with_penalty(points, time_budget, start_time=None):
    """
    Use penalty-based gradient ascent to refine points.
    Start with a target slightly above current score and iterate.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    current_score = _compute_score(current)
    
    # Start with target at current score
    target_t = current_score
    penalty_weight = 1.0
    
    for iteration in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Run gradient ascent
        refined, refined_obj = _gradient_ascent(
            current, target_t, penalty_weight,
            lr=0.005, max_iter=200,
            time_limit=min(10.0, time_budget - (time.time() - start_time)),
            start_time=start_time
        )
        
        refined_score = _compute_score(refined)
        
        if refined_score > current_score:
            current = refined
            current_score = refined_score
            # Increase target
            target_t = current_score * 1.01
            penalty_weight *= 1.5
        else:
            # Keep target, increase penalty
            penalty_weight *= 2.0
    
    return current, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate diverse initializations
    initializations = [
        _fermat_spiral(N, seed=0),
        _fermat_spiral(N, seed=1),
        _fermat_spiral(N, seed=2),
        _circle_initialization(N, seed=0),
        _circle_initialization(N, seed=1),
        _random_initialization(N, seed=0),
        _random_initialization(N, seed=1),
        _random_initialization(N, seed=2),
    ]
    
    best_points = None
    best_score = -float('inf')
    
    # First pass: evaluate all initializations
    for init in initializations:
        score = _compute_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()
    
    # Sort initializations by score
    scored_inits = []
    for init in initializations:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Process top initializations with SA + penalty refinement
    for i, (init_score, init) in enumerate(scored_inits[:5]):
        if time.time() > deadline - 25:
            break
        
        # Run SA
        sa_points, sa_score = _simulated_annealing(
            init, max_iter=15000, t_start=0.04, t_end=0.001,
            cooling_rate=0.99995, rng_seed=100 + i,
            time_limit=25.0, start_time=start_time
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Run penalty refinement
        if time.time() > deadline - 15:
            break
        
        time_budget = min(15.0, deadline - time.time() - 5.0)
        refined_points, refined_score = _refine_with_penalty(
            sa_points, time_budget, start_time=start_time
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
        
        # Final local search
        if time.time() > deadline - 8:
            break
        
        ls_points, ls_score = _local_search(
            best_points, max_iter=1000, step_size=0.003, rng_seed=200 + i,
            time_limit=min(8.0, deadline - time.time() - 3.0),
            start_time=start_time
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    # Final local search on the best point found
    if time.time() > deadline - 5:
        if best_points is not None:
            return best_points
        return initializations[0]
    
    final_points, final_score = _local_search(
        best_points, max_iter=1500, step_size=0.002, rng_seed=300,
        time_limit=min(10.0, deadline - time.time() - 2.0),
        start_time=start_time
    )
    
    if final_score > best_score:
        best_points = final_points
        best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
