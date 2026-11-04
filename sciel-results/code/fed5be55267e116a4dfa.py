"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

N = 13

# EVOLVE-BLOCK-START

def _triangle_area(p, q, r):
    """Compute area of triangle formed by points p, q, r."""
    return 0.5 * abs(np.cross(q - p, r - p))

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0.0 for degenerate cases.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    
    # Check for degenerate points
    if n < 3:
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                if min_area < 1e-10:
                    return 0.0
    
    return min_area / hull_area

def _differential_evolution(objective, bounds, maxiter=50, popsize=10, 
                            tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
                            seed=None, polish=True):
    """
    Simple differential evolution implementation.
    """
    import random
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lb, ub, size=(popsize, dim))
    # Use seeded random for reproducibility
    pop = np.array([lb + (ub - lb) * rng.random_sample() for _ in range(popsize)])
    
    best_idx = 0
    best_fitness = objective(pop[0])
    for i in range(1, popsize):
        fit = objective(pop[i])
        if fit < best_fitness:
            best_fitness = fit
            best_idx = i
    
    best = pop[best_idx].copy()
    
    for gen in range(maxiter):
        # Check convergence
        if gen > 10 and abs(objective(best) - best_fitness) < tol:
            break
        
        new_best = best.copy()
        new_best_fitness = best_fitness
        
        for i in range(popsize):
            # Select three distinct random indices
            while True:
                a, b, c = rng.sample(range(popsize), 3)
                if len(set([a, b, c, i])) == 4:
                    break
            
            # Mutation
            v = pop[a] + mutation[0] * (pop[b] - pop[c])
            
            # Recombination
            trial = pop[i].copy()
            jrand = rng.randint(0, dim - 1)
            for j in range(dim):
                if j == jrand or rng.random() < recombination:
                    trial[j] = v[j]
            
            # Bound check
            trial = np.clip(trial, lb, ub)
            
            # Selection
            trial_fitness = objective(trial)
            if trial_fitness < pop[i] and trial_fitness < best_fitness:
                pop[i] = trial
                if trial_fitness < new_best_fitness:
                    new_best_fitness = trial_fitness
                    new_best = trial.copy()
        
        best = new_best
        best_fitness = new_best_fitness
    
    if polish:
        from scipy.optimize import minimize
        result = minimize(objective, best, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_fitness:
            best = result.x
            best_fitness = result.fun
    
    return best, best_fitness

def _polar_to_cartesian(log_dists, angles):
    """
    Convert polar dual representation (log distances and angles) to Cartesian points.
    The points are vertices of the convex polygon defined by intersection of half-planes.
    """
    n = len(angles)
    d = np.exp(log_dists)
    
    # Build the system of equations for each edge
    # Each edge i is defined by: cos(angle_i)*x + sin(angle_i)*y = d[i]
    # The vertex i is the intersection of edges i-1 and i
    
    points = np.zeros((n, 2))
    for i in range(n):
        i1 = i
        i2 = (i - 1) % n
        
        a1 = np.array([np.cos(angles[i1]), np.sin(angles[i1])])
        a2 = np.array([np.cos(angles[i2]), np.sin(angles[i2])])
        
        # Solve the 2x2 system
        A = np.array([a1, a2])
        b = np.array([d[i1], d[i2]])
        
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                # Degenerate, use fallback
                points[i] = a1 * d[i1]
            else:
                points[i] = np.linalg.solve(A, b)
        except Exception:
            points[i] = a1 * d[i1]
    
    return points

def _objective_from_params(params):
    """
    Objective function for optimization.
    params = [log_dists..., angles...]
    """
    n = N
    log_dists = params[:n]
    angles = params[n:]
    
    # Ensure angles are sorted and cover full circle
    angles = np.sort(angles)
    
    # Convert to Cartesian
    points = _polar_to_cartesian(log_dists, angles)
    
    # Score (negated for minimization)
    score = _score(points)
    return -score

def construct():
    """
    Construct 13 points to maximize minimum triangle area / convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial angles: equally spaced
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Bounds for log distances and angle perturbations
    log_d_bounds = [(-1.0, 1.0)] * N
    angle_bounds = [(-0.3, 0.3)] * N  # Perturbations from base angles
    
    bounds = log_d_bounds + angle_bounds
    
    # Multi-start differential evolution
    best_points = None
    best_score = -np.inf
    
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        # Create initial population centered around different perturbations
        seed = 42 + s
        
        # For each start, use different initial angle perturbations
        angle_perturbations = np.zeros(N)
        if s > 0:
            rng = np.random.RandomState(seed)
            angle_perturbations = rng.uniform(-0.2, 0.2, size=N)
        
        # Define objective with fixed angle base
        def obj(params):
            log_dists = params[:N]
            angle_perts = params[N:]
            angles = base_angles + angle_perts
            angles = np.sort(angles)
            # Wrap angles to [0, 2*pi)
            angles = angles % (2 * np.pi)
            angles = np.sort(angles)
            
            points = _polar_to_cartesian(log_dists, angles)
            score = _score(points)
            return -score
        
        try:
            result_x, result_f = _differential_evolution(
                obj, bounds, maxiter=30, popsize=8, 
                tol=1e-5, mutation=(0.5, 1.0), recombination=0.7,
                seed=seed, polish=True
            )
            
            if result_f < best_score:
                best_score = result_f
                log_dists = result_x[:N]
                angle_perts = result_x[N:]
                angles = np.sort(base_angles + angle_perts)
                angles = angles % (2 * np.pi)
                angles = np.sort(angles)
                best_points = _polar_to_cartesian(log_dists, angles)
                
        except Exception as e:
            pass
    
    # Fallback to regular polygon if optimization failed
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        d = np.ones(N)
        best_points = _polar_to_cartesian(np.zeros(N), angles)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_points)
        area = hull.volume
        if area > 1e-10:
            best_points = best_points / np.sqrt(area)
    except Exception:
        pass
    
    return best_points

# EVOLVE-BLOCK-END
