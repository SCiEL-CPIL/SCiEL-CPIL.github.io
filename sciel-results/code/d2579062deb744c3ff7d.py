"""Rewritten program with stronger optimization for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Calculate areas of all triangles formed by triples of points."""
    n = points.shape[0]
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    
    indices = np.array(indices)
    pi = points[indices[:, 0]]
    pj = points[indices[:, 1]]
    pk = points[indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        min_area = np.min(_triangle_areas(points))
        return min_area / hull_area
    except Exception:
        return 0.0

def _score_from_angles(angles):
    """Calculate score given angles on the unit circle."""
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return _score(points)

def _negative_score_from_angles(angles):
    """Objective function for minimization."""
    return -_score_from_angles(angles)

def _differential_evolution_step(population, scores, scale=0.5, crossover_rate=0.8, mutation_rate=0.5):
    """
    Perform one step of differential evolution on the population.
    population: list of angle arrays
    scores: list of corresponding scores
    Returns: new population and scores
    """
    n_pop = len(population)
    if n_pop < 3:
        return population, scores
    
    new_population = list(population)
    new_scores = list(scores)
    
    for i in range(n_pop):
        # Select three distinct individuals
        while True:
            a, b, c = np.random.choice(n_pop, 3, replace=False)
            if len({a, b, c, i}) == 4:
                break
        
        # Differential mutation
        mutant = population[a] + scale * (population[b] - population[c])
        # Wrap angles to [0, 2*pi)
        mutant = np.mod(mutant, 2 * np.pi)
        
        # Crossover
        trial = np.copy(population[i])
        for j in range(N):
            if np.random.random() < crossover_rate:
                trial[j] = mutant[j]
        
        # Evaluate
        trial_score = _score_from_angles(trial)
        
        # Selection
        if trial_score >= new_scores[i]:
            new_population[i] = trial
            new_scores[i] = trial_score
    
    return new_population, new_scores

def _simulated_annealing_step(angles, score, temperature, cooling_rate=0.999):
    """
    Perform one step of simulated annealing.
    Returns: new angles, new score, new temperature
    """
    # Generate neighbor by perturbing one or two random angles
    n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
    indices = np.random.choice(N, n_perturb, replace=False)
    
    new_angles = np.copy(angles)
    for idx in indices:
        new_angles[idx] += np.random.normal(0, 0.1)
    new_angles = np.mod(new_angles, 2 * np.pi)
    
    new_score = _score_from_angles(new_angles)
    
    # Accept if better, or accept worse with probability exp((new_score - score) / temperature)
    if new_score >= score:
        return new_angles, new_score, temperature * cooling_rate
    else:
        prob = np.exp((new_score - score) / max(temperature, 1e-10))
        if np.random.random() < prob:
            return new_angles, new_score, temperature * cooling_rate
        else:
            return angles, score, temperature * cooling_rate

def _local_refinement(angles, max_iter=500):
    """
    Perform local refinement using L-BFGS-B.
    Returns: refined angles and score
    """
    try:
        def objective(x):
            return _negative_score_from_angles(x)
        
        res = minimize(objective, angles, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        refined_angles = np.mod(res.x, 2 * np.pi)
        refined_score = _score_from_angles(refined_angles)
        return refined_angles, refined_score
    except Exception:
        return angles, _score_from_angles(angles)

def _generate_initial_population(rng, n_pop=20):
    """
    Generate initial population for differential evolution.
    Mix of regular polygon, random, and structured perturbations.
    """
    population = []
    
    # Regular polygon
    regular = np.linspace(0, 2*np.pi, N, endpoint=False)
    population.append(regular.copy())
    
    # Slightly perturbed regular
    for _ in range(5):
        perturbed = regular + rng.normal(0, 0.05, N)
        population.append(np.mod(perturbed, 2*np.pi))
    
    # Random configurations
    for _ in range(n_pop - 6):
        random_angles = rng.uniform(0, 2*np.pi, N)
        population.append(np.sort(random_angles))
    
    # Structured: alternating close/far pairs
    for _ in range(3):
        base = np.linspace(0, 2*np.pi, N//2, endpoint=False)
        angles = np.concatenate([base, base + np.pi/N + rng.normal(0, 0.02, N//2)])
        population.append(np.mod(np.sort(angles), 2*np.pi))
    
    return population[:n_pop]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = np.random.default_rng(12345)
    
    best_score = -np.inf
    best_angles = None
    
    # Phase 1: Differential Evolution for global search
    n_pop = 20
    population = _generate_initial_population(rng, n_pop)
    scores = [_score_from_angles(angles) for angles in population]
    
    # Track best
    for i, (angles, score) in enumerate(zip(population, scores)):
        if score > best_score:
            best_score = score
            best_angles = angles.copy()
    
    # Run DE iterations
    de_iter = 0
    while time.time() < deadline - 40:  # Reserve 40s for local refinement and SA
        # Check time
        if time.time() > deadline - 40:
            break
        
        population, scores = _differential_evolution_step(
            population, scores, scale=0.5, crossover_rate=0.8, mutation_rate=0.5
        )
        de_iter += 1
        
        # Update best
        for i, (angles, score) in enumerate(zip(population, scores)):
            if score > best_score:
                best_score = score
                best_angles = angles.copy()
        
        # Every 10 iterations, do local refinement on the best
        if de_iter % 10 == 0 and time.time() < deadline - 30:
            refined_angles, refined_score = _local_refinement(best_angles, max_iter=200)
            if refined_score > best_score:
                best_score = refined_score
                best_angles = refined_angles.copy()
                # Replace worst in population with refined
                worst_idx = np.argmin(scores)
                population[worst_idx] = best_angles.copy()
                scores[worst_idx] = best_score
    
    # Phase 2: Simulated Annealing from best
    if time.time() < deadline - 20:
        sa_angles = best_angles.copy()
        sa_score = best_score
        temperature = 0.01
        cooling_rate = 0.9995
        
        sa_iter = 0
        while time.time() < deadline - 15:
            sa_angles, sa_score, temperature = _simulated_annealing_step(
                sa_angles, sa_score, temperature, cooling_rate
            )
            sa_iter += 1
            
            if sa_score > best_score:
                best_score = sa_score
                best_angles = sa_angles.copy()
            
            # Occasionally do local refinement
            if sa_iter % 50 == 0 and time.time() < deadline - 10:
                refined_angles, refined_score = _local_refinement(sa_angles, max_iter=300)
                if refined_score > best_score:
                    best_score = refined_score
                    best_angles = refined_angles.copy()
                    sa_angles = best_angles.copy()
                    sa_score = best_score
    
    # Phase 3: Final local refinement
    if time.time() < deadline - 5:
        refined_angles, refined_score = _local_refinement(best_angles, max_iter=1000)
        if refined_score > best_score:
            best_score = refined_score
            best_angles = refined_angles.copy()
    
    # If all failed, fallback to regular polygon
    if best_angles is None or best_score < 0.1:
        best_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Convert to points
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return final_points

# EVOLVE-BLOCK-END
