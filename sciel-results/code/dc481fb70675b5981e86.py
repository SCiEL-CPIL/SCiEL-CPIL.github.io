"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _ratio(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area(points)
    ha = _convex_hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _local_refine(points, max_iters=200, step_size=0.01):
    """Local refinement using coordinate descent with adaptive step."""
    best_points = points.copy()
    best_score = _ratio(best_points)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for dim in range(2):
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    trial = best_points.copy()
                    trial[i, dim] += delta
                    score = _ratio(trial)
                    if score > best_score:
                        best_points = trial
                        best_score = score
                        improved = True
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    return best_points, best_score

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Approach: Differential evolution global search + local coordinate descent refinement.
    """
    start_time = time.time()
    timeout = 115.0
    
    # Differential evolution parameters
    pop_size = 40
    max_generations = 50
    crossover_rate = 0.8
    mutation_scale = 0.5
    
    np.random.seed(42)
    
    # Initialize population
    population = np.zeros((pop_size, N, 2))
    scores = np.zeros(pop_size)
    
    for i in range(pop_size):
        if i == 0:
            # Regular polygon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            population[i] = np.column_stack([np.cos(angles), np.sin(angles)])
        elif i == 1:
            # Slightly perturbed regular polygon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            base = np.column_stack([np.cos(angles), np.sin(angles)])
            population[i] = base + np.random.normal(0, 0.1, (N, 2))
        else:
            # Random points in unit square
            population[i] = np.random.rand(N, 2) * 2 - 1
        scores[i] = _ratio(population[i])
    
    # Differential evolution loop
    for gen in range(max_generations):
        if time.time() - start_time > timeout - 20:
            break
        
        for i in range(pop_size):
            if time.time() - start_time > timeout - 20:
                break
            
            # Select three distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = population[r1] + mutation_scale * (population[r2] - population[r3])
            
            # Crossover
            trial = population[i].copy()
            for j in range(N):
                for k in range(2):
                    if np.random.rand() < crossover_rate:
                        trial[j, k] = mutant[j, k]
            
            # Ensure at least one gene is from mutant
            if not np.any(trial != population[i]):
                j_rand = np.random.randint(N)
                trial[j_rand, 0] = mutant[j_rand, 0]
            
            # Evaluate
            trial_score = _ratio(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score
    
    # Get best from population
    best_idx = np.argmax(scores)
    best_points = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Local refinement
    if time.time() - start_time < timeout - 10:
        best_points, best_score = _local_refine(best_points, max_iters=100, step_size=0.005)
    
    # Additional local refinement from a few other good candidates
    if time.time() - start_time < timeout - 5:
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices[1:4]:
            if time.time() - start_time > timeout - 5:
                break
            cand_points, cand_score = _local_refine(population[idx].copy(), max_iters=50, step_size=0.005)
            if cand_score > best_score:
                best_points = cand_points
                best_score = cand_score
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
