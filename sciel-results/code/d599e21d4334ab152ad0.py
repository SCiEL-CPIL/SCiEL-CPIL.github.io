# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from math import atan2, pi

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position.
    Uses vectorized NumPy operations on pre-generated indices.
    """
    n = len(points)
    # Pre-generate indices if not cached
    if not hasattr(_count_convex7, 'indices'):
        _count_convex7.indices = np.array(list(itertools.combinations(range(n), 7)))
    
    indices = _count_convex7.indices
    pts = points[indices]
    
    p0 = pts[:, 0, :]
    p1 = pts[:, 1, :]
    p2 = pts[:, 2, :]
    p3 = pts[:, 3, :]
    p4 = pts[:, 4, :]
    p5 = pts[:, 5, :]
    p6 = pts[:, 6, :]
    
    c0 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p1[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p1[:, 0])
    c1 = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p2[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p2[:, 0])
    c2 = (p3[:, 0] - p2[:, 0]) * (p4[:, 1] - p3[:, 1]) - (p3[:, 1] - p2[:, 1]) * (p4[:, 0] - p3[:, 0])
    c3 = (p4[:, 0] - p3[:, 0]) * (p5[:, 1] - p4[:, 1]) - (p4[:, 1] - p3[:, 1]) * (p5[:, 0] - p4[:, 0])
    c4 = (p5[:, 0] - p4[:, 0]) * (p6[:, 1] - p5[:, 1]) - (p5[:, 1] - p4[:, 1]) * (p6[:, 0] - p5[:, 0])
    c5 = (p6[:, 0] - p5[:, 0]) * (p0[:, 1] - p6[:, 1]) - (p6[:, 1] - p5[:, 1]) * (p0[:, 0] - p6[:, 0])
    c6 = (p0[:, 0] - p6[:, 0]) * (p1[:, 1] - p0[:, 1]) - (p0[:, 1] - p6[:, 1]) * (p1[:, 0] - p6[:, 0])
    
    is_convex = (c0 > 1e-9) & (c1 > 1e-9) & (c2 > 1e-9) & (c3 > 1e-9) & (c4 > 1e-9) & (c5 > 1e-9) & (c6 > 1e-9)
    
    return int(np.sum(is_convex))

def _is_valid(points: np.ndarray) -> bool:
    """Check validity constraints."""
    n = len(points)
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.all(points >= -1e-6) or not np.all(points <= 1 + 1e-6):
        return False
    
    # Check minimum distance
    dist_sq = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    if np.any(dist_sq < 1e-12):
        return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                val = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                if abs(val) < 1e-9:
                    return False
    
    return True

def _make_initial_points() -> np.ndarray:
    """
    Generate initial points clustered in corners to reduce convex polygons.
    """
    n = 33
    np.random.seed(42)
    
    points = np.zeros((n, 2))
    corner_centers = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    sigma = 0.03
    
    for i in range(n):
        corner = corner_centers[i % 4]
        points[i, 0] = corner[0] + np.random.normal(0, sigma)
        points[i, 1] = corner[1] + np.random.normal(0, sigma)
    
    points = np.clip(points, 0.02, 0.98)
    
    if not _is_valid(points):
        for _ in range(100):
            points = points + np.random.uniform(-0.005, 0.005, size=(n, 2))
            points = np.clip(points, 0.02, 0.98)
            if _is_valid(points):
                break
    
    return points

def _ga_step(population, fitness, mutation_rate=0.1, crossover_rate=0.7, elite_size=2):
    """
    Perform one generation of genetic algorithm.
    """
    n_points = 33
    pop_size = len(population)
    
    # Sort by fitness (higher is better)
    sorted_indices = np.argsort(fitness)[::-1]
    sorted_population = population[sorted_indices]
    sorted_fitness = fitness[sorted_indices]
    
    # Keep elites
    new_population = [sorted_population[i].copy() for i in range(min(elite_size, pop_size))]
    
    # Generate rest via crossover and mutation
    while len(new_population) < pop_size:
        # Tournament selection
        idx1 = np.random.randint(pop_size)
        idx2 = np.random.randint(pop_size)
        if sorted_fitness[idx1] > sorted_fitness[idx2]:
            parent1 = sorted_population[idx1]
        else:
            parent1 = sorted_population[idx2]
        
        idx1 = np.random.randint(pop_size)
        idx2 = np.random.randint(pop_size)
        if sorted_fitness[idx1] > sorted_fitness[idx2]:
            parent2 = sorted_population[idx1]
        else:
            parent2 = sorted_population[idx2]
        
        # Crossover
        if np.random.random() < crossover_rate:
            # Uniform crossover
            mask = np.random.random((n_points, 2)) < 0.5
            child = np.where(mask, parent1, parent2)
        else:
            child = parent1.copy()
        
        # Mutation
        if np.random.random() < mutation_rate:
            # Move a random point
            idx = np.random.randint(n_points)
            child[idx, 0] += np.random.normal(0, 0.05)
            child[idx, 1] += np.random.normal(0, 0.05)
            child = np.clip(child, 0.005, 0.995)
        
        new_population.append(child)
    
    return np.array(new_population)

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points
    in general position inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points.
    """
    n = 33
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin
    
    # Pre-generate indices for _count_convex7
    _count_convex7.indices = np.array(list(itertools.combinations(range(n), 7)))
    
    # Initial configuration: cluster points in corners
    points = _make_initial_points()
    count = _count_convex7(points)
    best_points = points.copy()
    best_count = count
    best_value = 1.0 / (1.0 + best_count)
    
    # Genetic algorithm
    pop_size = 10
    population = np.array([points.copy() for _ in range(pop_size)])
    fitness = np.array([1.0 / (1.0 + _count_convex7(population[i])) for i in range(pop_size)])
    
    generation = 0
    max_generations = 500
    
    while time.time() < deadline and generation < max_generations:
        # Evaluate population
        fitness = np.array([1.0 / (1.0 + _count_convex7(population[i])) for i in range(pop_size)])
        
        # Update best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_value:
            best_value = fitness[best_idx]
            best_count = int(1.0 / best_value - 1.0)
            best_points = population[best_idx].copy()
        
        # Next generation
        population = _ga_step(population, fitness)
        generation += 1
    
    # Final hill climbing polish on best solution
    current_points = best_points.copy()
    current_count = best_count
    step_size = 0.01
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 10:
        idx = np.random.randint(n)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = current_points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        new_points = np.clip(new_points, 0.005, 0.995)
        
        if not _is_valid(new_points):
            no_improve += 1
            continue
        
        new_count = _count_convex7(new_points)
        
        if new_count < current_count:
            current_points = new_points.copy()
            current_count = new_count
            if current_count < best_count:
                best_points = current_points.copy()
                best_count = current_count
                best_value = 1.0 / (1.0 + best_count)
            no_improve = 0
            step_size *= 0.95
        else:
            no_improve += 1
            if no_improve % 50 == 0:
                step_size = min(step_size * 1.2, 0.05)
        
        if no_improve >= max_no_improve:
            break
    
    return best_points

# EVOLVE-BLOCK-END
