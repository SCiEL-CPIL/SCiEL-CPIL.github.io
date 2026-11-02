"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Island Model Genetic Algorithm (GA) to explore the configuration space.
2. Representation: Cartesian coordinates with projection to ensure validity.
3. Fitness: Exact minimum triangle area (non-differentiable, but robust).
4. Operators:
   - Initialization: Greedy placement followed by random perturbations.
   - Mutation: Random walk with adaptive step size.
   - Crossover: Uniform crossover between parents from different islands.
5. Parallel evaluation of multiple islands to maintain diversity.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _greedy_init(rng):
    """
    Initialize points using a greedy strategy:
    Place points one by one to maximize the minimum distance to existing points.
    This provides a good starting configuration for the GA.
    """
    pts = []
    # Start with a random point
    pts.append(np.array(_random_point_in_triangle(rng)))
    
    for _ in range(N - 1):
        best_point = None
        best_score = -1.0
        
        # Sample candidate points
        for _ in range(100):
            cand = np.array(_random_point_in_triangle(rng))
            # Calculate minimum distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_dist = np.min(dists)
            
            # We want to maximize the minimum distance
            if min_dist > best_score:
                best_score = min_dist
                best_point = cand
        
        if best_point is not None:
            pts.append(best_point)
    
    return np.array(pts)

def _mutate(pts, step_size, rng):
    """
    Mutate a configuration by adding Gaussian noise to each point.
    """
    new_pts = pts.copy()
    noise = rng.gauss(0, step_size, size=new_pts.shape)
    new_pts += noise
    return _project_to_triangle(new_pts)

def _crossover(p1, p2, rng):
    """
    Uniform crossover: for each point index, randomly choose from parent 1 or 2.
    """
    new_pts = np.zeros_like(p1)
    for i in range(N):
        if rng.random() < 0.5:
            new_pts[i] = p1[i]
        else:
            new_pts[i] = p2[i]
    return _project_to_triangle(new_pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses an Island Model Genetic Algorithm.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Island configuration
    n_islands = 4
    pop_size = 10
    migration_rate = 0.1  # Fraction of population to migrate
    mutation_rate = 0.8   # Probability of mutation per individual
    crossover_rate = 0.5  # Probability of crossover
    
    # Initialize islands
    islands = []
    for _ in range(n_islands):
        pop = []
        # First individual: Greedy init
        pop.append(_greedy_init(rng))
        # Rest: Random init with some perturbation
        for _ in range(pop_size - 1):
            base = _greedy_init(rng)
            perturbed = _mutate(base, 0.05, rng)
            pop.append(perturbed)
        
        # Evaluate fitness
        fitnesses = [_min_triangle_area(p) for p in pop]
        islands.append({'pop': pop, 'fit': fitnesses, 'best': pop[0], 'best_fit': fitnesses[0]})
    
    # Find global best
    global_best = islands[0]['best']
    global_best_fit = islands[0]['best_fit']
    for isl in islands:
        if isl['best_fit'] > global_best_fit:
            global_best = isl['best']
            global_best_fit = isl['best_fit']
    
    step_size = 0.05
    gen = 0
    
    while time.time() - start_time < DEADLINE - 5:
        gen += 1
        
        # Adapt step size: decrease if global best hasn't improved recently
        # (Simplified: just decay slowly)
        step_size *= 0.995
        if step_size < 0.001:
            step_size = 0.001
            
        for i in range(n_islands):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            island = islands[i]
            pop = island['pop']
            fit = island['fit']
            
            # Sort population by fitness
            indices = np.argsort(fit)[::-1]
            pop = [pop[j] for j in indices]
            fit = [fit[j] for j in indices]
            
            # Selection: Keep top half, replace bottom half with offspring
            n_offspring = pop_size // 2
            new_pop = pop[:pop_size - n_offspring]
            
            for _ in range(n_offspring):
                # Tournament selection for parents
                p1_idx = rng.randint(0, pop_size - 1)
                p2_idx = rng.randint(0, pop_size - 1)
                # Simple selection: just pick random from top half
                p1_idx = rng.randint(0, pop_size // 2 - 1)
                p2_idx = rng.randint(0, pop_size // 2 - 1)
                
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]
                
                # Crossover
                if rng.random() < crossover_rate:
                    child = _crossover(p1, p2, rng)
                else:
                    child = p1.copy()
                
                # Mutation
                if rng.random() < mutation_rate:
                    child = _mutate(child, step_size, rng)
                
                new_pop.append(child)
            
            # Evaluate new population
            new_fit = [_min_triangle_area(p) for p in new_pop]
            
            # Update island best
            best_idx = np.argmax(new_fit)
            if new_fit[best_idx] > island['best_fit']:
                island['best'] = new_pop[best_idx]
                island['best_fit'] = new_fit[best_idx]
                
                # Update global best
                if new_fit[best_idx] > global_best_fit:
                    global_best = new_pop[best_idx].copy()
                    global_best_fit = new_fit[best_idx]
            
            island['pop'] = new_pop
            island['fit'] = new_fit
            
            # Migration: Move best individual from this island to a random other island
            if rng.random() < migration_rate:
                target_island = rng.randint(0, n_islands - 1)
                if target_island != i:
                    # Replace worst individual in target island with best from source
                    target = islands[target_island]
                    worst_idx = np.argmin(target['fit'])
                    target['pop'][worst_idx] = island['best']
                    target['fit'][worst_idx] = island['best_fit']
                    if island['best_fit'] > target['best_fit']:
                        target['best'] = island['best']
                        target['best_fit'] = island['best_fit']
                        if target['best_fit'] > global_best_fit:
                            global_best = target['best']
                            global_best_fit = target['best_fit']
        
        # Final check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final polish: Run a few iterations of local search on the global best
    if time.time() - start_time < DEADLINE - 2:
        current_best = global_best
        current_fit = global_best_fit
        local_step = 0.01
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 2:
                break
            candidate = _mutate(current_best, local_step, rng)
            candidate_fit = _min_triangle_area(candidate)
            if candidate_fit > current_fit:
                current_best = candidate
                current_fit = candidate_fit
                local_step *= 0.9
            else:
                local_step *= 0.95
        
        if current_fit > global_best_fit:
            global_best = current_best
            global_best_fit = current_fit
    
    return global_best

if __name__ == "__main__":
    pts = construct()
    print(f"Best min area: {_min_triangle_area(pts)}")
