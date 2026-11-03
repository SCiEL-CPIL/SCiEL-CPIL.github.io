"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time


def compute_radii_for_positions(x, y, n):
    """
    Given positions (x, y), compute the maximum sum of radii subject to:
    - r_i + r_j <= dist_ij for all i < j
    - r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    - r_i >= 0
    
    Uses iterative growth with constraint enforcement.
    Returns (radii, sum_radii, is_valid)
    """
    # Compute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary constraints
    boundary_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Initialize radii small
    radii = np.ones(n) * 0.005
    
    # Iteratively grow radii
    for iteration in range(200):
        updated = False
        for i in range(n):
            # Max radius for circle i given current other radii
            max_r = boundary_r[i]
            for j in range(n):
                if i != j:
                    candidate = (dist[i, j] - radii[j])
                    if candidate < max_r:
                        max_r = candidate
            max_r = max(max_r, 0.0)
            
            # Try to increase radius
            new_r = min(radii[i] * 1.05, max_r)
            if new_r > radii[i] + 1e-10:
                radii[i] = new_r
                updated = True
        
        if not updated:
            break
    
    # Verify validity: check all constraints
    is_valid = True
    for i in range(n):
        if radii[i] < 0:
            is_valid = False
            break
        if radii[i] > x[i] + 1e-9 or radii[i] > (1 - x[i]) + 1e-9 or \
           radii[i] > y[i] + 1e-9 or radii[i] > (1 - y[i]) + 1e-9:
            is_valid = False
            break
        for j in range(i + 1, n):
            if radii[i] + radii[j] > dist[i, j] + 1e-9:
                is_valid = False
                break
        if not is_valid:
            break
    
    # If invalid, scale down
    if not is_valid:
        scale = 0.5
        for _ in range(10):
            radii = radii * scale
            valid_now = True
            for i in range(n):
                if radii[i] > x[i] or radii[i] > (1 - x[i]) or \
                   radii[i] > y[i] or radii[i] > (1 - y[i]):
                    valid_now = False
                    break
                for j in range(i + 1, n):
                    if radii[i] + radii[j] > dist[i, j]:
                        valid_now = False
                        break
                if not valid_now:
                    break
            if valid_now:
                break
            scale *= 0.5
        is_valid = True
    
    sum_radii = np.sum(radii)
    return radii, sum_radii, is_valid


def evaluate_solution(params, n):
    """
    Evaluate a solution. params has shape (2*n,) with x coords then y coords.
    Returns (overlap_penalty, boundary_penalty, sum_radii)
    """
    x = params[:n]
    y = params[n:2*n]
    
    # Clip to [0,1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Compute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary penalty: how far circles would extend outside if we use a nominal radius
    # Use a reference radius for penalty calculation
    boundary_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Overlap penalty: for each pair, if distance is small, that's bad
    # We'll use the actual computed radii to determine overlap
    radii, sum_radii, is_valid = compute_radii_for_positions(x, y, n)
    
    # Overlap penalty: sum of violations
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if radii[i] + radii[j] > dist[i, j]:
                overlap_penalty += (radii[i] + radii[j] - dist[i, j])**2
    
    # Boundary penalty
    boundary_penalty = 0.0
    for i in range(n):
        if radii[i] > x[i]:
            boundary_penalty += (radii[i] - x[i])**2
        if radii[i] > 1 - x[i]:
            boundary_penalty += (radii[i] - (1 - x[i]))**2
        if radii[i] > y[i]:
            boundary_penalty += (radii[i] - y[i])**2
        if radii[i] > 1 - y[i]:
            boundary_penalty += (radii[i] - (1 - y[i]))**2
    
    return overlap_penalty, boundary_penalty, sum_radii


def fast_non_dominated_sort(fitnesses):
    """
    Fast non-dominated sorting. fitnesses is a list of tuples (f1, f2, f3).
    f1 = overlap_penalty (minimize), f2 = boundary_penalty (minimize), f3 = -sum_radii (minimize)
    Returns list of fronts, where each front is a list of indices.
    """
    n = len(fitnesses)
    S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
    n_dom = [0] * n  # number of solutions dominating i
    rank = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j
            if all(fitnesses[i][k] <= fitnesses[j][k] for k in range(3)) and \
               any(fitnesses[i][k] < fitnesses[j][k] for k in range(3)):
                S[i].append(j)
                n_dom[j] += 1
            elif all(fitnesses[j][k] <= fitnesses[i][k] for k in range(3)) and \
                 any(fitnesses[j][k] < fitnesses[i][k] for k in range(3)):
                S[j].append(i)
                n_dom[i] += 1
    
    # First front
    front = []
    for i in range(n):
        if n_dom[i] == 0:
            rank[i] = 0
            front.append(i)
    
    fronts = [front]
    current_front = front
    while current_front:
        next_front = []
        for i in current_front:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    rank[j] = rank[i] + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current_front = next_front
    
    return fronts


def crowding_distance(front_indices, fitnesses):
    """
    Compute crowding distance for a front.
    Returns dict: index -> distance
    """
    m = len(front_indices)
    if m <= 2:
        return {i: float('inf') for i in front_indices}
    
    distances = {i: 0.0 for i in front_indices}
    
    for k in range(3):  # 3 objectives
        # Sort by objective k
        sorted_front = sorted(front_indices, key=lambda i: fitnesses[i][k])
        
        # Assign infinite distance to boundary solutions
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # Normalize
        fmin = fitnesses[sorted_front[0]][k]
        fmax = fitnesses[sorted_front[-1]][k]
        f_range = fmax - fmin
        if f_range < 1e-12:
            continue
        
        for i in range(1, m - 1):
            distances[sorted_front[i]] += (fitnesses[sorted_front[i + 1]][k] - 
                                           fitnesses[sorted_front[i - 1]][k]) / f_range
    
    return distances


def tournament_selection(population, fitnesses, ranks, crowd_distances, k=2):
    """
    Tournament selection. population: list of param arrays.
    Returns index of selected individual.
    """
    n = len(population)
    candidates = random.sample(range(n), k)
    best = candidates[0]
    for c in candidates[1:]:
        # Lower rank is better
        if ranks[c] < ranks[best]:
            best = c
        elif ranks[c] == ranks[best]:
            # Higher crowding distance is better
            if crowd_distances[c] > crowd_distances[best]:
                best = c
    return best


def crossover(parent1, parent2, n):
    """
    Uniform crossover. Returns two children.
    """
    child1 = np.zeros(2 * n)
    child2 = np.zeros(2 * n)
    for i in range(2 * n):
        if random.random() < 0.5:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    return child1, child2


def mutation(params, n, sigma=0.05):
    """
    Gaussian mutation.
    """
    mutated = params.copy()
    for i in range(2 * n):
        if random.random() < 0.3:  # 30% chance to mutate each gene
            mutated[i] += random.gauss(0, sigma)
    # Clip to [0, 1]
    mutated = np.clip(mutated, 0, 1)
    return mutated


def nsga2_optimize(n, pop_size=100, n_generations=50, time_limit=100):
    """
    NSGA-II optimization for circle packing.
    """
    start_time = time.time()
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(n)))
            params = np.zeros(2 * n)
            idx = 0
            for row in range(side):
                for col in range(side):
                    if idx >= n:
                        break
                    params[idx] = (col + 0.5) / side  # x
                    params[idx + n] = (row + 0.5) / side  # y
                    idx += 1
            population.append(params)
        else:
            # Random initialization
            params = np.random.uniform(0.05, 0.95, 2 * n)
            population.append(params)
    
    # Evaluate initial population
    fitnesses = []
    for params in population:
        op, bp, sr = evaluate_solution(params, n)
        fitnesses.append((op, bp, -sr))
    
    best_sum_radii = max(-f[2] for f in fitnesses)
    best_params = None
    for i, f in enumerate(fitnesses):
        if -f[2] == best_sum_radii:
            best_params = population[i].copy()
    
    # Main loop
    for gen in range(n_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Fast non-dominated sorting
        fronts = fast_non_dominated_sort(fitnesses)
        
        # Compute ranks
        ranks = [0] * pop_size
        for front_idx, front in enumerate(fronts):
            for i in front:
                ranks[i] = front_idx
        
        # Compute crowding distance
        all_crowd = {i: 0.0 for i in range(pop_size)}
        for front in fronts:
            crowd = crowding_distance(front, fitnesses)
            for i in crowd:
                all_crowd[i] = crowd[i]
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individuals
        sorted_indices = sorted(range(pop_size), key=lambda i: (ranks[i], -all_crowd[i]))
        for i in sorted_indices[:pop_size // 2]:
            new_population.append(population[i].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
            
            # Selection
            idx1 = tournament_selection(population, fitnesses, ranks, all_crowd)
            idx2 = tournament_selection(population, fitnesses, ranks, all_crowd)
            
            # Crossover
            child1, child2 = crossover(population[idx1], population[idx2], n)
            
            # Mutation
            child1 = mutation(child1, n)
            child2 = mutation(child2, n)
            
            # Evaluate
            op1, bp1, sr1 = evaluate_solution(child1, n)
            op2, bp2, sr2 = evaluate_solution(child2, n)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
            
            # Update fitnesses for new individuals
            fitnesses.append((op1, bp1, -sr1))
            if len(new_population) < pop_size:
                fitnesses.append((op2, bp2, -sr2))
            
            # Track best
            if sr1 > best_sum_radii:
                best_sum_radii = sr1
                best_params = child1.copy()
            if sr2 > best_sum_radii and len(new_population) < pop_size:
                best_sum_radii = sr2
                best_params = child2.copy()
        
        # Truncate to population size
        new_population = new_population[:pop_size]
        fitnesses = fitnesses[:pop_size]
        
        # Update population
        population = new_population
    
    # Return best solution
    if best_params is None:
        # Fallback: use first solution
        best_params = population[0]
    
    x = best_params[:n]
    y = best_params[n:2*n]
    
    # Compute final radii
    radii, sum_radii, is_valid = compute_radii_for_positions(x, y, n)
    
    return np.column_stack([x, y, radii])


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses NSGA-II evolutionary algorithm to explore the non-convex landscape.
    """
    n = N
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run NSGA-II
    result = nsga2_optimize(n, pop_size=80, n_generations=40, time_limit=100)
    
    # Final validation and adjustment
    x = result[:, 0]
    y = result[:, 1]
    radii = result[:, 2]
    
    # Ensure validity
    for _ in range(10):
        valid = True
        for i in range(n):
            if radii[i] < 0:
                valid = False
                break
            if radii[i] > x[i] + 1e-9 or radii[i] > (1 - x[i]) + 1e-9 or \
               radii[i] > y[i] + 1e-9 or radii[i] > (1 - y[i]) + 1e-9:
                valid = False
                break
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx**2 + dy**2)
                if radii[i] + radii[j] > d + 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii *= 0.5
    
    return result
# EVOLVE-BLOCK-END
