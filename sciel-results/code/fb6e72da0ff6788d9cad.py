# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize(points):
        """Center and scale points so the max distance from origin is 1."""
        center = np.mean(points, axis=0)
        points = points - center
        max_norm = np.max(np.linalg.norm(points, axis=1))
        if max_norm > 1e-10:
            points = points / max_norm
        return points

    def local_optimize(points, max_iter=2000):
        """
        Local optimization using a combination of coordinate descent and small perturbations.
        This is more robust than pure gradient ascent for this non-smooth objective.
        """
        best_points = normalize(points.copy())
        best_score = compute_score(best_points)
        current_points = best_points.copy()
        current_score = best_score
        
        # Adaptive step size
        step_size = 0.05
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try moving each point in small random directions
            for _ in range(10):  # 10 random perturbations per iteration
                idx = np.random.randint(n)
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                
                new_points = current_points.copy()
                new_points[idx] += step_size * direction
                new_points = normalize(new_points)
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                else:
                    # Accept with some probability (simulated annealing style)
                    if np.random.random() < 0.1:  # 10% acceptance rate for worse moves
                        current_points = new_points
                        current_score = new_score
            
            if not improved:
                no_improve_count += 1
                # Reduce step size if no improvement
                if no_improve_count > 50:
                    step_size *= 0.95
                    no_improve_count = 0
                    # Reset to best found so far to escape local minima
                    if np.random.random() < 0.1:
                        current_points = best_points.copy()
                        current_score = best_score
            
            # Occasionally increase step size to escape local optima
            if it % 100 == 0 and it > 0:
                step_size = min(step_size * 1.2, 0.1)
        
        return best_points, best_score

    def differential_evolution_simple(bounds, max_iter=100, pop_size=50, F=0.8, CR=0.9):
        """
        Simple implementation of differential evolution for global optimization.
        """
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, n, 2))
        population = np.array([normalize(p) for p in population])
        fitness = np.array([compute_score(p) for p in population])
        
        best_idx = np.argmax(fitness)
        best_points = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for gen in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            for i in range(pop_size):
                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                cross_idx = np.random.randint(n * 2)
                for j in range(n * 2):
                    if np.random.random() < CR or j == cross_idx:
                        trial[j // 2, j % 2] = mutant[j // 2, j % 2]
                
                trial = normalize(trial)
                trial_fitness = compute_score(trial)
                
                # Selection
                if trial_fitness >= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_points = trial.copy()
            
            # Elitism: keep the best individual
            pop_best_idx = np.argmax(fitness)
            if fitness[pop_best_idx] > best_fitness:
                best_fitness = fitness[pop_best_idx]
                best_points = population[pop_best_idx].copy()
        
        return best_points, best_fitness

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(config1)
    
    # 2. Two concentric circles (8 points each)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.6 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config2 = np.vstack([inner_pts, outer_pts])
    initial_configs.append(config2)
    
    # 3. Three concentric circles (5, 5, 6 points)
    c1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    c2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    c3_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    c1_pts = 0.4 * np.column_stack([np.cos(c1_angles), np.sin(c1_angles)])
    c2_pts = 0.7 * np.column_stack([np.cos(c2_angles), np.sin(c2_angles)])
    c3_pts = np.column_stack([np.cos(c3_angles), np.sin(c3_angles)])
    config3 = np.vstack([c1_pts, c2_pts, c3_pts])
    initial_configs.append(config3)
    
    # 4. Square grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config4 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    initial_configs.append(config4)
    
    # 5. Hexagonal lattice subset
    # Center + 6 neighbors + 9 more in second ring
    center = np.array([[0.0, 0.0]])
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_r = 1.0
    ring1 = hex_r * np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    # Second ring: 12 points, take 9
    hex_angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring2 = 1.732 * np.column_stack([np.cos(hex_angles2), np.sin(hex_angles2)])
    config5 = np.vstack([center, ring1, ring2[:9]])
    initial_configs.append(config5)
    
    # 6. Random configurations
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed)
        rand_pts = np.random.randn(n, 2)
        rand_pts = normalize(rand_pts)
        initial_configs.append(rand_pts)
    
    # Optimize each configuration
    best_overall = None
    best_overall_score = -np.inf
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Local optimization
        opt_points, opt_score = local_optimize(config, max_iter=1000)
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall = opt_points.copy()
        
        # If we have time, try differential evolution from a few starting points
        if i < 3 and time.time() < deadline - 60:
            de_points, de_score = differential_evolution_simple(
                bounds=[-2, 2], 
                max_iter=50, 
                pop_size=30
            )
            if de_score > best_overall_score:
                best_overall_score = de_score
                best_overall = de_points.copy()
    
    # Final local optimization from the best found
    if best_overall is not None and time.time() < deadline - 20:
        final_points, final_score = local_optimize(best_overall, max_iter=2000)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_points.copy()
    
    # Final normalization
    if best_overall is not None:
        best_overall = normalize(best_overall)
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END
