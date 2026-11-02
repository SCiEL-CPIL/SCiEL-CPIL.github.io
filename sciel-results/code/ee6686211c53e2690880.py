"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

def _compute_overlaps(h):
    """
    Compute overlap for all shifts k=0..N-1.
    overlap_k = sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    where dw = 2/N.
    """
    dw = 2.0 / N
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        valid = N - k
        overlaps[k] = np.sum(h[:valid] * (1.0 - h[k:])) * dw
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _soft_min_surrogate(overlaps, beta):
    """
    Log-sum-exp based soft minimum of (target - overlaps) to approximate
    minimizing the maximum overlap. We want to minimize max(overlaps),
    which is equivalent to maximizing min(target - overlaps) for some target.
    Actually, let's use: soft_max(overlaps) = (1/beta) * log(sum(exp(beta * overlaps)))
    This is a smooth upper bound on max(overlaps).
    """
    scaled = beta * overlaps
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        # numerical stability
        return (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
    return (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta

def _objective_smooth(h, beta):
    """Smooth objective: soft-max of overlaps."""
    overlaps = _compute_overlaps(h)
    return _soft_min_surrogate(overlaps, beta)

def _project_to_sum(h, target_sum):
    """Project h to have exact sum = target_sum while staying in [0,1]."""
    h = h.copy()
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    h += diff / len(h)
    h = np.clip(h, 0.0, 1.0)
    # Re-correct
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) > 1e-10:
        diff = target_sum - current_sum
        # Distribute to elements not at boundary
        mask = (h > 1e-6) & (h < 1.0 - 1e-6)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Fallback: add to element closest to 0.5
            idx = np.argmin(np.abs(h - 0.5))
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0.0, 1.0)
    return h

def _random_initial(n, seed=None):
    """Generate a random valid h with sum = n/2."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Start with random values, then project
    h = rng.uniform(0.0, 1.0, size=n)
    h = _project_to_sum(h, n / 2.0)
    return h

def _genetic_algorithm(h_init, pop_size=40, n_generations=200, mutation_rate=0.3,
                       crossover_rate=0.7, elite_size=5, deadline=None,
                       start_time=None):
    """
    Genetic algorithm to minimize max overlap.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(h_init)
    target_sum = n / 2.0
    
    # Initialize population
    population = [h_init.copy()]
    for i in range(pop_size - 1):
        population.append(_random_initial(n, seed=42 + i))
    
    # Evaluate initial population
    scores = [_max_overlap(h) for h in population]
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(n_generations):
        if deadline is not None:
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
        
        # Sort by score
        order = np.argsort(scores)
        sorted_pop = [population[i] for i in order]
        sorted_scores = [scores[i] for i in order]
        
        # Update best
        if sorted_scores[0] < best_score:
            best_score = sorted_scores[0]
            best_h = sorted_pop[0].copy()
        
        # Elitism
        new_population = [sorted_pop[i].copy() for i in range(min(elite_size, pop_size))]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection: tournament
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] <= scores[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
            
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] <= scores[idx2]:
                parent2 = population[idx1]
            else:
                parent2 = population[idx2]
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.random(n) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.1, size=n)
                child += noise
                child = _project_to_sum(child, target_sum)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = [_max_overlap(h) for h in population]
        
        # Periodic re-injection of random individuals to maintain diversity
        if gen % 20 == 19:
            for i in range(3):
                idx = np.random.randint(0, pop_size)
                population[idx] = _random_initial(n, seed=gen * 100 + i)
                scores[idx] = _max_overlap(population[idx])
    
    return best_h, best_score

def _local_refinement(h, max_iterations=500, step_size=0.01, deadline=None, start_time=None):
    """
    Local refinement using coordinate descent with adaptive step size.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    current_score = _max_overlap(h)
    
    for it in range(max_iterations):
        if deadline is not None:
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
        
        improved = False
        for i in range(n):
            # Try small perturbations
            for delta in [step_size, -step_size, step_size * 0.5, -step_size * 0.5,
                          step_size * 0.1, -step_size * 0.1]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0.0 or h_trial[i] > 1.0:
                    continue
                # Re-project to maintain sum
                h_trial = _project_to_sum(h_trial, target_sum)
                score = _max_overlap(h_trial)
                if score < current_score - 1e-12:
                    h = h_trial
                    current_score = score
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Phase 1: Multi-restart smooth optimization
    best_h = None
    best_score = np.inf
    
    # Start with anti-periodic structure as a good initial guess
    half = N // 2
    h_anti = np.zeros(N)
    h_anti[:half] = 0.7
    h_anti[half:] = 0.3
    h_anti = _project_to_sum(h_anti, N / 2.0)
    
    # Also try a few other structured initializations
    initializations = [h_anti]
    for seed in [0, 1, 2, 3, 4, 5]:
        initializations.append(_random_initial(N, seed=seed))
    
    for h_init in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Smooth optimization with increasing beta
        h_current = h_init.copy()
        
        for beta in [10, 30, 100, 300]:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5.0:
                break
            
            # Use gradient-free optimization via Nelder-Mead on a subset of coordinates
            # For speed, we do a simple hill-climbing on the smooth objective
            h_curr = h_current.copy()
            curr_obj = _objective_smooth(h_curr, beta)
            
            step = 0.05
            for it in range(200):
                elapsed = time.time() - start_time
                if elapsed > deadline - 5.0:
                    break
                
                improved = False
                # Try random coordinate updates
                for _ in range(10):
                    i = np.random.randint(0, N)
                    delta = np.random.uniform(-step, step)
                    h_trial = h_curr.copy()
                    h_trial[i] += delta
                    if h_trial[i] < 0.0 or h_trial[i] > 1.0:
                        continue
                    h_trial = _project_to_sum(h_trial, N / 2.0)
                    obj_trial = _objective_smooth(h_trial, beta)
                    if obj_trial < curr_obj - 1e-12:
                        h_curr = h_trial
                        curr_obj = obj_trial
                        improved = True
                        break
                
                if not improved:
                    step *= 0.8
                    if step < 1e-5:
                        break
            
            h_current = h_curr
        
        # Evaluate with true max overlap
        score = _max_overlap(h_current)
        if score < best_score:
            best_score = score
            best_h = h_current.copy()
    
    # Phase 2: Genetic algorithm
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0 and best_h is not None:
        ga_h, ga_score = _genetic_algorithm(
            best_h, pop_size=30, n_generations=100,
            mutation_rate=0.4, crossover_rate=0.6, elite_size=4,
            deadline=deadline - 5.0, start_time=start_time
        )
        if ga_score < best_score:
            best_score = ga_score
            best_h = ga_h.copy()
    
    # Phase 3: Local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0 and best_h is not None:
        refined_h, refined_score = _local_refinement(
            best_h, max_iterations=300, step_size=0.01,
            deadline=deadline - 2.0, start_time=start_time
        )
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h.copy()
    
    # Final: ensure valid output
    h_final = _project_to_sum(best_h, N / 2.0)
    
    # Verify sum is exactly N/2
    current_sum = np.sum(h_final)
    if abs(current_sum - N / 2.0) > 1e-12:
        diff = N / 2.0 - current_sum
        h_final = h_final + diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
        current_sum = np.sum(h_final)
        if abs(current_sum - N / 2.0) > 1e-12:
            diff = N / 2.0 - current_sum
            idx = np.argmin(np.abs(h_final - 0.5))
            h_final[idx] += diff
            h_final[idx] = np.clip(h_final[idx], 0.0, 1.0)
    
    return h_final
