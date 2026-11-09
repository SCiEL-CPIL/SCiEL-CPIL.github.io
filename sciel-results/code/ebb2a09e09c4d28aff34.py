# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Generate all pair sums and their probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w) # (n, n)
    
    # Flatten
    sums_flat = pair_sums.flatten() # (n^2,)
    probs_flat = pair_probs.flatten() # (n^2,)
    
    # Sort sums and compute cumulative probabilities
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Ensure cum_probs ends at 1.0 (floating point safety)
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
             cum_probs[-1] = 1.0
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    thresholds = 2.0 * s_row - s_col  # (n, n)
    
    # For each threshold, find the count of pair_sums < threshold
    # searchsorted with side='left' gives the index of the first element >= threshold.
    # We want sum of probs for elements < threshold.
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _normalize(s, w):
    """
    Normalize support and weights to ensure validity.
    s >= 0, w > 0, sum(w) = 1.
    """
    if len(s) == 0 or len(w) == 0:
        return None, None
    
    # Clip support to non-negative
    s = np.maximum(s, 0.0)
    
    # Clip weights to positive (small epsilon)
    w = np.maximum(w, 1e-9)
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return None, None
    w = w / w_sum
    
    # Sort by support to maintain consistency
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    return s, w

def _ga_mutation(individual, sigma_s=0.1, sigma_w=0.1, prob_add=0.1, prob_remove=0.1):
    """
    Mutate an individual (support, weights).
    """
    s, w = individual
    n = len(s)
    
    # 1. Standard Gaussian noise
    s_new = s + np.random.normal(0, sigma_s, n)
    w_new = w + np.random.normal(0, sigma_w, n)
    
    # 2. Structural mutation: Add or Remove atom
    r = np.random.random()
    
    if r < prob_add and n < 20:
        # Add a new atom
        # Sample new support from existing range or slightly outside
        if n > 0:
            s_min, s_max = np.min(s), np.max(s)
            # 50% chance to extend range, 50% to fill gap
            if np.random.random() < 0.5:
                s_add = np.random.uniform(s_min - 0.5, s_max + 0.5)
            else:
                s_add = np.random.uniform(s_min, s_max)
        else:
            s_add = np.random.uniform(0, 1)
        
        # New weight: small initial weight
        w_add = np.random.uniform(0.01, 0.1)
        
        s_new = np.append(s_new, s_add)
        w_new = np.append(w_new, w_add)
        
    elif r < prob_add + prob_remove and n > 2:
        # Remove a random atom (keep at least 2)
        idx_to_remove = np.random.choice(n)
        s_new = np.delete(s_new, idx_to_remove)
        w_new = np.delete(w_new, idx_to_remove)
        
    # 3. Occasional "Jump" mutation for one support point
    if np.random.random() < 0.1:
        idx = np.random.choice(len(s_new))
        s_new[idx] = np.random.uniform(0, 2.0) # Jump to random location
        
    return _normalize(s_new, w_new)

def _ga_crossover(ind1, ind2):
    """
    Crossover between two individuals of potentially different lengths.
    Uses "unified" crossover: sample points from both parents.
    """
    s1, w1 = ind1
    s2, w2 = ind2
    
    # Combine all points
    s_all = np.concatenate([s1, s2])
    w_all = np.concatenate([w1, w2])
    
    # Randomly select a subset to form child
    # Target size: average of parents, with some variation
    target_n = int(np.round((len(s1) + len(s2)) / 2.0 + np.random.normal(0, 1)))
    target_n = max(2, min(target_n, 20)) # Bounds
    
    if len(s_all) < target_n:
        target_n = len(s_all)
        
    # Sample indices without replacement
    idx = np.random.choice(len(s_all), target_n, replace=False)
    
    s_child = s_all[idx]
    w_child = w_all[idx]
    
    return _normalize(s_child, w_child)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using a Genetic Algorithm with adaptive population size.
    """
    t_start = time.time()
    deadline = t_start + 580.0 # Safety margin
    
    # Initialize population
    # Strategy: Start with a mix of "heavy-tail" candidates (geometric on integers) and random ones.
    # Inspiration: Geometric weights on integer support creates heavy tail near 0.
    
    population = []
    
    # Candidate 1: Geometric on 0..5
    s_geo = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    w_geo = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125])
    s_norm, w_norm = _normalize(s_geo, w_geo)
    if s_norm is not None:
        population.append((s_norm, w_norm))
        
    # Candidate 2: Geometric on 0..10 (finer)
    s_geo2 = np.linspace(0, 10, 11)
    w_geo2 = 0.5 ** np.arange(11)
    s_norm2, w_norm2 = _normalize(s_geo2, w_geo2)
    if s_norm2 is not None:
        population.append((s_norm2, w_norm2))
        
    # Candidate 3: Standard [0, 1] with optimized weights from seed
    s_std = np.array([0.0, 1.0])
    w_std = np.array([0.618, 0.382])
    s_norm3, w_norm3 = _normalize(s_std, w_std)
    if s_norm3 is not None:
        population.append((s_norm3, w_norm3))
        
    # Candidate 4: Random heavy tail
    s_rand = np.sort(np.random.exponential(1.0, 10))
    w_rand = np.random.uniform(0.1, 1.0, 10)
    s_norm4, w_norm4 = _normalize(s_rand, w_rand)
    if s_norm4 is not None:
        population.append((s_norm4, w_norm4))
        
    # Fill rest of population with random variations
    pop_size = 40
    while len(population) < pop_size:
        # Random init: n in [3, 10]
        n = np.random.randint(3, 11)
        s = np.sort(np.random.uniform(0, 2.0, n))
        # Heavy tail bias in weights
        w = np.random.exponential(1.0, n)
        s_n, w_n = _normalize(s, w)
        if s_n is not None:
            population.append((s_n, w_n))
            
    # Evaluate initial population
    scores = []
    for ind in population:
        s, w = ind
        p = _compute_P_fast(s, w)
        scores.append(p)
        
    best_ind = population[np.argmax(scores)]
    best_score = max(scores)
    
    # Evolutionary Loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # 1. Selection: Tournament selection
        # We will generate offspring for the next generation
        
        new_population = []
        
        # Elitism: Keep top 5
        top_indices = np.argsort(scores)[::-1][:5]
        for i in top_indices:
            new_population.append(population[i])
            
        # Generate rest via Crossover and Mutation
        while len(new_population) < pop_size:
            # Tournament select 2 parents
            idx1, idx2 = np.random.randint(0, len(population), 2)
            p1 = population[idx1]
            p2 = population[idx2]
            
            # Crossover
            child = _ga_crossover(p1, p2)
            
            if child is None:
                # Fallback to mutation of one parent
                child = _ga_mutation(p1)
                
            if child is None:
                continue
                
            # Mutation
            child = _ga_mutation(child, sigma_s=0.05, sigma_w=0.05)
            
            if child is not None:
                new_population.append(child)
                
        # Evaluate new population
        new_scores = []
        for ind in new_population:
            s, w = ind
            p = _compute_P_fast(s, w)
            new_scores.append(p)
            
        # Update best
        max_idx = np.argmax(new_scores)
        if new_scores[max_idx] > best_score:
            best_score = new_scores[max_idx]
            best_ind = new_population[max_idx]
            
        # Replace population
        population = new_population
        scores = new_scores
        
        # Adaptive mutation rate: if stagnating, increase mutation
        # Check stagnation over last 5 generations (simplified: just check current best vs previous)
        # For now, keep rates constant but slightly increase as time passes to explore more
        elapsed = time.time() - t_start
        progress = elapsed / (deadline - t_start)
        # Increase sigma slightly
        # (Implemented in mutation call above with fixed values, could make dynamic)
        
    # Final Validation
    if best_ind is None:
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    s_best, w_best = best_ind
    
    # Ensure valid output
    s_best = np.maximum(s_best, 0.0)
    w_best = np.maximum(w_best, 0.0)
    
    w_sum = np.sum(w_best)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    w_best = w_best / w_sum
    
    if not (np.all(np.isfinite(s_best)) and np.all(np.isfinite(w_best))):
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    return s_best, w_best

# EVOLVE-BLOCK-END
