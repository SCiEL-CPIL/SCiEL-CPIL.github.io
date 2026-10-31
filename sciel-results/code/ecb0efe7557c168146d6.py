import numpy as np

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def construct():
    # Parameters for the Genetic Algorithm
    pop_size = 100
    generations = 40
    mutation_rate = 0.05
    crossover_rate = 0.8
    elite_size = 5
    
    # Initialize population
    # Start with a mix of random, uniform, and peaked distributions
    population = np.zeros((pop_size, N))
    
    # 1. Uniform
    population[0] = np.ones(N)
    
    # 2. Random positive
    for i in range(1, pop_size // 2):
        population[i] = np.random.rand(N)
        
    # 3. Gaussian centered
    for i in range(pop_size // 2, 3 * pop_size // 4):
        x = np.linspace(-1, 1, N)
        sigma = 0.2 + 0.5 * np.random.rand()
        population[i] = np.exp(-x**2 / (2 * sigma**2))
        
    # 4. Triangular / Peaked
    for i in range(3 * pop_size // 4, pop_size):
        x = np.linspace(-1, 1, N)
        width = 0.3 + 0.7 * np.random.rand()
        population[i] = np.maximum(0, 1 - np.abs(x) / width)

    # Ensure non-negative
    population = np.maximum(population, 0)
    
    # Evaluate initial population
    scores = np.array([_score(ind) for ind in population])
    
    best_score = scores.min()
    best_ind = population[scores.argmin()]
    
    # Main GA Loop
    for gen in range(generations):
        # Sort population by score (ascending, lower is better)
        sorted_indices = np.argsort(scores)
        population = population[sorted_indices]
        scores = scores[sorted_indices]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_ind = population[0].copy()
            
        # Elitism: Keep the best individuals
        new_population = np.zeros((pop_size, N))
        new_population[:elite_size] = population[:elite_size]
        
        # Generate offspring
        for i in range(elite_size, pop_size):
            # Selection: Tournament selection
            # Pick 3 random candidates, select the best
            candidates = np.random.choice(pop_size, 3, replace=False)
            candidate_scores = scores[candidates]
            winner_idx = candidates[np.argmin(candidate_scores)]
            parent1 = population[winner_idx]
            
            # Selection for second parent
            candidates2 = np.random.choice(pop_size, 3, replace=False)
            candidate_scores2 = scores[candidates2]
            winner_idx2 = candidates2[np.argmin(candidate_scores2)]
            parent2 = population[winner_idx2]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Arithmetic crossover with random alpha
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()
                
            # Mutation
            if np.random.rand() < mutation_rate:
                # Gaussian mutation
                noise = np.random.normal(0, 0.1, N)
                child = child + noise
            else:
                # Small uniform mutation
                mask = np.random.rand(N) < 0.01
                child[mask] += np.random.uniform(-0.1, 0.1, mask.sum())
                
            # Enforce non-negativity
            child = np.maximum(child, 0)
            
            new_population[i] = child
            
        population = new_population
        scores = np.array([_score(ind) for ind in population])

    # Final evaluation
    final_score = _score(best_ind)
    
    # Return the best individual found
    return best_ind.tolist()
