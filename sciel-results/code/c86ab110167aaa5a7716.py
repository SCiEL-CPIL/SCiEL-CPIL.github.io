"""Rewritten program for the Third Autocorrelation Inequality task."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_c3(signed):
    """
    Compute the upper bound C3 for a signed step function.
    The task mentions 'autoconvolution upper bound' and 'absolute value differently'.
    Standard C3 inequality context often involves:
    C3 = max_x | (f * f)(x) | / ||f||_2^2 or similar.
    However, the prompt says "score is the upper bound on C3" and "lower is better".
    Usually, we want to minimize the maximum absolute value of the autocorrelation
    (or autoconvolution) relative to the energy, or just the max absolute value
    if the norm is fixed.
    
    Given the ambiguity, a common metric in these optimization problems is:
    Objective = max_{x} | (f * f)(x) |
    where f is the step function.
    Since the intervals are equal, the convolution is a discrete convolution.
    Let h be the heights.
    The convolution c_k = sum_{i} h_i h_{i+k} (circular or linear?).
    "Signed step function with 400 equal intervals on [-1/4, 1/4]".
    This is a finite domain. Linear convolution is likely.
    
    Let's assume the metric is the maximum absolute value of the linear autocorrelation.
    If we normalize by the sum of squares (which is constant for +/-1), minimizing max|autocorr|
    is equivalent to minimizing the raw max|autocorr|.
    
    Let's try:
    c = np.correlate(signed, signed, mode='full')
    # Remove the trivial center peak? No, usually the whole thing matters.
    # But the DC component (k=0) is always sum(h^2) = N.
    # Often the "bound" refers to the sidelobes or the max over all lags excluding 0?
    # Or maybe the max over all lags?
    # If k=0 is included, the value is at least N.
    # Let's look at the "seed" value. The seed returns np.ones(N).
    # For ones, autocorr at k=0 is N. At k!=0, it is N-|k|.
    # Max is N.
    # If the score is "upper bound on C3", and lower is better.
    # If we return ones, score = N = 400.
    # If we can reduce the max autocorrelation, we improve.
    # Random signs will have autocorrelation ~ sqrt(N) at k=0? No, k=0 is always N.
    # So if k=0 is included, the minimum possible is N.
    # This suggests the metric likely EXCLUDES k=0 or normalizes differently.
    # Common in "third autocorrelation inequality" (e.g., related to C3 in harmonic analysis or
    # specific combinatorial designs) might involve the sum of absolute autocorrelations
    # or the max sidelobe.
    
    # Let's assume the metric is the maximum absolute autocorrelation for lags k != 0.
    # This is a common "sidelobe" minimization problem.
    """
    c = np.correlate(signed, signed, mode='full')
    # c has length 2*N - 1. The center is at index N-1.
    # We want to minimize max(|c|) excluding the center?
    # Let's try excluding the center first.
    c_sidelobes = np.concatenate([c[:N-1], c[N:]])
    return np.max(np.abs(c_sidelobes))

def _compute_c3_full(signed):
    """Alternative: max over all lags."""
    c = np.correlate(signed, signed, mode='full')
    return np.max(np.abs(c))

def _compute_c3_sum(signed):
    """Alternative: sum of absolute sidelobes."""
    c = np.correlate(signed, signed, mode='full')
    c_sidelobes = np.concatenate([c[:N-1], c[N:]])
    return np.sum(np.abs(c_sidelobes))

# The prompt says "Paper and published result notebook place the absolute value differently; 
# execution must explicitly select a metric."
# And "score is the upper bound on C3".
# In many contexts, C3 is bounded by something like 2 * max_sidelobe or similar.
# Let's stick to the most standard "minimize max sidelobe" interpretation for now, 
# as it allows for significant improvement over the trivial 'ones' case.
# For 'ones', max sidelobe is N-1 = 399.
# For random signs, max sidelobe is approx sqrt(N * log N) ~ 10-20.
# This is a huge improvement.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization
    # Use random signs as a good starting point
    pop_size = 50
    population = []
    
    # Generate diverse initial population
    for _ in range(pop_size):
        # Mix of random and structured
        if np.random.rand() < 0.5:
            s = np.random.choice([-1, 1], size=N)
        else:
            # Block structure
            block_size = np.random.randint(2, 20)
            s = np.random.choice([-1, 1], size=N//block_size + 1)
            s = np.repeat(s, block_size)[:N]
        population.append(s)
    
    # Evaluate initial population
    scores = []
    for s in population:
        scores.append(_compute_c3(s))
    scores = np.array(scores)
    
    best_idx = np.argmin(scores)
    best_solution = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # 2. Genetic Algorithm / Evolutionary Search
    # Parameters
    tournament_size = 5
    mutation_rate = 0.05  # Probability of flipping a single bit
    crossover_rate = 0.7
    elite_count = 5
    
    generation = 0
    while time.time() < deadline:
        # Selection: Tournament Selection
        new_population = []
        
        # Elitism: Keep the best solutions
        elite_indices = np.argsort(scores)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Fill the rest
        while len(new_population) < pop_size:
            # Select parents
            parent1_idx = _tournament_select(scores, population, tournament_size)
            parent2_idx = _tournament_select(scores, population, tournament_size)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Uniform crossover
                mask = np.random.rand(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            # Flip bits with probability mutation_rate
            flip_mask = np.random.rand(N) < mutation_rate
            child[flip_mask] *= -1
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population
        for i in range(len(population)):
            if i < elite_count:
                # Reuse old scores for elites? No, we copied them, so scores are valid.
                # But to be safe and allow for slight variations in evaluation, we keep the score.
                pass
            else:
                scores[i] = _compute_c3(population[i])
        
        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        generation += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # 3. Local Search Polish
    # Use the remaining time to polish the best solution
    if time.time() < deadline - 2.0:
        best_solution = _local_search_polish(best_solution, time.time(), deadline - 1.0)
        best_score = _compute_c3(best_solution)
    
    return best_solution

def _tournament_select(scores, population, k):
    indices = np.random.randint(0, len(population), size=k)
    best_idx = indices[0]
    best_score = scores[best_idx]
    for i in indices[1:]:
        if scores[i] < best_score:
            best_score = scores[i]
            best_idx = i
    return best_idx

def _local_search_polish(solution, start_time, deadline):
    """
    Perform multi-flip local search to escape shallow minima.
    """
    current_solution = solution.copy()
    current_score = _compute_c3(current_solution)
    
    # Pre-compute convolution for efficiency? 
    # np.correlate is fast enough for N=400.
    
    while time.time() < deadline:
        # Try single flip
        improved = False
        for i in range(N):
            # Flip bit i
            current_solution[i] *= -1
            new_score = _compute_c3(current_solution)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                current_solution[i] *= -1
        
        if not improved:
            # Try 2-flip moves
            improved_2 = False
            # Sample random pairs to avoid O(N^2)
            num_pairs = 1000
            for _ in range(num_pairs):
                i, j = np.random.choice(N, size=2, replace=False)
                current_solution[i] *= -1
                current_solution[j] *= -1
                new_score = _compute_c3(current_solution)
                if new_score < current_score:
                    current_score = new_score
                    improved_2 = True
                else:
                    current_solution[i] *= -1
                    current_solution[j] *= -1
            
            if not improved_2:
                # Try 3-flip moves
                improved_3 = False
                num_triples = 500
                for _ in range(num_triples):
                    i, j, k = np.random.choice(N, size=3, replace=False)
                    current_solution[i] *= -1
                    current_solution[j] *= -1
                    current_solution[k] *= -1
                    new_score = _compute_c3(current_solution)
                    if new_score < current_score:
                        current_score = new_score
                        improved_3 = True
                    else:
                        current_solution[i] *= -1
                        current_solution[j] *= -1
                        current_solution[k] *= -1
                
                if not improved_3:
                    break
                    
        # Check time inside loop
        if time.time() > deadline:
            break

    return current_solution

# EVOLVE-BLOCK-END
