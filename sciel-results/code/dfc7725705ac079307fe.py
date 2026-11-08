import numpy as np
import time
import math

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    Uses rfft for speed.
    """
    M = 4 * N
    dx = 0.5 / M
    
    f_fine = np.repeat(signed_step, M // N)
    
    # Use rfft for real input, zero-pad to 2*M for linear convolution
    fft_f = np.fft.rfft(f_fine, n=2*M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2*M).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Population-based search (Genetic Algorithm with Local Search)
    pop_size = 50
    
    # Initialize population
    population = []
    scores = []
    
    # Include the all-ones baseline
    baseline = np.ones(N, dtype=np.int8)
    population.append(baseline)
    scores.append(_compute_c3(baseline))
    
    # Fill with random individuals
    while len(population) < pop_size:
        ind = np.random.choice([-1, 1], size=N).astype(np.int8)
        population.append(ind)
        scores.append(_compute_c3(ind))
    
    # Sort by score (lower is better)
    indices = np.argsort(scores)
    population = [population[i] for i in indices]
    scores = [scores[i] for i in indices]
    
    best_step = population[0].copy()
    best_score = scores[0]
    
    # GA Loop
    while time.time() - start_time < DEADLINE - 2.0:
        new_population = []
        
        # Elitism: keep top 20%
        elite_count = max(1, pop_size // 5)
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Check time
            if time.time() - start_time > DEADLINE - 2.0:
                break
                
            # Selection: Tournament selection
            a = np.random.randint(0, pop_size)
            b = np.random.randint(0, pop_size)
            if scores[a] < scores[b]:
                parent1 = population[a]
            else:
                parent1 = population[b]
                
            c = np.random.randint(0, pop_size)
            d = np.random.randint(0, pop_size)
            if scores[c] < scores[d]:
                parent2 = population[c]
            else:
                parent2 = population[d]
            
            # Crossover: Uniform
            mask = np.random.random(N) < 0.5
            child = np.where(mask, parent1, parent2).astype(np.int8)
            
            # Mutation: Flip a few bits
            num_mutations = np.random.randint(1, 15)
            mut_indices = np.random.choice(N, size=num_mutations, replace=False)
            child[mut_indices] *= -1
            
            new_population.append(child)
        
        # If we broke early, pad with best existing
        while len(new_population) < pop_size:
            new_population.append(best_step.copy())
        
        # Evaluate new population
        new_scores = []
        for ind in new_population:
            if time.time() - start_time > DEADLINE - 2.0:
                break
            s = _compute_c3(ind)
            new_scores.append(s)
        
        # Pad scores if evaluation broke early
        while len(new_scores) < pop_size:
            new_scores.append(best_score)
            
        # Combine and sort
        all_population = new_population
        all_scores = new_scores
        
        indices = np.argsort(all_scores)
        population = [all_population[i] for i in indices[:pop_size]]
        scores = [all_scores[i] for i in indices[:pop_size]]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_step = population[0].copy()
            
        # Local search (Memetic step) on the best individual
        if time.time() - start_time < DEADLINE - 5.0:
            current = best_step.copy()
            current_score = best_score
            T = 0.001
            # Run a short SA chain
            for _ in range(100):
                if time.time() - start_time > DEADLINE - 2.0:
                    break
                idx = np.random.randint(N)
                current[idx] *= -1
                new_s = _compute_c3(current)
                delta = new_s - current_score
                if delta < 0 or np.random.random() < math.exp(-delta / T):
                    current_score = new_s
                    if current_score < best_score:
                        best_score = current_score
                        best_step = current.copy()
                else:
                    current[idx] *= -1
                T *= 0.95
                
    return best_step
