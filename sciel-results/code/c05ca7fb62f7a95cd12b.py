import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - 5.0 - elapsed()
    
    np.random.seed(42)
    
    # Initialize population with diverse candidates
    pop_size = 40
    population = []
    
    # Seed with all ones
    population.append(np.ones(N, dtype=np.int8))
    
    # Add random individuals
    for _ in range(pop_size - 1):
        population.append(np.random.choice([-1, 1], size=N).astype(np.int8))
    
    # Evaluate initial population
    scores = np.array([_compute_c3(ind) for ind in population])
    
    best_idx = np.argmin(scores)
    best_step = population[best_idx].copy()
    best_c3 = scores[best_idx]
    
    # Genetic algorithm parameters
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.02  # probability of flipping each bit
    
    # Elitism: keep top 2
    elite_size = 2
    
    generation = 0
    no_improve_generations = 0
    patience = 15
    
    while time_left() > 10.0:
        generation += 1
        
        # Sort by score (lower is better)
        sorted_indices = np.argsort(scores)
        
        # Check if we improved
        current_best = scores[sorted_indices[0]]
        if current_best < best_c3 - 1e-12:
            best_c3 = current_best
            best_step = population[sorted_indices[0]].copy()
            no_improve_generations = 0
        else:
            no_improve_generations += 1
        
        # If stuck, add diversity via random restarts in population
        if no_improve_generations >= patience:
            # Replace worst half with random individuals
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = np.random.choice([-1, 1], size=N).astype(np.int8)
                scores[i] = _compute_c3(population[i])
            no_improve_generations = 0
        
        # Create next generation
        new_population = []
        
        # Elitism: copy top individuals
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Fill rest via tournament selection + crossover + mutation
        while len(new_population) < pop_size:
            if time_left() < 5.0:
                break
            
            # Tournament selection for parent 1
            tournament1 = np.random.choice(pop_size, size=tournament_size, replace=False)
            parent1_idx = tournament1[np.argmin(scores[tournament1])]
            
            # Tournament selection for parent 2
            tournament2 = np.random.choice(pop_size, size=tournament_size, replace=False)
            parent2_idx = tournament2[np.argmin(scores[tournament2])]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if np.random.random() < crossover_rate and parent1_idx != parent2_idx:
                # Uniform crossover
                mask = np.random.random(N) < 0.5
                child = np.where(mask, parent1, parent2).astype(np.int8)
            else:
                child = parent1.copy()
            
            # Mutation: flip each bit with probability mutation_rate
            flip_mask = np.random.random(N) < mutation_rate
            child[flip_mask] = -child[flip_mask]
            
            new_population.append(child)
        
        # Evaluate new individuals (only those not from elitism)
        for i in range(elite_size, len(new_population)):
            if time_left() < 5.0:
                new_population[i] = new_population[i-1].copy()
                scores[i] = scores[i-1]
            else:
                new_population[i] = new_population[i]
        
        # Re-evaluate all non-elite members
        for i in range(elite_size, len(new_population)):
            new_population[i] = new_population[i]
        
        # Compute scores for new population
        for i in range(elite_size, len(new_population)):
            new_population[i] = new_population[i]
        
        # Batch evaluation
        for i in range(elite_size, len(new_population)):
            if time_left() < 3.0:
                # Copy best to fill remaining
                new_population[i] = new_population[i-1].copy()
                scores[i] = scores[i-1]
            else:
                scores[i] = _compute_c3(new_population[i])
        
        # Update population
        population = new_population[:pop_size]
        scores = scores[:pop_size]
    
    # Final local search polish on the best solution
    if time_left() > 5.0:
        max_iterations = 20000
        no_improve_count = 0
        patience_local = 2000
        
        for iteration in range(max_iterations):
            if time_left() < 3.0:
                break
            
            r = np.random.random()
            if r < 0.7:
                num_flips = 1
            elif r < 0.9:
                num_flips = 2
            else:
                num_flips = 3
            
            if num_flips == 1:
                idxs = [np.random.randint(N)]
            elif num_flips == 2:
                idxs = list(np.random.choice(N, size=2, replace=False))
            else:
                idxs = list(np.random.choice(N, size=3, replace=False))
            
            old_vals = best_step[idxs].copy()
            
            for idx in idxs:
                best_step[idx] = -best_step[idx]
            
            new_c3 = _compute_c3(best_step)
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                no_improve_count = 0
            else:
                for i, idx in enumerate(idxs):
                    best_step[idx] = old_vals[i]
                no_improve_count += 1
            
            if no_improve_count >= patience_local:
                if time_left() < 5.0:
                    break
                perturbation = np.random.choice([-1, 1], size=N).astype(np.int8)
                mask = np.random.random(N) < 0.2
                candidate = best_step.copy()
                candidate[mask] = perturbation[mask]
                c3_val = _compute_c3(candidate)
                if c3_val < best_c3:
                    best_step = candidate
                    best_c3 = c3_val
                no_improve_count = 0
    
    return best_step
