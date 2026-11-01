"""
Genetic Algorithm approach for the autocorrelation inequality task.
Strategy: Population-based search with uniform crossover and bit-flip mutation.
This is structurally different from the single-point simulated annealing approach
by leveraging recombination to combine good features from multiple solutions.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_sidelobe(h):
    """Compute max |autocorr| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))

def _evaluate_population(pop):
    """Evaluate a population of sequences, returning array of values."""
    values = np.array([_compute_sidelobe(h) for h in pop])
    return values

def _genetic_algorithm(deadline):
    """
    Genetic Algorithm to minimize max autocorrelation sidelobe.
    
    - Population of random +/-1 sequences
    - Selection: Tournament selection
    - Crossover: Uniform crossover
    - Mutation: Bit flip with low probability
    - Elitism: Keep best individuals
    """
    pop_size = 200  # Small population to fit in time
    elite_size = 10
    mutation_rate = 0.01  # 1% chance to flip each bit
    
    start_time = time.time()
    
    # Initialize population
    pop = np.random.choice([-1.0, 1.0], size=(pop_size, N))
    values = _evaluate_population(pop)
    
    # Track best
    best_idx = np.argmin(values)
    best_h = pop[best_idx].copy()
    best_val = values[best_idx]
    
    generation = 0
    
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(values)
        sorted_pop = pop[sorted_indices]
        sorted_values = values[sorted_indices]
        
        # Update best
        if sorted_values[0] < best_val:
            best_val = sorted_values[0]
            best_h = sorted_pop[0].copy()
        
        # Elitism: keep top elite_size individuals
        new_pop = [sorted_pop[i].copy() for i in range(min(elite_size, pop_size))]
        
        # Fill rest of population with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            def tournament_select():
                # Pick 3 random candidates
                candidates = np.random.randint(0, pop_size, size=3)
                # Select the one with best value
                best_cand = min(candidates, key=lambda c: values[c])
                return pop[best_cand]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Uniform crossover
            mask = np.random.rand(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation: flip bits with probability mutation_rate
            mut_mask = np.random.rand(N) < mutation_rate
            child[mut_mask] *= -1
            
            new_pop.append(child)
        
        pop = np.array(new_pop)
        values = _evaluate_population(pop)
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_h, best_val

def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run genetic algorithm
    h_best, val_best = _genetic_algorithm(deadline)
    
    return h_best
# EVOLVE-BLOCK-END
