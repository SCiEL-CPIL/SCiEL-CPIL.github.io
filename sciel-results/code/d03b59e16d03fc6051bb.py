"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def _compute_c3(h):
    """Compute the C3 upper bound for a signed step function h of length N.
    
    h is a 1D array of ±1 values.
    The autoconvolution is computed, and C3 is the maximum absolute value
    of the autoconvolution at non-zero lags (normalized).
    """
    # Autoconvolution via FFT for efficiency
    # Pad to avoid circular convolution effects
    size = 2 * N
    h_padded = np.zeros(size)
    h_padded[:N] = h
    # Linear autoconvolution
    conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2, n=size)
    # The autoconvolution at lag k for k=0..N-1 is conv[k]
    # conv[0] = sum(h^2) = N (always)
    # We care about non-zero lags
    # The maximum absolute value at non-zero lags
    nonzero_lags = conv[1:N]  # lags 1 to N-1
    if len(nonzero_lags) == 0:
        return 0.0
    max_val = np.max(np.abs(nonzero_lags))
    return max_val

def construct():
    deadline = time.time() + 110  # 10s safety margin
    
    # Initial population
    pop_size = 200
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Evaluate initial population
    fitness = np.array([_compute_c3(population[i]) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    best_sol = population[best_idx].copy()
    best_fit = fitness[best_idx]
    
    # Genetic algorithm
    gen = 0
    while time.time() < deadline:
        gen += 1
        # Tournament selection
        new_population = np.empty_like(population)
        for i in range(pop_size):
            # Tournament of size 3
            contestants = np.random.randint(0, pop_size, size=3)
            winner = contestants[np.argmin(fitness[contestants])]
            new_population[i] = population[winner]
        
        # Crossover and mutation
        for i in range(pop_size):
            if np.random.random() < 0.9:  # crossover rate
                # Single-point crossover
                parent2_idx = np.random.randint(0, pop_size)
                if parent2_idx != i:
                    point = np.random.randint(1, N)
                    child = new_population[i].copy()
                    child[point:] = new_population[parent2_idx].copy()[point:]
                    new_population[i] = child
            
            # Mutation: flip bits with small probability
            mutation_rate = 0.01
            mask = np.random.random(N) < mutation_rate
            new_population[i][mask] *= -1
        
        # Evaluate new population
        fitness = np.array([_compute_c3(new_population[i]) for i in range(pop_size)])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit = fitness[best_idx]
            best_sol = new_population[best_idx].copy()
        
        # Elitism: keep the best individual
        population = new_population
        population[best_idx] = best_sol
        fitness[best_idx] = best_fit
    
    return best_sol
# EVOLVE-BLOCK-END
