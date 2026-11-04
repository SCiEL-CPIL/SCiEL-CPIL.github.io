"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Approach: Genetic Algorithm (GA)
    # Objective: Minimize Integrated Sidelobe Level (ISL)
    # ISL = sum_{tau != 0} R(tau)^2
    
    # 1. Helper to compute ISL
    def compute_isl(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        # R[0] is the main lobe (lag 0). We sum the rest.
        # Due to floating point, R might have tiny imaginary parts, but we took real.
        # R is circular autocorrelation.
        return np.sum(R[1:]**2)
        
    # 2. Initialization
    pop_size = 60
    num_generations = 150
    
    # Create initial population of random binary vectors
    # f is a numpy array of shape (pop_size, N) with values -1, 1
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Evaluate initial fitness
    # We want to minimize ISL
    fitness = np.array([compute_isl(ind) for ind in population])
    
    best_fitness = np.min(fitness)
    best_individual = population[np.argmin(fitness)].copy()
    
    # 3. GA Loop
    for gen in range(num_generations):
        # Selection: Tournament Selection
        # We create a new population of the same size
        new_population = []
        
        for _ in range(pop_size):
            # Pick 3 random candidates
            idxs = np.random.choice(pop_size, 3, replace=False)
            # Pick the one with the lowest fitness (best)
            best_idx_in_tournament = idxs[np.argmin(fitness[idxs])]
            new_population.append(population[best_idx_in_tournament].copy())
            
        new_population = np.array(new_population)
        
        # Crossover: Uniform Crossover
        # For each pair, mix bits
        # We can do pairwise crossover
        for i in range(0, pop_size - 1, 2):
            # Create mask
            mask = np.random.random(N) > 0.5
            # Child 1
            child1 = new_population[i].copy()
            child1[mask] = new_population[i+1][mask]
            # Child 2
            child2 = new_population[i+1].copy()
            child2[mask] = new_population[i][mask]
            
            new_population[i] = child1
            new_population[i+1] = child2
            
        # Mutation: Bit Flip
        # Mutation rate 0.01
        mutation_rate = 0.01
        for i in range(pop_size):
            # Randomly select indices to flip
            flip_mask = np.random.random(N) < mutation_rate
            new_population[i][flip_mask] *= -1
            
        # Evaluate new population
        fitness = np.array([compute_isl(ind) for ind in new_population])
        
        # Elitism: Keep the best individual from the previous generation
        best_idx = np.argmin(fitness)
        prev_best_idx = np.argmin(np.array([compute_isl(best_individual)])) # This is just best_individual
        # Actually, simpler: track global best
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_individual = new_population[np.argmin(fitness)].copy()
            
        population = new_population
        
    return best_individual
# EVOLVE-BLOCK-END
