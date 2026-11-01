import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    Uses a Genetic Algorithm (Evolutionary Strategy) to explore the search space.
    """
    np.random.seed(42)
    
    # Population size
    pop_size = 200
    # Mutation rate (probability of flipping a bit)
    mutation_rate = 0.02
    # Crossover rate
    crossover_rate = 0.8
    
    def get_peak_sidelobe(f):
        """
        Computes the peak absolute value of the non-zero lag autocorrelation.
        """
        # Zero-pad to 2*N to avoid circular convolution artifacts
        F = np.fft.fft(f, n=2*N)
        acf = np.fft.ifft(F * np.conj(F)).real
        
        # Extract non-zero lags
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        pos_lags = acf[1:N]
        neg_lags = acf[N+1:2*N]
        
        if len(pos_lags) == 0 and len(neg_lags) == 0:
            return 0.0
            
        return np.max(np.abs(np.concatenate([pos_lags, neg_lags])))

    # Initialize population with random signs
    population = np.random.choice([-1, 1], (pop_size, N))
    fitnesses = np.array([get_peak_sidelobe(ind) for ind in population])
    
    best_idx = np.argmin(fitnesses)
    best_f = population[best_idx].copy()
    best_peak = fitnesses[best_idx]
    
    start_time = time.time()
    max_time = 100.0 # Leave buffer for safety
    
    generation = 0
    while (time.time() - start_time) < max_time:
        generation += 1
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitnesses)
        population = population[sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        
        # Keep the top half as parents (elite)
        n_parents = pop_size // 2
        parents = population[:n_parents]
        
        # Generate offspring
        new_population = []
        
        for i in range(pop_size // 2):
            # Tournament selection to pick two parents
            p1_idx = np.random.choice(n_parents)
            p2_idx = np.random.choice(n_parents)
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Uniform crossover
                mask = np.random.rand(N) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            # Flip bits with probability mutation_rate
            mut_mask = np.random.rand(N) < mutation_rate
            child[mut_mask] *= -1
            
            new_population.append(child)
            
            # Ensure we fill the rest of the population if pop_size is odd or if we want to keep elites
            # Actually, let's keep the top elites in the new population directly
            # So we only need to generate pop_size - n_parents new individuals?
            # Let's simplify: Keep top n_parents, generate pop_size - n_parents new ones.
            # The loop above generates pop_size//2 which is n_parents.
            # So new_population has n_parents individuals.
            
        # Combine elites and new offspring
        new_population = np.vstack([parents, np.array(new_population)])
        
        # Evaluate fitness of the new population
        new_fitnesses = np.array([get_peak_sidelobe(ind) for ind in new_population])
        
        # Update best
        current_best_idx = np.argmin(new_fitnesses)
        if new_fitnesses[current_best_idx] < best_peak:
            best_peak = new_fitnesses[current_best_idx]
            best_f = new_population[current_best_idx].copy()
            
        population = new_population
        fitnesses = new_fitnesses

    return best_f
# EVOLVE-BLOCK-END
