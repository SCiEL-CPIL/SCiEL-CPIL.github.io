import numpy as np
import time

N = 100
SUM_VAL = N // 2

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Population of sets of indices where h is 1
    # Each individual is a set of 50 indices
    pop_size = 100
    # Initialize population
    # Individual is a list of 50 indices
    population = []
    for _ in range(pop_size):
        indices = np.random.choice(N, SUM_VAL, replace=False)
        population.append(indices)
    
    best_fitness = -float('inf')
    best_individual = population[0].copy()
    
    def calculate_fitness(indices):
        # Create binary sequence
        h = np.zeros(N)
        h[indices] = 1.0
        
        # Calculate autocorrelation
        # C(k) = sum h_i h_{i-k}
        # Using FFT
        f = np.fft.fft(h)
        power = f * np.conj(f)
        ac = np.real(np.fft.ifft(power))
        # ac[0] is N/2 = 50
        # We want min of ac[1:]
        min_ac = np.min(ac[1:])
        return min_ac

    # Evaluate initial population
    fitnesses = []
    for ind in population:
        fit = calculate_fitness(ind)
        fitnesses.append(fit)
        if fit > best_fitness:
            best_fitness = fit
            best_individual = ind.copy()
    
    generations = 0
    while time.time() - start_time < time_limit:
        generations += 1
        new_population = []
        
        # Elitism: Keep the best 20%
        sorted_indices = np.argsort(fitnesses)[::-1]
        elite_count = pop_size // 5
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if fitnesses[idx1] > fitnesses[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
                
            idx3 = np.random.randint(0, pop_size)
            idx4 = np.random.randint(0, pop_size)
            if fitnesses[idx3] > fitnesses[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]
            
            # Crossover: Uniform crossover on the set of indices
            # Convert to boolean masks for easy mixing
            mask1 = np.zeros(N, dtype=bool)
            mask1[parent1] = True
            mask2 = np.zeros(N, dtype=bool)
            mask2[parent2] = True
            
            # Random mask
            cross_mask = np.random.rand(N) < 0.5
            
            # Child mask
            child_mask = np.logical_and(cross_mask, mask1) | np.logical_and(~cross_mask, mask2)
            
            # Check count
            count = np.sum(child_mask)
            if count != SUM_VAL:
                # Repair: if count > SUM_VAL, remove random ones from child_mask
                # if count < SUM_VAL, add random ones
                # It's easier to just sample new indices if repair is complex, 
                # but repair is faster.
                if count > SUM_VAL:
                    # Remove (count - SUM_VAL) random 1s
                    ones = np.where(child_mask)[0]
                    to_remove = np.random.choice(ones, count - SUM_VAL, replace=False)
                    child_mask[to_remove] = False
                else:
                    # Add (SUM_VAL - count) random 0s
                    zeros = np.where(~child_mask)[0]
                    to_add = np.random.choice(zeros, SUM_VAL - count, replace=False)
                    child_mask[to_add] = True
            
            child_indices = np.where(child_mask)[0]
            
            # Mutation: Swap 2 random indices (one 1, one 0) with probability
            if np.random.rand() < 0.1:
                ones = np.where(child_mask)[0]
                zeros = np.where(~child_mask)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    idx1_m = np.random.choice(ones)
                    idx0_m = np.random.choice(zeros)
                    child_mask[idx1_m] = False
                    child_mask[idx0_m] = True
                    child_indices = np.where(child_mask)[0]
            
            new_population.append(child_indices)
            
            # Time check inside loop
            if time.time() - start_time > time_limit:
                break
        
        population = new_population
        
        # Evaluate new population
        fitnesses = []
        for ind in population:
            fit = calculate_fitness(ind)
            fitnesses.append(fit)
            if fit > best_fitness:
                best_fitness = fit
                best_individual = ind.copy()
        
        # If no improvement in last 50 generations, maybe break? 
        # But let's run until time limit to be safe.
        
    # Convert best_individual to h
    h = np.zeros(N)
    h[best_individual] = 1.0
    return h
