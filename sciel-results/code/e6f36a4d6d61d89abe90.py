import numpy as np
import time

def construct():
    N = 24
    # Precompute the DFT matrix for the 24th roots of unity
    # M[i, j] = exp(2 * pi * i * i * j / N)
    # Note: P(z) = sum c_k z^k. At z = exp(2*pi*i*m/N), P(z) = sum c_k exp(2*pi*i*m*k/N).
    # This is the DFT of c.
    k_idx = np.arange(N)
    m_idx = np.arange(N)
    # Matrix of shape (N, N) where rows are m, cols are k
    # DFT matrix element (m, k) = exp(2 * pi * i * m * k / N)
    DFT = np.exp(2j * np.pi * np.outer(m_idx, k_idx) / N)
    
    def evaluate_population(pop):
        # pop: (P, N) array of +1/-1
        # vals: (P, N) complex array
        # vals[p, m] = sum_k pop[p, k] * DFT[m, k]
        # This is pop @ DFT.T
        vals = pop @ DFT.T
        return np.max(np.abs(vals), axis=1)
    
    def local_search(seq, max_steps=100):
        # seq: 1D array of +1/-1
        # Returns refined seq and its fitness
        current_V = DFT @ seq
        current_fitness = np.max(np.abs(current_V))
        
        # Precompute the "flip vectors" for each bit position
        # If we flip bit j (index j), the new value at m is V[m] - 2 * seq[j] * DFT[m, j]
        # Let delta_j[m] = -2 * seq[j] * DFT[m, j]
        # We need to check max(abs(current_V + delta_j))
        
        for step in range(max_steps):
            improved = False
            best_j = -1
            best_f = current_fitness
            best_new_V = None
            
            for j in range(N):
                # Calculate the change vector for flipping bit j
                # delta_j = -2 * seq[j] * DFT[:, j]
                delta_j = -2.0 * seq[j] * DFT[:, j]
                new_V = current_V + delta_j
                f = np.max(np.abs(new_V))
                
                if f < best_f - 1e-12:
                    best_f = f
                    best_j = j
                    best_new_V = new_V
            
            if best_j != -1:
                # Apply the flip
                seq[best_j] *= -1
                current_V = best_new_V
                current_fitness = best_f
                improved = True
            else:
                break
                
        return seq, current_fitness

    # Initialize population
    pop_size = 500
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Start with a strong seed: Golay-Rudin-Shapiro
    def grs(n):
        if n == 1: return [1]
        a = grs(n//2)
        b = grs(n//2)
        return a + b + [-x for x in b]
    g32 = grs(32)
    population[0] = np.array(g32[:N])
    
    # Add a few random seeds that are slightly modified GRS
    for i in range(1, 10):
        pop = population[0].copy()
        flips = np.random.choice(N, size=2, replace=False)
        pop[flips] *= -1
        population[i] = pop

    # Evaluate initial fitness
    fitness = evaluate_population(population)
    
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_seq = population[best_idx].copy()
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # GA Parameters
    generations = 5000
    elite_size = 10
    tournament_size = 5
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Update best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_seq = population[0].copy()
            
        # Perform local search on the top few individuals to refine them
        # This is the key improvement: GA explores, LS exploits.
        for i in range(min(10, pop_size)):
            # Only do LS if we have time
            if time.time() - start_time > time_limit - 5:
                break
            # Run local search on a copy
            refined_seq, refined_fit = local_search(population[i].copy(), max_steps=50)
            if refined_fit < fitness[i]:
                population[i] = refined_seq
                fitness[i] = refined_fit
                if refined_fit < best_fitness:
                    best_fitness = refined_fit
                    best_seq = refined_seq.copy()
        
        # Create next generation
        new_population = np.empty_like(population)
        # Elitism
        new_population[:elite_size] = population[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            cand_fitness = fitness[candidates]
            winner_idx = candidates[np.argmin(cand_fitness)]
            parent = population[winner_idx]
            
            # Mutation: Flip 1 or 2 bits
            num_flips = 1 if np.random.random() < 0.7 else 2
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            child = parent.copy()
            child[flip_indices] *= -1
            
            new_population[idx] = child
            idx += 1
            
        population = new_population
        fitness = evaluate_population(population)
        
    # Final local search on the absolute best
    if time.time() - start_time < time_limit:
        final_seq, final_fit = local_search(best_seq.copy(), max_steps=100)
        if final_fit < best_fitness:
            best_seq = final_seq
            best_fitness = final_fit

    return [int(c) for c in best_seq]
