import numpy as np
import time

def construct():
    N = 24
    # Precompute DFT matrix
    k_idx = np.arange(N)
    m_idx = np.arange(N)
    DFT = np.exp(2j * np.pi * np.outer(m_idx, k_idx) / N)
    DFT_T = DFT.T # For pop @ DFT.T
    
    def evaluate_population(pop):
        # pop: (P, N)
        # vals: (P, N)
        vals = pop @ DFT_T
        return np.max(np.abs(vals), axis=1)
        
    def evaluate_single(seq):
        vals = DFT @ seq
        return np.max(np.abs(vals))

    # Initialization
    # Use a mix of random and GRS-like sequences
    pop_size = 1000
    population = np.random.choice([-1, 1], size=(pop_size, N))
    
    # Add some structured sequences
    # Rudin-Shapiro for 16, padded? Or just random.
    # Let's add a few known good patterns if possible, or just rely on the large population.
    # To ensure we find the global optimum, a large population + strong LS is key.
    
    fitness = evaluate_population(population)
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_seq = population[best_idx].copy()
    
    start_time = time.time()
    time_limit = 105.0
    
    # Enhanced Local Search
    def local_search(seq, max_steps=1000):
        # seq is a numpy array of +1/-1
        # V = DFT @ seq
        V = DFT @ seq
        current_fitness = np.max(np.abs(V))
        
        # Precompute columns of DFT for fast updates
        # DFT[:, j] is the change vector for flipping bit j (scaled by -2*seq[j])
        
        improved = True
        while improved and max_steps > 0:
            improved = False
            max_steps -= 1
            
            # Try single flips
            best_j = -1
            best_f = current_fitness
            
            # Vectorized check for single flips is hard to do all at once for max, 
            # but we can loop j and update V in O(N)
            for j in range(N):
                # V_new = V - 2 * seq[j] * DFT[:, j]
                delta = -2 * seq[j] * DFT[:, j]
                V_new = V + delta
                f = np.max(np.abs(V_new))
                if f < best_f - 1e-9:
                    best_f = f
                    best_j = j
            
            if best_j != -1:
                # Apply flip
                seq[best_j] *= -1
                V = V - 2 * (seq[best_j] * -1) * DFT[:, best_j] # Wait, seq[best_j] already flipped?
                # Let's just recalculate V from scratch to avoid bugs, or update carefully.
                # V_new = V_old - 2 * c_old * DFT_col
                # c_new = -c_old
                # So V_new = V_old + 2 * c_new * DFT_col ? No.
                # V = sum c_k DFT_k
                # V_new = sum c'_k DFT_k = V - c_j DFT_j + (-c_j) DFT_j = V - 2 c_j DFT_j
                # We calculated V_new = V + delta where delta = -2 * seq[j] * DFT[:, j].
                # So V = V_new.
                V = V + (-2 * (seq[best_j] * -1) * DFT[:, best_j]) # This is confusing.
                # Let's just do:
                # old_c = seq[best_j] (before flip)
                # We didn't store old_c. 
                # Let's restart the loop logic cleanly.
                
                # Actually, let's just recompute V = DFT @ seq. N=24, it's cheap.
                V = DFT @ seq
                current_fitness = best_f
                improved = True
                continue # Restart search from new state
                
            # Try double flips if single didn't improve
            # This helps escape local minima
            best_pair = None
            best_pair_f = current_fitness
            
            # Sample some pairs if N is large, but N=24 is small enough to check all?
            # 276 pairs. Each check is O(N) if we have V.
            # V_new = V - 2 c_j DFT_j - 2 c_k DFT_k
            # We can do this.
            
            for j in range(N - 1):
                for k in range(j + 1, N):
                    delta = -2 * seq[j] * DFT[:, j] - 2 * seq[k] * DFT[:, k]
                    V_new = V + delta
                    f = np.max(np.abs(V_new))
                    if f < best_pair_f - 1e-9:
                        best_pair_f = f
                        best_pair = (j, k)
                        
            if best_pair:
                j, k = best_pair
                seq[j] *= -1
                seq[k] *= -1
                V = DFT @ seq
                current_fitness = best_pair_f
                improved = True
                continue # Restart
                
            # If no improvement, break
            break
            
        return seq, current_fitness

    # GA Loop
    generations = 2000
    elite_size = 20
    tournament_size = 5
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort population
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_seq = population[0].copy()
            
        # Local Search on top individuals
        # We have time for maybe 5-10 LS runs per generation?
        # LS is now O(N^2 * steps). N=24. Steps ~ 100. 24*24*100 = 57600 ops. Very fast.
        # We can run LS on top 10.
        for i in range(min(10, pop_size)):
            if time.time() - start_time > time_limit - 10:
                break
            refined_seq, refined_fit = local_search(population[i].copy(), max_steps=200)
            if refined_fit < fitness[i]:
                population[i] = refined_seq
                fitness[i] = refined_fit
                if refined_fit < best_fitness:
                    best_fitness = refined_fit
                    best_seq = refined_seq.copy()
        
        # Create next generation
        new_population = np.empty_like(population)
        new_population[:elite_size] = population[:elite_size]
        
        idx = elite_size
        while idx < pop_size:
            # Tournament selection
            candidates = np.random.choice(pop_size, size=tournament_size, replace=False)
            cand_fitness = fitness[candidates]
            winner_idx = candidates[np.argmin(cand_fitness)]
            parent = population[winner_idx]
            
            # Crossover? For binary sequences, uniform crossover or just mutation.
            # Let's use mutation only for simplicity and speed, or 1-point crossover.
            # Given the "flattest" nature, mutation is often better.
            child = parent.copy()
            
            # Adaptive mutation rate?
            # Start high, decrease?
            # Or just fixed 0.1
            flip_mask = np.random.random(N) < 0.15
            child[flip_mask] *= -1
            
            new_population[idx] = child
            idx += 1
            
        population = new_population
        fitness = evaluate_population(population)
        
    # Final LS on best
    if time.time() - start_time < time_limit:
        final_seq, final_fit = local_search(best_seq.copy(), max_steps=500)
        if final_fit < best_fitness:
            best_seq = final_seq
            best_fitness = final_fit

    return [int(c) for c in best_seq]
