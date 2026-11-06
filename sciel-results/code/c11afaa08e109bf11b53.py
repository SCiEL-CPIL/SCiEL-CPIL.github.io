import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 24
    
    def grs_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    grs = grs_sequence(N)
    
    # Create diverse initial population
    # 1. GRS
    # 2. -GRS
    # 3. Reversed GRS
    # 4. Random
    def to_bits(coeffs):
        return [0 if c == -1 else 1 for c in coeffs]
    
    grs_bits = to_bits(grs)
    neg_grs_bits = to_bits([-c for c in grs])
    rev_grs_bits = to_bits(grs[::-1])
    neg_rev_grs_bits = to_bits([-c for c in grs[::-1]])
    
    pop_size = 400
    population = [grs_bits, neg_grs_bits, rev_grs_bits, neg_rev_grs_bits]
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(N)])
    
    # Grid for GA
    num_grid_points = 500
    angles = np.linspace(0, 2 * np.pi, num_grid_points, endpoint=False)
    z_grid = np.exp(1j * angles)
    z_powers = np.power(z_grid[:, None], np.arange(N)[None, :])
    
    def fast_evaluate(coeffs_bits):
        coeffs = np.array([2*b - 1 for b in coeffs_bits], dtype=float)
        vals = z_powers @ coeffs
        return np.max(np.abs(vals))
    
    fitnesses = [fast_evaluate(ind) for ind in population]
    best_val = min(fitnesses)
    best_ind = population[fitnesses.index(best_val)]
    
    # GA
    while time.time() < deadline:
        new_population = []
        # Elitism
        sorted_indices = np.argsort(fitnesses)
        for i in range(10):
            new_population.append(population[sorted_indices[i]])
            
        while len(new_population) < pop_size:
            # Tournament Selection
            c1_idx = random.randint(0, pop_size - 1)
            c2_idx = random.randint(0, pop_size - 1)
            c3_idx = random.randint(0, pop_size - 1)
            cands = [fitnesses[c1_idx], fitnesses[c2_idx], fitnesses[c3_idx]]
            p1_idx = [c1_idx, c2_idx, c3_idx][cands.index(min(cands))]
            
            c1_idx = random.randint(0, pop_size - 1)
            c2_idx = random.randint(0, pop_size - 1)
            c3_idx = random.randint(0, pop_size - 1)
            cands = [fitnesses[c1_idx], fitnesses[c2_idx], fitnesses[c3_idx]]
            p2_idx = [c1_idx, c2_idx, c3_idx][cands.index(min(cands))]
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Uniform Crossover
            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(N)]
            
            # Mutation
            if random.random() < 0.9:
                mut_pos = random.randint(0, N - 1)
                child[mut_pos] = 1 - child[mut_pos]
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [fast_evaluate(ind) for ind in population]
        
        current_best = min(fitnesses)
        if current_best < best_val:
            best_val = current_best
            best_ind = population[fitnesses.index(current_best)]
            
    # Polishing
    # Use high precision evaluation
    def high_precision_evaluate(coeffs_bits):
        coeffs = np.array([2*b - 1 for b in coeffs_bits], dtype=float)
        # 2000 points
        angles_hp = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        z_hp = np.exp(1j * angles_hp)
        z_powers_hp = np.power(z_hp[:, None], np.arange(N)[None, :])
        vals = z_powers_hp @ coeffs
        return np.max(np.abs(vals))
    
    current_best_bits = best_ind[:]
    current_best_val = high_precision_evaluate(current_best_bits)
    
    # Local Search: 1-bit flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            neighbor = current_best_bits[:]
            neighbor[i] = 1 - neighbor[i]
            n_val = high_precision_evaluate(neighbor)
            if n_val < current_best_val:
                current_best_val = n_val
                current_best_bits = neighbor
                improved = True
                
    # 2-bit flips if time remains
    if time.time() < deadline:
        for i in range(N):
            for j in range(i+1, N):
                neighbor = current_best_bits[:]
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                n_val = high_precision_evaluate(neighbor)
                if n_val < current_best_val:
                    current_best_val = n_val
                    current_best_bits = neighbor
                    # Restart 1-bit search?
                    # Just continue to next pair or break?
                    # Let's just update and break to restart the 1-bit loop
                    # But we are in a nested loop. 
                    # Let's just do one pass of 2-bit flips.
                    pass
                    
    # If we found a better one with 2-bit, run 1-bit again
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                neighbor = current_best_bits[:]
                neighbor[i] = 1 - neighbor[i]
                n_val = high_precision_evaluate(neighbor)
                if n_val < current_best_val:
                    current_best_val = n_val
                    current_best_bits = neighbor
                    improved = True

    return [2*b - 1 for b in current_best_bits]
# EVOLVE-BLOCK-END
