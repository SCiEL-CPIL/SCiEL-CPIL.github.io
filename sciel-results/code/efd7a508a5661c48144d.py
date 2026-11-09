import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    MAX_LEN = MAX_VAL + 1 # 129

    # Pre-compute indices for N_same check
    # For each k in [128, 256], store k//2 if k even, else -1
    half_indices = []
    for k in range(N_LOW, N_HIGH + 1):
        if k % 2 == 0:
            half_indices.append(k // 2)
        else:
            half_indices.append(-1)
    half_indices = np.array(half_indices, dtype=np.int32)

    def evaluate_bitmask(mask):
        """
        mask: numpy array of 0/1 of length 129
        Returns (peak, is_covered)
        """
        # Use FFT for convolution? Or np.convolve?
        # np.convolve is fine for small arrays.
        v = mask.astype(np.float64)
        c = np.convolve(v, v, mode='full')
        
        # Extract relevant part [128, 256]
        # c indices 0..256
        c_slice = c[N_LOW:N_HIGH+1]
        
        # Calculate N_same for each k in slice
        # half_indices[i] corresponds to k = N_LOW + i
        # If half_indices[i] != -1, check v[half_indices[i]]
        valid_halves = half_indices[half_indices >= 0]
        # We need to add 1 to c_slice where half is valid and v[half] is 1.
        # This is slightly tricky vectorized.
        
        # Simpler loop for N_same since RANGE_SIZE is 129
        n_same_arr = np.zeros(RANGE_SIZE, dtype=np.int32)
        for i in range(RANGE_SIZE):
            h = half_indices[i]
            if h != -1:
                n_same_arr[i] = mask[h]
                
        # M[k] = (c[k] + N_same[k]) / 2
        # c_slice is float, convert to int
        c_int = c_slice.astype(np.int32)
        
        # Avoid floating point issues in division?
        # c[k] and n_same[k] have same parity?
        # c[k] = 2 * N_distinct + N_same.
        # c[k] + N_same = 2 * N_distinct + 2 * N_same = 2 * (N_distinct + N_same).
        # So (c[k] + N_same) is always even.
        
        counts = (c_int + n_same_arr) // 2
        
        if not np.all(counts > 0):
            return float('inf'), False
            
        return int(np.max(counts)), True

    # Genetic Algorithm
    POP_SIZE = 40
    TOURNAMENT_SIZE = 5
    
    def init_pop():
        pop = []
        for _ in range(POP_SIZE):
            mask = np.zeros(MAX_LEN, dtype=np.int8)
            mask[0] = 1
            mask[MAX_VAL] = 1
            # Random density
            num_add = random.randint(20, 60)
            candidates = np.arange(1, MAX_VAL)
            np.random.shuffle(candidates)
            mask[candidates[:num_add]] = 1
            pop.append(mask)
        return pop

    population = init_pop()
    
    # Evaluate initial
    best_peak_global = float('inf')
    best_mask_global = None
    
    for mask in population:
        peak, covered = evaluate_bitmask(mask)
        if covered and peak < best_peak_global:
            best_peak_global = peak
            best_mask_global = mask.copy()
            
    # Store fitness in list of tuples (mask, peak, covered)
    # To avoid re-evaluating, we keep the population as masks and a parallel list of fitnesses?
    # Or just re-evaluate in tournament? 
    # With POP_SIZE 40 and TOURNAMENT 5, we do 20 * 5 = 100 evaluations per generation?
    # No, we select 2 parents, so 2 tournaments. 2 * 5 = 10 evaluations per child.
    # 40 children * 10 evals = 400 evals per generation.
    # 1 eval is ~10-50us? np.convolve is fast.
    # 400 * 50us = 20ms per generation.
    # 85s / 20ms = 4000 generations. This is plenty.

    generation = 0
    stagnation_counter = 0
    
    while time.time() < deadline - 5:
        generation += 1
        new_population = []
        
        # Elitism
        if best_mask_global is not None:
            new_population.append(best_mask_global.copy())
            
        # Generate offspring
        while len(new_population) < POP_SIZE:
            # Selection
            parent1 = None
            parent2 = None
            
            # Tournament 1
            contenders = random.sample(population, TOURNAMENT_SIZE)
            best_c = None
            best_score = float('inf')
            for c in contenders:
                # We need fitness. 
                # Strategy: Keep population as list of (mask, peak, covered)
                # But mutation changes mask, so we need to re-evaluate offspring.
                # Let's maintain population as just masks, and compute fitness on the fly in tournament?
                # That's inefficient.
                # Better: Maintain list of (mask, peak, covered).
                pass
                
            # Let's restructure: population is list of dicts or tuples.
            # But I already initialized `population` as list of masks.
            # Let's convert to list of (mask, peak, covered)
            # I'll do this conversion inside the loop or before.
            
            # To keep it simple and fast, let's just re-evaluate in tournament.
            # It's only 10 evals per child.
            
            def tournament_select(pop_masks):
                best_m = None
                best_p = float('inf')
                for _ in range(TOURNAMENT_SIZE):
                    m = random.choice(pop_masks)
                    p, cov = evaluate_bitmask(m)
                    if not cov:
                        p = float('inf')
                    if p < best_p:
                        best_p = p
                        best_m = m
                return best_m
                
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover (Uniform)
            child = np.random.choice([parent1, parent2], size=MAX_LEN)
            
            # Mutation
            # Flip bits with probability MUTATION_RATE
            # MUTATION_RATE = 0.05 (tuneable, but let's fix)
            mut_mask = np.random.random(MAX_LEN) < 0.05
            child[mut_mask] = 1 - child[mut_mask]
            
            # Ensure 0 and 128 are in set?
            # Not strictly necessary, but helpful for boundaries.
            # If we remove them, coverage might fail.
            # Let's force them?
            # child[0] = 1
            # child[MAX_VAL] = 1
            # This reduces search space but might help.
            # Let's try without forcing first to see if GA finds it.
            
            new_population.append(child)
            
        population = new_population
        
        # Evaluate new population and update best
        for mask in population:
            peak, covered = evaluate_bitmask(mask)
            if covered and peak < best_peak_global:
                best_peak_global = peak
                best_mask_global = mask.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1 # This logic is wrong, stagnation should be per generation best
        
        # Check stagnation
        gen_best_peak = float('inf')
        for mask in population:
            p, c = evaluate_bitmask(mask)
            if c and p < gen_best_peak:
                gen_best_peak = p
                
        if gen_best_peak < best_peak_global:
            best_peak_global = gen_best_peak
            # Find the mask
            for mask in population:
                p, c = evaluate_bitmask(mask)
                if c and p == gen_best_peak:
                    best_mask_global = mask.copy()
                    break
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If stagnating, restart some individuals?
        if stagnation_counter > 20:
            # Random restart a few individuals
            for i in range(5):
                population[i] = np.zeros(MAX_LEN, dtype=np.int8)
                population[i][0] = 1
                population[i][MAX_VAL] = 1
                num_add = random.randint(20, 60)
                candidates = np.arange(1, MAX_VAL)
                np.random.shuffle(candidates)
                population[i][candidates[:num_add]] = 1
            stagnation_counter = 0

    if best_mask_global is None:
        # Fallback
        mask = np.ones(MAX_LEN, dtype=np.int8)
        best_mask_global = mask
        
    result = [i for i in range(MAX_LEN) if best_mask_global[i] == 1]
    if not result:
        result = [0]
        
    return result
