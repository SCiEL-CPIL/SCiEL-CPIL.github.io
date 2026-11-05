import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_score = float('inf')
    best_poly = None
    
    def update_best(poly_dict, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            max_exp = max(poly_dict.keys())
            lst = [0] * (max_exp + 1)
            for e, c in poly_dict.items():
                lst[e] = c
            best_poly = lst
    
    def get_score(d):
        """Exact count of nonzero terms in square."""
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)
    
    def is_valid(poly_dict):
        if len(poly_dict) != 12:
            return False
        for c in poly_dict.values():
            if not (-64 <= c <= 64) or c == 0:
                return False
        if max(poly_dict.keys()) > 40:
            return False
        return True
    
    # ====================================================================
    # STRATEGY: Genetic Algorithm + Systematic Pattern Enumeration
    # 
    # 1. Enumerate small coefficient patterns (±1, ±2) on fixed exponent sets
    #    that are known to produce good cancellations.
    # 2. Use a genetic algorithm with tournament selection, crossover, and
    #    mutation to explore the 12-dimensional coefficient space.
    # 3. Apply aggressive hill-climbing from each GA generation's best.
    # ====================================================================
    
    # Fixed exponent sets to try (ascending degree, 12 terms each)
    exponent_sets = [
        list(range(12)),          # 0..11
        list(range(0, 41, 4)),    # 0,4,8,...,44 -> trim to 12
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],  # clustered
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # even AP
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],  # pairs
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],  # AP step 3
    ]
    
    # Trim exponent sets to valid range
    valid_exp_sets = []
    for es in exponent_sets:
        if max(es) <= 40 and len(es) == 12:
            valid_exp_sets.append(es)
    
    if not valid_exp_sets:
        valid_exp_sets = [list(range(12))]
    
    def compute_square_count(coeffs, exponents):
        """Compute nonzero term count in square given coeffs and exponents."""
        d = {exponents[i]: coeffs[i] for i in range(12) if coeffs[i] != 0}
        if len(d) != 12:
            return float('inf')
        return get_score(d)
    
    def hill_climb(poly_dict, max_iters=5000):
        """Aggressive local search from a starting point."""
        curr = dict(poly_dict)
        curr_score = get_score(curr)
        T = 3.0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick 1-2 coefficients to modify
            n_mods = random.choice([1, 1, 2])
            keys = list(curr.keys())
            indices = random.sample(range(12), n_mods)
            
            old_vals = [curr[keys[i]] for i in indices]
            for idx in indices:
                k = keys[idx]
                # Try small perturbations
                old = curr[k]
                new = old + random.choice([-2, -1, 1, 2])
                if -64 <= new <= 64 and new != 0:
                    curr[k] = new
                else:
                    curr[k] = random.choice([-2, -1, 1, 2])
            
            new_score = get_score(curr)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    update_best(curr, curr_score)
            else:
                delta = new_score - curr_score
                if random.random() < pow(2.718, -delta / max(T, 0.1)):
                    curr_score = new_score
                else:
                    for idx, old in zip(indices, old_vals):
                        curr[keys[idx]] = old
            
            T *= 0.998
        
        return curr_score
    
    # Phase 1: Systematic enumeration of small patterns
    small_coeffs = [-2, -1, 1, 2]
    
    for exp_set in valid_exp_sets:
        if time.time() > deadline:
            break
        
        # Try all combinations of ±1, ±2 for first few, then randomize rest
        # This is too many, so we do smart sampling
        
        # Pattern 1: all ±1
        for sign_pattern in [[1]*12, [-1]*12, [1,-1]*6, [1,1,-1,-1]*3]:
            if time.time() > deadline:
                break
            coeffs = sign_pattern[:12]
            d = {exp_set[i]: coeffs[i] for i in range(12)}
            if is_valid(d):
                s = get_score(d)
                if s < best_score:
                    update_best(d, s)
                if time.time() < deadline:
                    hill_climb(d, max_iters=2000)
        
        # Pattern 2: mixed ±1, ±2
        for _ in range(200):
            if time.time() > deadline:
                break
            coeffs = [random.choice(small_coeffs) for _ in range(12)]
            d = {exp_set[i]: coeffs[i] for i in range(12)}
            if is_valid(d):
                s = get_score(d)
                if s < best_score:
                    update_best(d, s)
                if time.time() < deadline:
                    hill_climb(d, max_iters=1000)
    
    # Phase 2: Genetic Algorithm
    population_size = 100
    generation = 0
    
    # Initialize population
    population = []
    for i in range(population_size):
        exp_set = random.choice(valid_exp_sets)
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        d = {exp_set[j]: coeffs[j] for j in range(12)}
        if is_valid(d):
            score = get_score(d)
            population.append((score, exp_set, coeffs))
    
    # Sort by score
    population.sort(key=lambda x: x[0])
    
    # Update best from initial population
    if population:
        best_in_pop = population[0]
        if best_in_pop[0] < best_score:
            d = {best_in_pop[1][j]: best_in_pop[2][j] for j in range(12)}
            update_best(d, best_in_pop[0])
    
    # GA loop
    while time.time() < deadline:
        generation += 1
        
        # Selection: tournament
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, min(k, len(pop)))
            return min(candidates, key=lambda x: x[0])
        
        # Create new population
        new_pop = []
        
        # Elitism: keep top 5
        for i in range(min(5, len(population))):
            new_pop.append(population[i])
        
        # Fill rest with offspring
        while len(new_pop) < population_size:
            if time.time() > deadline:
                break
            
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: uniform
            child_exp = list(parent1[1])
            child_coeffs = []
            for j in range(12):
                if random.random() < 0.5:
                    child_coeffs.append(parent1[2][j])
                else:
                    child_coeffs.append(parent2[2][j])
            
            # Mutation: 20% chance per coefficient
            for j in range(12):
                if random.random() < 0.2:
                    child_coeffs[j] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            
            # Ensure nonzero and in range
            for j in range(12):
                if child_coeffs[j] == 0:
                    child_coeffs[j] = 1
                child_coeffs[j] = max(-64, min(64, child_coeffs[j]))
            
            d = {child_exp[j]: child_coeffs[j] for j in range(12)}
            if is_valid(d):
                score = get_score(d)
                new_pop.append((score, child_exp, child_coeffs))
                if score < best_score:
                    update_best(d, score)
            
            # Occasionally try a different exponent set
            if random.random() < 0.05:
                child_exp = random.choice(valid_exp_sets)
                d = {child_exp[j]: child_coeffs[j] for j in range(12)}
                if is_valid(d):
                    score = get_score(d)
                    new_pop.append((score, child_exp, child_coeffs))
                    if score < best_score:
                        update_best(d, score)
        
        population = new_pop
        population.sort(key=lambda x: x[0])
        
        # Hill climb from the best individual every 5 generations
        if generation % 5 == 0 and population:
            best_in_pop = population[0]
            d = {best_in_pop[1][j]: best_in_pop[2][j] for j in range(12)}
            hill_climb(d, max_iters=3000)
    
    # Phase 3: Final aggressive hill climbing from best found
    if best_poly is not None and time.time() < deadline:
        final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
        if len(final_dict) == 12:
            hill_climb(final_dict, max_iters=10000)
    
    # Fallback
    if best_poly is None:
        best_poly = [1] * 12
    
    # Final validation
    final_dict = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    if len(final_dict) != 12:
        best_poly = [1] * 12
    else:
        for c in final_dict.values():
            if not (-64 <= c <= 64):
                best_poly = [1] * 12
                break
        max_exp = max(final_dict.keys())
        if max_exp > 40:
            best_poly = [1] * 12
    
    return best_poly
