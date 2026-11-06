<<<<<<< SEARCH
def _genetic_algorithm(initial_comparators, deadline, rng):
    """
    Genetic algorithm evolving layer-based network representations.
    """
    pop_size = 24
    population = []
    
    initial_layers = _decompose_to_peels(initial_comparators)
    population.append(initial_layers)
    
    for _ in range(pop_size - 1):
        if rng.random() < 0.5:
            mutated = _mutate_layers(initial_layers, rng, mutation_rate=0.5)
            population.append(mutated)
        else:
            n_comps = rng.randint(40, 80)
            random_comps = []
            for _ in range(n_comps):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                random_comps.append((i, j))
            random_layers = _decompose_to_peels(random_comps)
            population.append(random_layers)
    
    best_layers = None
    best_penalty = float('inf')
    best_comps = None
    
    def _fitness(comps):
        n_comps = len(comps)
        if n_comps > 4096:
            return float('inf')
        
        mismatches_small = _simulate_on_inputs(comps, _SAMPLE_INPUTS_SMALL, _SAMPLE_SORTED_SMALL)
        if mismatches_small > 0:
            return n_comps + mismatches_small * 0.1
        
        mismatches_large = _simulate_on_inputs(comps, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE)
        if mismatches_large > 0:
            return n_comps + mismatches_large * 0.05
        
        if is_sorted_network_full(comps):
            return float(n_comps)
        
        return n_comps + 100

    for layers in population:
        comps = _layers_to_comparators(layers)
        penalty = _fitness(comps)
        if penalty < best_penalty:
            best_penalty = penalty
            best_layers = [layer[:] for layer in layers]
            best_comps = comps[:]
        if time.time() > deadline:
            break
    
    generations = 0
    max_generations = 200
    
    while generations < max_generations:
        if time.time() > deadline:
            break
        
        generations += 1
        
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            penalty = _fitness(comps)
            scored.append((penalty, layers, comps))
            if time.time() > deadline:
                break
        
        scored.sort(key=lambda x: x[0])
        
        if scored[0][0] < best_penalty:
            best_penalty = scored[0][0]
            best_layers = [layer[:] for layer in scored[0][1]]
            best_comps = scored[0][2][:]
        
        elite_count = max(4, pop_size // 4)
        new_population = []
        for i in range(elite_count):
            if i < len(scored):
                new_population.append([layer[:] for layer in scored[i][1]])
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            parent1_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            parent2_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            
            if parent1_idx == parent2_idx:
                parent2_idx = (parent2_idx + 1) % len(scored)
            
            p1 = scored[parent1_idx][1]
            p2 = scored[parent2_idx][1]
            
            if rng.random() < 0.7:
                child = _crossover_layers(p1, p2, rng)
            else:
                child = [layer[:] for layer in p1]
            
            mutation_rate = 0.2 + 0.3 * (generations / max_generations)
            child = _mutate_layers(child, rng, mutation_rate=mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_comps, best_layers
=======
def _genetic_algorithm(initial_comparators, deadline, rng):
    """
    Genetic algorithm evolving layer-based network representations.
    """
    pop_size = 24
    population = []
    
    initial_layers = _decompose_to_peels(initial_comparators)
    population.append(initial_layers)
    
    for _ in range(pop_size - 1):
        if rng.random() < 0.5:
            mutated = _mutate_layers(initial_layers, rng, mutation_rate=0.5)
            population.append(mutated)
        else:
            n_comps = rng.randint(40, 80)
            random_comps = []
            for _ in range(n_comps):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                random_comps.append((i, j))
            random_layers = _decompose_to_peels(random_comps)
            population.append(random_layers)
    
    best_layers = None
    best_penalty = float('inf')
    best_comps = None
    
    def _fitness(comps):
        n_comps = len(comps)
        if n_comps > 4096:
            return float('inf')
        
        mismatches_small = _simulate_on_inputs(comps, _SAMPLE_INPUTS_SMALL, _SAMPLE_SORTED_SMALL)
        if mismatches_small > 0:
            return n_comps + mismatches_small * 0.1
        
        mismatches_large = _simulate_on_inputs(comps, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE)
        if mismatches_large > 0:
            return n_comps + mismatches_large * 0.05
        
        if is_sorted_network_full(comps):
            return float(n_comps)
        
        return n_comps + 100

    for layers in population:
        comps = _layers_to_comparators(layers)
        penalty = _fitness(comps)
        if penalty < best_penalty:
            best_penalty = penalty
            best_layers = [layer[:] for layer in layers]
            best_comps = comps[:]
        if time.time() > deadline:
            break
    
    generations = 0
    max_generations = 200
    
    while generations < max_generations:
        if time.time() > deadline:
            break
        
        generations += 1
        
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            penalty = _fitness(comps)
            scored.append((penalty, layers, comps))
            if time.time() > deadline:
                break
        
        scored.sort(key=lambda x: x[0])
        
        if scored[0][0] < best_penalty:
            best_penalty = scored[0][0]
            best_layers = [layer[:] for layer in scored[0][1]]
            best_comps = scored[0][2][:]
        
        elite_count = max(4, pop_size // 4)
        new_population = []
        for i in range(elite_count):
            if i < len(scored):
                new_population.append([layer[:] for layer in scored[i][1]])
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            parent1_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            parent2_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            
            if parent1_idx == parent2_idx:
                parent2_idx = (parent2_idx + 1) % len(scored)
            
            p1 = scored[parent1_idx][1]
            p2 = scored[parent2_idx][1]
            
            if rng.random() < 0.7:
                child = _crossover_layers(p1, p2, rng)
            else:
                child = [layer[:] for layer in p1]
            
            mutation_rate = 0.2 + 0.3 * (generations / max_generations)
            child = _mutate_layers(child, rng, mutation_rate=mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_comps, best_layers


def _greedy_optimization(initial_comparators, deadline, rng):
    """
    Greedy constructive optimization:
    1. Build a network greedily from scratch.
    2. Validate and prune.
    3. Use as alternative to GA.
    """
    best_comps = None
    best_len = float('inf')
    
    # Multiple restarts with different random seeds
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed_offset = restart * 1000
        restart_rng = random.Random(seed_offset + 12345)
        
        # Greedy build
        candidate = _greedy_build_network(deadline, restart_rng, max_comparators=90)
        
        if not candidate:
            continue
        
        # Validate on full set
        if is_sorted_network_full(candidate):
            if len(candidate) < best_len:
                best_comps = candidate[:]
                best_len = len(candidate)
        
        # Try to improve via deletion
        if time.time() < deadline - 5.0:
            current = candidate[:]
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                progress = False
                i = 0
                while i < len(current):
                    if time.time() > deadline - 5.0:
                        break
                    success, new_current = _try_delete(current, i, deadline - 5.0)
                    if success:
                        current = new_current
                        if len(current) < best_len:
                            best_comps = current[:]
                            best_len = len(current)
                        progress = True
                        continue
                    i += 1
                if not progress:
                    break
    
    if best_comps is None:
        # Fallback to initial
        return initial_comparators, None
    
    return best_comps, None
>>>>>>> REPLACE

