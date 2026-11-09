<<<<<<< SEARCH
def _swap_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Swap local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n - 1):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            for j in range(i + 1, n):
                candidate = best_order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms
=======
def _swap_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Swap local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n - 1):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            for j in range(i + 1, n):
                candidate = best_order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def _order_crossover(parent1, parent2, rng):
    """Order Crossover (OX) for permutations."""
    n = len(parent1)
    if n <= 2:
        return list(parent1)
    
    # Choose two random cut points
    i = rng.randint(0, n)
    j = rng.randint(0, n)
    if i > j:
        i, j = j, i
    
    # Create child
    child = [None] * n
    # Copy segment from parent1
    child[i:j+1] = parent1[i:j+1]
    
    # Fill remaining positions with elements from parent2 in order
    remaining = [x for x in parent2 if x not in child[i:j+1]]
    pos = 0
    for k in range(n):
        if k >= i and k <= j:
            continue
        child[k] = remaining[pos]
        pos += 1
    
    return child


def _mutation_swap(order, rng, rate=0.1):
    """Apply swap mutation with given probability."""
    n = len(order)
    child = list(order)
    for _ in range(int(rate * n)):
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i != j:
            child[i], child[j] = child[j], child[i]
    return child


def _ga_search(initial_orders, P, pop_size=20, generations=50, time_limit=None, start_time=None):
    """Genetic Algorithm with Order Crossover and mutation."""
    n = len(initial_orders[0])
    if n <= 2:
        return initial_orders[0], _makespan(initial_orders[0], P)
    
    # Initialize population
    population = [list(order) for order in initial_orders[:pop_size]]
    # Add random permutations to fill population
    while len(population) < pop_size:
        population.append(list(np.random.permutation(n)))
    
    # Evaluate initial population
    fitness = [_makespan(ind, P) for ind in population]
    best_idx = np.argmin(fitness)
    best_order = population[best_idx]
    best_ms = fitness[best_idx]
    
    for gen in range(generations):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        # Selection: tournament selection
        new_population = []
        for _ in range(pop_size):
            # Tournament selection
            tournament_size = 3
            indices = np.random.choice(pop_size, size=tournament_size, replace=False)
            winner = indices[np.argmin([fitness[i] for i in indices])]
            new_population.append(list(population[winner]))
        
        # Crossover and mutation
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                child = _order_crossover(new_population[i], new_population[i+1], np.random)
                child = _mutation_swap(child, np.random, rate=0.1)
                offspring.append(child)
            else:
                offspring.append(new_population[i])
        
        population = offspring
        fitness = [_makespan(ind, P) for ind in population]
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_ms - 1e-10:
            best_ms = fitness[gen_best_idx]
            best_order = population[gen_best_idx]
        
        # Elitism: keep best individual
        if fitness[gen_best_idx] < fitness[np.argmin(fitness)]:
            pass  # Already updated
    
    return best_order, best_ms
>>>>>>> REPLACE

