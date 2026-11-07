<<<<<<< SEARCH
def _local_search(routes, instance, dist, deadline):
    """Full local search with deadline."""
    start_time = time.time()
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # 2-opt on each route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _two_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Or-opt
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _or_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Inter-route moves
        if len(routes) > 1:
            if time.time() > deadline:
                break
            routes = _relocate(routes, instance, dist)
            if time.time() > deadline:
                break
            routes = _swap_nodes(routes, instance, dist)

    return routes
=======
def _ga_crossover(routes_a, routes_b, instance, dist):
    """Order-preserving crossover: merge routes preserving relative order of
    time-window-compatible subsequences. Returns a new set of routes."""
    n_customers = instance["n_customers"]
    all_customers = set(range(1, n_customers + 1))
    stations = set(instance["stations"])
    
    # Flatten all customers from both parents
    seq_a = [v for r in routes_a for v in r if v in all_customers]
    seq_b = [v for r in routes_b for v in r if v in all_customers]
    
    if not seq_a or not seq_b:
        return None
    
    # Order-preserving crossover: pick random cut points
    n = len(seq_a)
    if n < 2:
        return None
    
    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(0, n - 1)
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    # Segment from parent A
    segment = seq_a[cut1:cut2 + 1]
    segment_set = set(segment)
    
    # Fill remaining positions from parent B in order
    fill = [v for v in seq_b if v not in segment_set]
    
    # Build new sequence
    new_seq = []
    fill_idx = 0
    for i in range(n):
        if cut1 <= i <= cut2:
            new_seq.append(segment[i - cut1])
        else:
            if fill_idx < len(fill):
                new_seq.append(fill[fill_idx])
                fill_idx += 1
    
    # If we have extra or missing, pad/trim
    if len(new_seq) < n:
        for v in seq_b:
            if v not in set(new_seq):
                new_seq.append(v)
    elif len(new_seq) > n:
        new_seq = new_seq[:n]
    
    # Check all customers present exactly once
    if set(new_seq) != all_customers:
        return None
    
    # Split into routes using greedy insertion by proximity
    # Group customers into routes based on spatial clustering
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    
    # Simple spatial split: sort by angle from depot
    angles = []
    for c in new_seq:
        x, y = coords[c]
        angle = math.atan2(y, x)
        angles.append((angle, c))
    angles.sort()
    
    # Try different numbers of routes
    best_routes = None
    best_dist = float('inf')
    
    for n_routes in [2, 3, 4, 5]:
        # Assign customers to routes by angular sectors
        sector_size = 2 * math.pi / n_routes
        route_assignments = [[] for _ in range(n_routes)]
        
        for angle, c in angles:
            sector = int((angle + math.pi) / sector_size) % n_routes
            route_assignments[sector].append(c)
        
        # Build routes and check feasibility
        valid = True
        total_d = 0.0
        new_routes = []
        
        for ra in route_assignments:
            if not ra:
                continue
            # Check capacity
            load = sum(demands[c] for c in ra)
            if load > capacity:
                valid = False
                break
            # Simulate
            d, feas = _simulate_route(ra, instance, dist)
            if not feas:
                valid = False
                break
            total_d += d
            new_routes.append(ra)
        
        if valid and total_d < best_dist:
            best_dist = total_d
            best_routes = new_routes
    
    return best_routes


def _ga_local_search(routes, instance, dist, deadline):
    """Genetic algorithm with order-preserving crossover + local search polish."""
    start_time = time.time()
    population = [list(r) for r in routes]
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, instance, dist)
    
    pop_size = 6
    # Fill population with perturbed copies
    while len(population) < pop_size:
        base = random.choice(population)
        perturbed = [list(r) for r in base]
        # Random perturbation: swap two customers in a random route
        if perturbed and len(perturbed[0]) > 2:
            idx = random.randint(0, len(perturbed) - 1)
            if len(perturbed[idx]) > 2:
                i = random.randint(0, len(perturbed[idx]) - 2)
                j = random.randint(i + 1, len(perturbed[idx]) - 1)
                perturbed[idx][i], perturbed[idx][j] = perturbed[idx][j], perturbed[idx][i]
        population.append(perturbed)
    
    max_generations = 30
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Evaluate population
        for i in range(len(population)):
            d = _total_distance(population[i], instance, dist)
            if d < best_dist and _validate(population[i], instance, dist):
                best_dist = d
                best_routes = [list(r) for r in population[i]]
        
        # Selection: keep top half
        scored = []
        for i in range(len(population)):
            d = _total_distance(population[i], instance, dist)
            scored.append((d, i))
        scored.sort()
        
        survivors = [population[idx] for _, idx in scored[:pop_size // 2]]
        
        # Crossover to fill rest
        offspring = []
        while len(survivors) + len(offspring) < pop_size:
            if time.time() > deadline:
                break
            pa = random.choice(survivors) if survivors else random.choice(population)
            pb = random.choice(survivors) if survivors else random.choice(population)
            child = _ga_crossover(pa, pb, instance, dist)
            if child is None:
                # Fallback: use parent
                child = [list(r) for r in random.choice([pa, pb])]
            offspring.append(child)
        
        population = survivors + offspring
        
        # Local search on best
        if time.time() < deadline - 10:
            polished = _two_opt_routes_polish(best_routes, instance, dist, deadline - 5)
            d = _total_distance(polished, instance, dist)
            if d < best_dist and _validate(polished, instance, dist):
                best_dist = d
                best_routes = polished
                population[0] = [list(r) for r in best_routes]
    
    return best_routes


def _two_opt_routes_polish(routes, instance, dist, deadline):
    """High-precision polish: multiple passes of 2-opt and or-opt with tight tolerance."""
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i] = _two_opt(routes[i], instance, dist, max_passes=5)
        if time.time() > deadline:
            break
        routes[i] = _or_opt(routes[i], instance, dist, max_passes=5)
    
    # Final inter-route relocate
    if len(routes) > 1 and time.time() < deadline:
        routes = _relocate(routes, instance, dist)
    
    return routes


def _local_search(routes, instance, dist, deadline):
    """Full local search with deadline."""
    start_time = time.time()
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # 2-opt on each route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _two_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Or-opt
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _or_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Inter-route moves
        if len(routes) > 1:
            if time.time() > deadline:
                break
            routes = _relocate(routes, instance, dist)
            if time.time() > deadline:
                break
            routes = _swap_nodes(routes, instance, dist)

    return routes
>>>>>>> REPLACE

