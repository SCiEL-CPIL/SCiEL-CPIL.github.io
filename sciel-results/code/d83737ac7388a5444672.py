def _solve_instance(instance, time_limit=25):
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    construct_time = min(5, time_limit * 0.2)
    routes = _construct_initial_routes(instance, dist, construct_time)
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
        
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, ls_time)
    
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
            
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes
