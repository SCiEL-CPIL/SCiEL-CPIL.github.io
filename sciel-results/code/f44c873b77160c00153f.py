<<<<<<< SEARCH
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices as safety
            results.append(set(range(n)))
            continue
        
        selected = solve_instance(n, p, seed)
        results.append(selected)
    
    return [sorted(r) for r in results]
=======
    results = []
    # Allocate time proportionally to instance size, with more time for harder instances
    total_time_budget = 140  # leave 10s safety margin
    time_per_instance = total_time_budget / 3
    
    for i, (n, p, seed) in enumerate((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices as safety
            results.append(set(range(n)))
            continue
        
        # Set deadline for this instance
        instance_deadline = start_time + (i + 1) * time_per_instance
        instance_deadline = min(instance_deadline, deadline)
        
        selected = solve_instance(n, p, seed, instance_deadline)
        results.append(selected)
    
    return [sorted(r) for r in results]
>>>>>>> REPLACE
