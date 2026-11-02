    # Try a few seeds
    for seed in [42, 123, 999, 314, 271]:
        if time.time() - start_time > time_budget:
            break
        
        result = _random_search_223_rank10(seed=seed, iterations=50)
