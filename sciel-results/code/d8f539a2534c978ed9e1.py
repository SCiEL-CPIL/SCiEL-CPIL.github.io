<<<<<<< SEARCH
def _try_all_configs():
    """Main search: evaluate initial configs, then local search on top ones."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        # Fallback
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    best_areas = scored_configs[0][1]
    
    # Run local search on top configurations with more time allocation
    top_k = min(6, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 8.0:
            break
        score, areas, config = scored_configs[i]
        target_area = max(areas, key=areas.get) if areas else None
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        try:
            lp, ls = _local_search(config, time.time() + remaining, target_area=target_area)
            if ls > best_score:
                best_score = ls
                best_points = lp
                _, best_areas = _score_fast(lp)
        except Exception:
            pass
    
    # Polish the best result
    if time.time() < deadline - 3.0:
        try:
            pp, ps = _polish(best_points, deadline - 1.0)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    return best_points
=======
def _try_all_configs():
    """Main search: evaluate initial configs, then local search on top ones."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 20.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        # Fallback
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    best_areas = scored_configs[0][1]
    
    # Run local search on top configurations
    top_k = min(8, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 10.0:
            break
        score, areas, config = scored_configs[i]
        target_area = max(areas, key=areas.get) if areas else None
        remaining = deadline - time.time() - 6.0
        if remaining < 2.0:
            break
        try:
            lp, ls = _local_search(config, time.time() + remaining, target_area=target_area)
            if ls > best_score:
                best_score = ls
                best_points = lp
                _, best_areas = _score_fast(lp)
        except Exception:
            pass
    
    # Polish the best result
    if time.time() < deadline - 4.0:
        try:
            pp, ps = _polish(best_points, deadline - 1.0)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    # Second polish pass if time allows
    if time.time() < deadline - 2.0:
        try:
            pp, ps = _polish(best_points, deadline - 0.5)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    return best_points
>>>>>>> REPLACE
