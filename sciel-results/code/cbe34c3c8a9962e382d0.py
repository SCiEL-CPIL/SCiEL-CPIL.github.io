def _perturb(comps, rng, k=5):
    """Apply k random mutations."""
    comps = list(comps)
    for _ in range(k):
        comps = _mutate(comps, rng)
    return comps

def construct():
    _init_constants()
    
    # 90s limit, keep 5s safety margin
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    best = _seed_network()
    best_size = len(best)
    
    # Initial Pruning
    best = _prune(best, deadline)
    best_size = len(best)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any comparator
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                best_size -= 1
                improved = True
                break # Restart scan from end
        
        if not improved:
            # 2. Try random mutations to escape local minima
            # Use batches of mutations to jump over barriers
            for _ in range(10): # 10 batches
                if time.time() > deadline:
                    break
                # Perturb by 5 random mutations
                mutated = _perturb(best, rng, 5)
                if _verify_network_bitparallel(mutated):
                    pruned = _prune(mutated, deadline)
                    if len(pruned) < best_size:
                        best = pruned
                        best_size = len(best)
                        improved = True
                        break
    return best
