def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 110
    
    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = float('inf')
    
    n = len(sequence)
    
    # 1. Generate initial candidates
    # Run greedy until we have a good seed or time runs out
    # We reserve 80% of time for SA
    greedy_deadline = start_time + (deadline - start_time) * 0.2
    
    while time.time() < greedy_deadline:
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            if best_chain is None:
                best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
                best_energy = _energy(sequence, best_chain)
            continue
            
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
            
        # Early stop if energy is very good
        if best_energy <= -n // 2:
            break
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    # 2. Refine with SA
    if time.time() < deadline - 2:
        time_left = deadline - 2 - time.time()
        if time_left > 0:
             refined_chain, refined_energy = _sa_search(sequence, best_chain, time_left, rng)
             if refined_energy < best_energy:
                 best_chain = refined_chain
                 best_energy = refined_energy
                 
    return [list(p) for p in best_chain]
