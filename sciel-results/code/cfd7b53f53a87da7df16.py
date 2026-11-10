<<<<<<< SEARCH
def construct(instance):
    """
    Find low-energy Dzugutov clusters using polytetrahedral seeds and basin hopping.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919)  # deterministic seed per n
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_generate_polytetrahedral_seed(n, rng))
    seeds.append(_generate_fcc_seed(n, rng))
    seeds.append(_generate_random_compact(n, rng))
    for _ in range(2):
        seeds.append(_generate_random_compact(n, rng))
    
    # Relax all seeds in parallel
    results = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_and_record, seed, potential, results, lock) for seed in seeds]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    
    for energy, positions in results:
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Parallel basin hopping from top seeds
    time_remaining = _get_time_remaining()
    if time_remaining > 60 and best_positions is not None:
        # Sort results by energy and take top 3 for basin hopping
        results_sorted = sorted(results, key=lambda x: x[0])
        bh_seeds = [pos for _, pos in results_sorted[:3]]
        
        # Determine iterations per chain based on time
        time_per_chain = time_remaining / 4  # 4 chains total (3 BH + 1 reserve)
        max_iters_per_chain = max(10, min(60, int(time_per_chain / 2.5)))
        
        bh_results = []
        bh_lock = threading.Lock()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, seed in enumerate(bh_seeds):
                chain_rng = np.random.default_rng(n * 7919 + i * 1000)
                futures.append(executor.submit(
                    _parallel_basin_hopping, seed, potential, chain_rng, 
                    max_iters_per_chain, 0.35, bh_results, bh_lock
                ))
            for f in concurrent.futures.as_completed(futures):
                f.result()
        
        for energy, positions in bh_results:
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Final high-precision polish: gradient-based optimization
    if best_positions is not None and _get_time_remaining() > 20:
        # Try gradient-based refinement
        def _energy_func(pos):
            flat = pos.flatten()
            e, _ = relax(flat.reshape(-1, 3), potential)
            return e
        
        def _gradient_func(pos):
            flat = pos.flatten()
            _, grad = dzugutov_energy_gradient(flat.reshape(-1, 3))
            return grad.flatten()
        
        # Multiple gradient-based refinements from perturbed best
        for trial in range(3):
            if _get_time_remaining() < 15:
                break
            if trial == 0:
                start_pos = best_positions.copy()
            else:
                start_pos = best_positions + 0.03 * rng.normal(0, 1, best_positions.shape)
            
            try:
                res = minimize(
                    _energy_func, start_pos.flatten(), 
                    jac=_gradient_func, method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success and res.fun < best_energy:
                    best_energy = res.fun
                    best_positions = res.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final relaxation
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: return a simple linear arrangement
        best_positions = np.array([[i * 1.1, 0, 0] for i in range(n)])
    
    return best_positions
=======
def construct(instance):
    """
    Find low-energy Dzugutov clusters using graph-based Voronoi topology optimization.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919)  # deterministic seed per n
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_generate_polytetrahedral_seed(n, rng))
    seeds.append(_generate_fcc_seed(n, rng))
    seeds.append(_generate_random_compact(n, rng))
    for _ in range(2):
        seeds.append(_generate_random_compact(n, rng))
    
    # Relax all seeds in parallel
    results = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_and_record, seed, potential, results, lock) for seed in seeds]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    
    for energy, positions in results:
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Parallel graph basin hopping from top seeds
    time_remaining = _get_time_remaining()
    if time_remaining > 60 and best_positions is not None:
        # Sort results by energy and take top 3 for graph basin hopping
        results_sorted = sorted(results, key=lambda x: x[0])
        bh_seeds = [pos for _, pos in results_sorted[:3]]
        
        # Determine iterations per chain based on time
        time_per_chain = time_remaining / 4  # 4 chains total (3 BH + 1 reserve)
        max_iters_per_chain = max(8, min(40, int(time_per_chain / 3.0)))
        
        bh_results = []
        bh_lock = threading.Lock()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, seed in enumerate(bh_seeds):
                chain_rng = np.random.default_rng(n * 7919 + i * 1000)
                futures.append(executor.submit(
                    _parallel_graph_basin_hopping, seed, potential, chain_rng, 
                    max_iters_per_chain, 0.2, bh_results, bh_lock
                ))
            for f in concurrent.futures.as_completed(futures):
                f.result()
        
        for energy, positions in bh_results:
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Final high-precision polish: gradient-based optimization
    if best_positions is not None and _get_time_remaining() > 20:
        # Try gradient-based refinement
        def _energy_func(pos):
            flat = pos.flatten()
            e, _ = relax(flat.reshape(-1, 3), potential)
            return e
        
        def _gradient_func(pos):
            flat = pos.flatten()
            _, grad = dzugutov_energy_gradient(flat.reshape(-1, 3))
            return grad.flatten()
        
        # Multiple gradient-based refinements from perturbed best
        for trial in range(3):
            if _get_time_remaining() < 15:
                break
            if trial == 0:
                start_pos = best_positions.copy()
            else:
                start_pos = best_positions + 0.03 * rng.normal(0, 1, best_positions.shape)
            
            try:
                res = minimize(
                    _energy_func, start_pos.flatten(), 
                    jac=_gradient_func, method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success and res.fun < best_energy:
                    best_energy = res.fun
                    best_positions = res.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final relaxation
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: return a simple linear arrangement
        best_positions = np.array([[i * 1.1, 0, 0] for i in range(n)])
    
    return best_positions
>>>>>>> REPLACE

