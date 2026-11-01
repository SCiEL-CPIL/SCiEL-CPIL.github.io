<<<<<<< SEARCH
def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_positions = None
    best_r = 0.0
    
    # Strategy 1: Ring-based init + L-BFGS-B
    if time.time() < deadline - 70.0:
        pos1 = _ring_init(N, center=True)
        pos1, r1 = _lbfgs_optimize(pos1, max_time=55.0)
        if r1 > best_r:
            best_r = r1
            best_positions = pos1.copy()
    
    # Strategy 2: Hex lattice init + L-BFGS-B
    if time.time() < deadline - 70.0:
        pos2 = _hex_init(N)
        pos2, r2 = _lbfgs_optimize(pos2, max_time=55.0)
        if r2 > best_r:
            best_r = r2
            best_positions = pos2.copy()
    
    # Strategy 3: Random init + L-BFGS-B
    if time.time() < deadline - 70.0:
        np.random.seed(42)
        pos3 = _random_init(N)
        pos3, r3 = _lbfgs_optimize(pos3, max_time=55.0)
        if r3 > best_r:
            best_r = r3
            best_positions = pos3.copy()
    
    # Local polish on best
    if best_positions is not None and time.time() < deadline - 10.0:
        best_positions, best_r = _local_polish(best_positions, best_r, max_time=25.0)
    
    # Final validation
    if best_positions is None:
        best_positions = _ring_init(N, center=True)
        best_r = _score(best_positions)
    
    radii = np.hypot(best_positions[:, 0], best_positions[:, 1])
    max_allowed = 1.0 - best_r * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        best_positions[mask] *= max_allowed / radii[mask]
    
    return best_positions
=======
def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_positions = None
    best_r = 0.0
    
    # Strategy 1: Ring-based init + Simulated Annealing
    if time.time() < deadline - 60.0:
        pos1 = _ring_init(N, center=True)
        pos1, r1 = _simulated_anneal(pos1, max_time=45.0)
        if r1 > best_r:
            best_r = r1
            best_positions = pos1.copy()
    
    # Strategy 2: Hex lattice init + Simulated Annealing
    if time.time() < deadline - 60.0:
        pos2 = _hex_init(N)
        pos2, r2 = _simulated_anneal(pos2, max_time=45.0)
        if r2 > best_r:
            best_r = r2
            best_positions = pos2.copy()
    
    # Strategy 3: Random init + Simulated Annealing
    if time.time() < deadline - 60.0:
        np.random.seed(42)
        pos3 = _random_init(N)
        pos3, r3 = _simulated_anneal(pos3, max_time=45.0)
        if r3 > best_r:
            best_r = r3
            best_positions = pos3.copy()
            
    # Strategy 4: Refine best found with local polish
    if best_positions is not None and time.time() < deadline - 10.0:
        best_positions, best_r = _local_polish(best_positions, best_r, max_time=25.0)
    
    # Final validation
    if best_positions is None:
        best_positions = _ring_init(N, center=True)
        best_r = _score_fast(best_positions)
    
    radii = np.hypot(best_positions[:, 0], best_positions[:, 1])
    max_allowed = 1.0 - best_r * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        best_positions[mask] *= max_allowed / radii[mask]
    
    return best_positions
>>>>>>> REPLACE
