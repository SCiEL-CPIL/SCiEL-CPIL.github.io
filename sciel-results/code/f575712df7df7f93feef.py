<<<<<<< SEARCH
def _local_search2(sequence, chain, rng, max_iter=3000):
    """
    More aggressive local search with multiple pivot positions per step.
    """
    n = len(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 0.5
    T_min = 0.05
    cooling = 0.998
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Try several random pivots
        found_improvement = False
        for _ in range(3):
            i = rng.randint(1, n - 1)
            new_chain = _pivot_move(chain, i, rng)
            if new_chain is None:
                continue
            
            new_energy = _calculate_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                found_improvement = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain
=======
def _local_search2(sequence, chain, rng, max_iter=800):
    """
    More aggressive local search with multiple pivot positions per step.
    """
    n = len(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 0.3
    T_min = 0.03
    cooling = 0.995
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        # Try several random pivots
        for _ in range(2):
            i = rng.randint(1, n - 1)
            new_chain = _pivot_move(chain, i, rng)
            if new_chain is None:
                continue
            
            new_energy = _calculate_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain
>>>>>>> REPLACE

