<<<<<<< SEARCH
def _polish(sequence, chain, time_limit, rng):
    """
    Deterministic hill-climbing polish: accept only improving moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    n = len(current_chain)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit and _time_left() > 15:
        improved = False
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        # Also try slither
        if not improved:
            new_chain = _apply_slither(current_chain, rng)
            if new_chain:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    
    return current_chain
=======
def _polish(sequence, chain, time_limit, rng):
    """
    Deterministic hill-climbing polish: accept only improving moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    n = len(current_chain)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit and _time_left() > 15:
        improved = False
        # Try pivot moves
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        # Try crankshaft moves
        if not improved:
            for pivot_idx in range(1, n - 2):
                new_chain = _apply_crankshaft(current_chain, pivot_idx)
                if new_chain:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
        # Try slither
        if not improved:
            new_chain = _apply_slither(current_chain, rng)
            if new_chain:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    
    return current_chain
>>>>>>> REPLACE

Now I'll update the fold function to allocate more time for search and try more initial candidates:

