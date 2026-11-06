<<<<<<< SEARCH
def _local_search(sequence, chain, rng, max_steps=1000):
    """
    Local search using single-residue moves.
    Moves a residue i to a new position that maintains connectivity with i-1 and i+1.
    Accepts moves that improve energy (lower value).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for _ in range(max_steps):
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid positions for the new location of residue i
        # It must be a neighbor of prev AND a neighbor of next_
        # And it must not be the current position
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        if not candidates:
            continue
            
        # Pick a random candidate from the valid positions
        new_pos = rng.choice(list(candidates))
        
        # Check if the new position is occupied by another residue
        if new_pos in occupied:
            continue
            
        # Perform the move
        del occupied[curr]
        occupied[new_pos] = i
        chain[i] = new_pos
        
        # Calculate new energy
        new_energy = _energy(sequence, chain)
        
        # Accept if better (more negative), otherwise revert
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Revert
            del occupied[new_pos]
            occupied[curr] = i
            chain[i] = curr
    
    return chain, current_energy
=======
def _local_search(sequence, chain, rng, max_steps=1000, time_limit=None):
    """
    Local search using single-residue moves.
    Moves a residue i to a new position that maintains connectivity with i-1 and i+1.
    Accepts moves that improve energy (lower value).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    steps_done = 0
    for _ in range(max_steps):
        if time_limit is not None and _time_remaining() < time_limit:
            break
        steps_done += 1
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid positions for the new location of residue i
        # It must be a neighbor of prev AND a neighbor of next_
        # And it must not be the current position
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        if not candidates:
            continue
            
        # Pick a random candidate from the valid positions
        new_pos = rng.choice(list(candidates))
        
        # Check if the new position is occupied by another residue
        if new_pos in occupied:
            continue
            
        # Perform the move
        del occupied[curr]
        occupied[new_pos] = i
        chain[i] = new_pos
        
        # Calculate new energy
        new_energy = _energy(sequence, chain)
        
        # Accept if better (more negative), otherwise revert
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Revert
            del occupied[new_pos]
            occupied[curr] = i
            chain[i] = curr
    
    return chain, current_energy
>>>>>>> REPLACE

