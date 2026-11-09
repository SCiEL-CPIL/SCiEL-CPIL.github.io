"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, occupied, i, new_pos):
    """
    Calculate the change in energy if residue i moves from chain[i] to new_pos.
    This is an approximation used in SA moves where we assume only local changes.
    For tail regrowth, we recalculate the tail energy or use a full recalc if needed.
    For simplicity and correctness in the SA loop, we might just recalc energy for the whole chain
    if the move is accepted, or use a faster delta for local moves.
    Given N is small (<140), full recalc is O(N^2) ~ 20k ops, which is fast.
    """
    pass # We will implement specific move logic instead

def _grow_random(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        # Random choice, or slightly biased towards H contacts? 
        # For initialization, random is fine.
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    return chain

def _tail_regrowth(sequence, chain, occupied, k, rng):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or None if stuck.
    """
    if k >= len(chain):
        return chain, occupied
    
    # Remove tail from k to end
    old_tail = chain[k:]
    new_occupied = set(occupied)
    for p in old_tail:
        new_occupied.discard(p)
    
    new_chain = chain[:k]
    
    for i in range(k, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in new_occupied:
                options.append(site)
        
        if not options:
            # Backtracking is complex here. If stuck, this regrowth fails.
            return None, None
        
        # Heuristic: prefer sites that create H-H contacts
        best_site = None
        best_score = -1
        for site in options:
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in new_occupied:
                        # Check if the residue at q is H and is before k (part of the fixed head)
                        # We need to know which indices are in new_occupied.
                        # Since we only removed the tail, all occupied sites in new_occupied 
                        # that are neighbors to site must be from the head (indices < k).
                        # We don't have the index mapping in new_occupied easily.
                        # Let's just count neighbors in new_occupied.
                        score += 1
            score += rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_site = site
        
        new_chain.append(best_site)
        new_occupied.add(best_site)
        
    return new_chain, new_occupied

def _sa_fold(sequence, dimension, rng, time_limit):
    """
    Simulated Annealing for a single sequence.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]
        
    # Initialize
    best_chain = None
    best_energy = float('inf')
    
    # Try to get a valid initial chain
    init_chain = _grow_random(sequence, dimension, rng)
    if init_chain is None:
        # Fallback to a simple line if random growth fails (unlikely for small N)
        init_chain = [(i, 0, 0) for i in range(n)]
        
    current_chain = init_chain
    current_occupied = set(current_chain)
    current_energy = _energy(sequence, current_chain)
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_chain = list(current_chain)
        
    start_time = time.time()
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.99
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Tail Regrowth
            # Pick a random k, biased towards the end but allowing deeper cuts
            k = rng.randint(max(1, n // 2), n - 1)
            new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng)
            if new_chain is None:
                T *= cooling_rate
                continue
                
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
        else:
            # Local Move: Try to move a single residue (or small segment)
            # This is harder to implement correctly with self-avoidance.
            # For now, we rely heavily on tail regrowth which is robust.
            # We can add a simple "kick" if we want, but tail regrowth is usually sufficient.
            pass
            
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation of a fixed number of randomized greedy growths.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    # Total time budget is 330s for 26 sequences.
    # Approx 12s per sequence.
    # We will run multiple SA restarts.
    
    # Estimate time per sequence
    # Let's allocate 10 seconds per sequence to be safe.
    seq_time_limit = 10.0
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts depends on time.
    # Let's do 5 restarts of SA, each with a time slice.
    # Or just one long SA run? Multiple restarts with different inits is better.
    
    rng = random.Random(42 + len(sequence))
    
    for restart in range(5):
        # Time slice for this restart
        slice_time = seq_time_limit / 5.0
        
        # Generate a new initial chain
        init_chain = _grow_random(sequence, dimension, rng)
        if init_chain is None:
            init_chain = [(i, 0, 0) for i in range(n)]
            
        # Run SA
        sa_chain = _sa_fold(sequence, dimension, rng, slice_time)
        
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain
                
        # Check global time just in case
        if time.time() > 300: # Safety margin
             break
             
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
