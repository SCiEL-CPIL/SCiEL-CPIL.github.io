"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP lattice energy: -1 for each non-consecutive H-H contact."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    # Count each pair once
                    if i < j:
                        contacts += 1
    return -contacts


def _valid_chain(chain):
    """Check if a chain is self-avoiding and has unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _local_search(sequence, dimension, chain, time_budget, rng):
    """
    Perform local search using simulated annealing with kink moves.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    # Convert to list of lists for easier mutation
    current = [list(p) for p in chain]
    current_energy = _energy(sequence, tuple(tuple(p) for p in current))
    
    best = [list(p) for p in current]
    best_energy = current_energy
    
    # Precompute which residues are H
    is_h = [res == 'H' for res in sequence]
    
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    # Number of iterations - we'll estimate based on time budget
    # For now, use a fixed number that should be fast enough
    # We can tune this later
    max_iters = 50000
    
    for it in range(max_iters):
        # Annealing schedule
        alpha = it / max_iters
        temp = t_start * math.exp(-10 * alpha) + t_end
        
        # Choose a random residue index (not endpoints)
        i = rng.randint(1, n - 2)
        
        # Save original position
        orig = list(current[i])
        
        # Try moving residue i to a neighboring site
        # But we must ensure the chain remains valid (self-avoiding, unit steps)
        # The move must preserve unit steps with neighbors i-1 and i+1
        
        # Get neighbors in the chain
        prev_pos = current[i - 1]
        next_pos = current[i + 1]
        
        # The new position must be adjacent to both prev_pos and next_pos
        # Actually, that's too restrictive. In a lattice walk, if we move residue i,
        # it must remain adjacent to i-1 and i+1.
        
        # Find all sites adjacent to both prev_pos and next_pos
        # But also, the site must not be occupied by any other residue (except i-1 and i+1)
        
        # First, build the set of occupied sites
        occupied = set()
        for j, p in enumerate(current):
            if j != i:
                occupied.add(tuple(p))
        
        # Find candidate sites: adjacent to prev_pos AND adjacent to next_pos
        candidates = []
        for site in _neighbours(tuple(prev_pos)):
            if site in _neighbours(tuple(next_pos)):
                # Check if site is free (not occupied by any other residue)
                if site not in occupied:
                    candidates.append(list(site))
        
        if not candidates:
            continue
        
        # Pick a random candidate
        new_pos = rng.choice(candidates)
        
        # Check if this is actually a different position
        if new_pos == orig:
            continue
        
        # Apply the move
        current[i] = new_pos
        
        # Compute new energy
        new_energy = _energy(sequence, tuple(tuple(p) for p in current))
        
        # Accept or reject
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = [list(p) for p in current]
        else:
            # Revert
            current[i] = orig
    
    return best


def _grow_initial(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using a simple greedy/random approach."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Start at origin
    chain = [[0] * dimension]
    occupied = {tuple(chain[0]): 0}
    
    for i in range(1, n):
        # Find free neighbors of the last residue
        last = chain[-1]
        options = []
        for site in _neighbours(tuple(last)):
            if site not in occupied:
                options.append(list(site))
        
        if not options:
            # Dead end - try to backtrack or just fail
            # For simplicity, return None to indicate failure
            return None
        
        # Choose randomly
        site = rng.choice(options)
        chain.append(site)
        occupied[tuple(site)] = i
    
    return chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    if n == 2:
        return [[0] * dimension, [1] + [0] * (dimension - 1)]
    
    rng = random.Random(12345 + len(sequence))
    
    # Generate multiple initial conformations and apply local search to each
    best_chain = None
    best_energy = float('inf')
    
    num_trials = 10
    
    for trial in range(num_trials):
        # Generate initial chain
        chain = _grow_initial(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search
        improved = _local_search(sequence, dimension, chain, 10.0, rng)
        
        # Compute energy
        energy = _energy(sequence, tuple(tuple(p) for p in improved))
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
    
    # If no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
