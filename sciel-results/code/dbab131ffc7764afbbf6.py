"""HP lattice protein folding using optimized Simulated Annealing with incremental energy."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return max(0.0, DEADLINE - (time.time() - _START_TIME))

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Returns delta_e (new_energy - old_energy).
    """
    delta = 0
    # Remove contacts involving idx at old_site
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1  # Removing a contact increases energy (less negative)
    # Add contacts involving idx at new_site
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta -= 1  # Adding a contact decreases energy (more negative)
    return delta

def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move (pivot). 
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        # Must be adjacent to prev and next
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        # Must be empty
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    # Compute energy delta incrementally
    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        return chain, where, current_energy + delta_e, True
    else:
        return chain, where, current_energy, False

def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal (flip).
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # Check boundary connectivity
    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    old_segment = list(chain[i:j + 1])
    new_segment = old_segment[::-1]
    
    # Check self-avoidance for the reversed segment
    # We need to ensure no point in new_segment (except boundaries which are handled by connectivity) 
    # collides with the rest of the chain or itself.
    # Actually, since we reverse, the set of points is the same. We just need to check if the 
    # reversed segment collides with the parts of the chain *outside* the segment [i, j].
    
    # Remove old segment points from a temporary where map
    temp_where = {k: v for k, v in where.items() if k not in old_segment}
    
    valid = True
    for pt in new_segment:
        if pt in temp_where:
            valid = False
            break
    
    if not valid:
        return chain, where, current_energy, False

    # Apply move
    chain[i:j + 1] = new_segment
    for k in range(i, j + 1):
        where[chain[k]] = k

    # Recalculate energy fully for flip as incremental is complex for multi-residue moves
    # unless we implement a sophisticated delta, but O(N) for flip is acceptable if flips are rare
    # or if N is small. For N=100, O(N) is fast.
    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        chain[i:j + 1] = old_segment
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False

def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing with optimized moves."""
    start_time = time.time()
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = max(0.5, -current_energy * 0.1)
    T_min = 0.01
    cooling_rate = 0.999  # Slower cooling to allow more moves
    
    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 5:
        # Choose move type: 80% Pivot, 20% Flip
        if rng.random() < 0.8:
            chain, where, current_energy, _ = _local_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, _ = _reverse_move(sequence, chain, where, current_energy, T, rng)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Generate multiple initial chains
    initial_chains = []
    # Try different biases for diversity
    biases = [0.5, 1.0, 1.5, 2.0, 0.1, 3.0]
    for bias in biases:
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    # Add a few random growths if we have time
    for _ in range(4):
        chain = _grow(sequence, dimension, rng, bias=rng.uniform(0.1, 3.0))
        if chain is not None:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback to a straight line if growth failed (unlikely)
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Sort by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    best_chain = initial_chains[0]
    best_energy = _energy(sequence, best_chain)
    
    # Allocate time
    # Reserve 5 seconds for safety
    total_time_budget = max(1.0, _remaining_time() - 5.0)
    
    # We will run SA on the top few initial chains
    # Give more time to better initial chains
    num_runs = min(len(initial_chains), 6)
    
    for i in range(num_runs):
        if _remaining_time() < 5:
            break
        
        # Time allocation: exponential decay
        # First chain gets ~40%, second ~20%, etc.
        # Simplified: divide remaining budget roughly equally but prioritize early
        time_for_this = total_time_budget / (num_runs - i) * 0.8
        time_for_this = max(0.5, min(time_for_this, _remaining_time() - 5.0))
        
        # Deep copy chain to avoid mutation issues across runs
        current_chain = list(initial_chains[i])
        
        improved_chain, improved_energy = _anneal(sequence, current_chain, dimension, rng, time_for_this)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
