"""HP lattice protein folding using greedy growth + simulated annealing local search."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_occupied(chain):
    """Build dict mapping lattice point -> index for O(1) lookup."""
    return {tuple(p): i for i, p in enumerate(chain)}


def _energy_from_contacts(sequence, chain, where):
    """Compute energy given chain and where mapping."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_pivot(chain, i, direction, sequence):
    """Try rotating a segment of the chain. Returns new chain if valid, else None.
    direction: 0=left turn, 1=right turn relative to current heading."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    # Get the current direction into position i
    prev = chain[i - 1]
    curr = chain[i]
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    
    # Rotate the direction by 90 degrees
    if direction == 0:  # left turn: (dx,dy) -> (-dy, dx)
        new_dx, new_dy = -dy, dx
    else:  # right turn: (dx,dy) -> (dy, -dx)
        new_dx, new_dy = dy, -dx
    
    new_curr = (curr[0] + new_dx, curr[1] + new_dy)
    
    # Check if new_curr is valid (not occupied by non-adjacent part)
    where = {tuple(p): j for j, p in enumerate(chain)}
    if new_curr in where:
        j = where[new_curr]
        if abs(j - i) > 1:
            return None
    
    # Check the rest of the chain from i+1 onwards can fit
    # We need to rotate the entire tail
    new_chain = chain[:i] + [new_curr]
    new_occupied = {tuple(p): k for k, p in enumerate(new_chain)}
    
    # Propagate the rotation to the rest of the chain
    for k in range(i + 1, n):
        prev_pt = new_chain[k - 1]
        orig_prev = chain[k - 1]
        orig_curr = chain[k]
        odx = orig_curr[0] - orig_prev[0]
        ody = orig_curr[1] - orig_prev[1]
        
        # Rotate this step the same way
        if direction == 0:
            ndx, ndy = -ody, odx
        else:
            ndx, ndy = ody, -odx
        
        new_pt = (prev_pt[0] + ndx, prev_pt[1] + ndy)
        
        if new_pt in new_occupied:
            j = new_occupied[new_pt]
            if abs(j - k) > 1:
                return None
        new_occupied[new_pt] = k
        new_chain.append(new_pt)
    
    return new_chain


def _try_reverse(chain, i, j, sequence):
    """Try reversing a segment of the chain from i to j. Returns new chain if valid, else None."""
    n = len(chain)
    if i < 1 or j > n - 2 or i >= j:
        return None
    
    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
    
    # Check self-avoidance
    where = {}
    for k, p in enumerate(new_chain):
        tp = tuple(p)
        if tp in where:
            if abs(where[tp] - k) > 1:
                return None
        where[tp] = k
    
    return new_chain


def _local_search(sequence, chain, max_iters=5000, time_limit=None):
    """Simulated annealing local search to improve a conformation."""
    n = len(chain)
    where = {tuple(p): i for i, p in enumerate(chain)}
    current_energy = _energy_from_contacts(sequence, chain, where)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 10.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Pivot move
            i = random.randint(1, n - 2)
            direction = random.randint(0, 1)
            new_chain = _try_pivot(chain, i, direction, sequence)
        else:
            # Reversal move
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2)
            new_chain = _try_reverse(chain, i, j, sequence)
        
        if new_chain is None:
            continue
        
        new_where = {tuple(p): k for k, p in enumerate(new_chain)}
        new_energy = _energy_from_contacts(sequence, new_chain, new_where)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    time_budget = 180.0  # seconds total for all sequences, but we get 180s per call
    
    rng = random.Random(12345 + len(sequence))
    
    # Generate multiple initial conformations via greedy growth
    initial_chains = []
    for _ in range(30):
        if time.time() - start_time > time_budget * 0.3:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
    
    # Find the best initial conformation
    best_chain = None
    best_energy = 1
    for chain in initial_chains:
        where = {tuple(p): i for i, p in enumerate(chain)}
        e = _energy_from_contacts(sequence, chain, where)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    # Apply local search to the best initial conformation
    remaining_time = time_budget * 0.7 - (time.time() - start_time)
    if remaining_time > 1:
        best_chain, best_energy = _local_search(sequence, best_chain, 
                                                  max_iters=10000, 
                                                  time_limit=remaining_time)
    
    # Also try local search on a few other initial conformations if time permits
    for chain in initial_chains[1:4]:
        if time.time() - start_time > time_budget * 0.9:
            break
        where = {tuple(p): i for i, p in enumerate(chain)}
        e = _energy_from_contacts(sequence, chain, where)
        if e >= best_energy:
            continue
        remaining_time = time_budget * 0.9 - (time.time() - start_time)
        if remaining_time > 0.5:
            improved_chain, improved_energy = _local_search(sequence, chain,
                                                             max_iters=3000,
                                                             time_limit=remaining_time)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
    
    # Final check: ensure self-avoidance
    seen = set()
    for p in best_chain:
        tp = tuple(p)
        if tp in seen:
            # Should not happen, but fallback to straight line
            return [[i, 0] for i in range(len(sequence))]
        seen.add(tp)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
