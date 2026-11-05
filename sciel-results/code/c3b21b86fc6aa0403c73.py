"""
HP lattice protein folding using Greedy Local Search with Restarts.
"""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    """Generate 4-neighbors of a point on the square lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk (SAW) using a greedy method.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((rng.random() + 0.1 * free, site))
        
        if not options:
            return None
        
        best_score, site = max(options)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _calculate_energy(sequence, chain):
    """
    Calculate the energy of a conformation.
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _pivot_move(chain, i, k, rng):
    """
    Attempt a pivot move on the chain.
    """
    L = len(chain)
    if i < 0 or i + k > L:
        return None
    
    if i > 0:
        pivot = chain[i - 1]
        segment = chain[i:i+k]
        prefix = chain[:i]
        suffix = chain[i+k:]
    else:
        pivot = chain[i + k]
        segment = chain[i:i+k]
        prefix = []
        suffix = chain[i+k:]
        
    if not segment:
        return None

    rotations = [
        lambda p: (p[0], p[1]),
        lambda p: (-p[1], p[0]),
        lambda p: (-p[0], -p[1]),
        lambda p: (p[1], -p[0])
    ]
    
    rot = rng.choice(rotations)
    
    new_segment = []
    for p in segment:
        v = (p[0] - pivot[0], p[1] - pivot[1])
        rv = rot(v)
        new_p = (pivot[0] + rv[0], pivot[1] + rv[1])
        new_segment.append(new_p)
        
    new_chain = prefix + new_segment + suffix
    
    if len(new_chain) != len(set(new_chain)):
        return None
        
    return new_chain

def _local_search_greedy(sequence, initial_chain, rng, time_limit):
    """
    Perform Greedy Local Search starting from initial_chain.
    """
    current_chain = initial_chain
    current_energy = _calculate_energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    L = len(sequence)
    max_k = 5
    
    while time.time() < time_limit:
        improved = False
        # Try random pivot moves
        # To be faster, we can try a few random moves per iteration
        for _ in range(100): # 100 attempts per "iteration"
            i = rng.randint(0, L - 1)
            k = rng.randint(1, min(max_k, L - i))
            
            new_chain = _pivot_move(current_chain, i, k, rng)
            if new_chain is None:
                continue
                
            new_energy = _calculate_energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
                    
        if not improved:
            # Local minimum reached
            break
            
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    L = len(sequence)
    if L <= 1:
        return [[0] * dimension]
        
    rng = random.Random(12345 + L)
    
    best_chain = None
    best_energy = float('inf')
    
    while time.time() < deadline - 0.5:
        initial_chain = _grow(sequence, dimension, rng)
        if initial_chain is None:
            continue
            
        time_for_search = min(2.0, deadline - time.time() - 0.2)
        if time_for_search <= 0:
            break
            
        local_best = _local_search_greedy(sequence, initial_chain, rng, time.time() + time_for_search)
        
        if local_best is not None:
            local_energy = _calculate_energy(sequence, local_best)
            if local_energy < best_energy:
                best_energy = local_energy
                best_chain = local_best
                
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(L)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
