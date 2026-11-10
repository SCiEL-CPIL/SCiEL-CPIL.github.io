"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_START_TIME = time.time()
_TOTAL_BUDGET = 170.0

def _time_remaining():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth with contact-aware scoring."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 2.0 + 0.1 * free + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: -x[0])
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        chosen = rng.choice(options[:top_k])
        site = chosen[1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _segment_swap(sequence, chain, rng):
    """
    Try to reverse a segment of the chain to find better contacts.
    """
    n = len(sequence)
    if n < 5:
        return chain[:]
    
    best_chain = chain[:]
    best_energy = _compute_energy(sequence, best_chain)
    
    for _ in range(50):
        if _time_remaining() < 3.0:
            break
        
        # Pick two random positions i < j
        i = rng.randint(1, n - 4)
        j = rng.randint(i + 2, n - 1)
        
        # Reverse segment [i, j]
        new_chain = best_chain[:]
        new_chain[i:j+1] = list(reversed(new_chain[i:j+1]))
        
        # Check self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        
        # Check unit steps
        valid = True
        for k in range(len(new_chain) - 1):
            dx = abs(new_chain[k][0] - new_chain[k+1][0])
            dy = abs(new_chain[k][1] - new_chain[k+1][1])
            if dx + dy != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple growth attempts
    num_growths = min(500, max(50, int(_time_remaining() * 5)))
    
    for _ in range(num_growths):
        if _time_remaining() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
    
    # Phase 2: Local search on best chains
    if best_chain is not None and _time_remaining() > 10.0:
        # Do segment swaps on the best chain
        improved = _segment_swap(sequence, best_chain, rng)
        improved_energy = _compute_energy(sequence, improved)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    # Phase 3: More growth attempts with different seeds
    if _time_remaining() > 10.0:
        for _ in range(200):
            if _time_remaining() < 5.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
        
        # Another round of local search
        if best_chain is not None and _time_remaining() > 5.0:
            improved = _segment_swap(sequence, best_chain, rng)
            improved_energy = _compute_energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
