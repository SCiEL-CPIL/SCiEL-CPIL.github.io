"""HP lattice protein folding using Simulated Annealing with single-residue moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 175.0

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain, where):
    contacts = 0
    n = len(sequence)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _delta_energy(sequence, chain, where, i, old_pos, new_pos):
    delta = 0
    # Break contacts
    for q in _neighbours(old_pos):
        j = where.get(q)
        if j is not None and j != i and abs(i - j) > 1:
            if sequence[i] == 'H' and sequence[j] == 'H':
                delta += 1
    # Form contacts
    for q in _neighbours(new_pos):
        j = where.get(q)
        if j is not None and j != i and abs(i - j) > 1:
            if sequence[i] == 'H' and sequence[j] == 'H':
                delta -= 1
    return delta

def _grow_greedy(sequence, dimension, rng):
    start = (0, 0)
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _simulate_annealing(sequence, dimension, rng, initial_chain):
    n = len(sequence)
    if n == 0:
        return [], 0
        
    chain = initial_chain[:]
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain, where)
    
    best_energy = current_energy
    best_chain = chain[:]
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9999
    steps = 0
    max_steps = 200000
    
    while time.time() - _START_TIME < _TIME_LIMIT and steps < max_steps:
        # Single residue move
        i = rng.randint(0, n - 1)
        p = chain[i]
        neighbors = _neighbours(p)
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in where:
                continue
                
            delta = _delta_energy(sequence, chain, where, i, p, new_pos)
            
            # Apply move
            del where[p]
            where[new_pos] = i
            chain[i] = new_pos
            current_energy += delta
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Reject: Revert
                chain[i] = p
                del where[new_pos]
                where[p] = i
                current_energy -= delta
                
            moved = True
            break
            
        if not moved:
            # No valid neighbor found, skip
            pass
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
        steps += 1
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(12345)
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on time remaining
    # Assume 10 sequences, ~12s each.
    # We will run SA for a fixed duration per restart.
    
    num_restarts = 3
    time_per_restart = 8.0
    
    for r in range(num_restarts):
        if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
            break
            
        start_t = time.time()
        init_chain = _grow_greedy(sequence, dimension, rng)
        if init_chain is None:
            init_chain = [(i, 0) for i in range(n)]
            
        chain, energy = _simulate_annealing(sequence, dimension, rng, init_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        elapsed = time.time() - start_t
        # If we finished early, we could do more, but let's keep it simple.
        
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
