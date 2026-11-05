"""HP lattice protein folding using greedy growth and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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
            # Add some randomness to escape local traps during growth
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, rng, steps=300, t_start=1.0, t_end=0.05):
    """Simulated annealing local search starting from an initial chain."""
    current = [list(p) for p in chain]
    current_energy = _energy(sequence, current)
    best = [list(p) for p in current]
    best_energy = current_energy
    
    n = len(sequence)
    if n <= 2:
        return best, best_energy
        
    occupied = {tuple(p): i for i, p in enumerate(current)}
    
    def delta_energy(index, new_pos):
        old_pos = tuple(current[index])
        dE = 0
        # Losing contacts from old position
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != index and abs(j - index) > 1:
                if sequence[index] == 'H' and sequence[j] == 'H':
                    dE += 1
        # Gaining contacts from new position
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != index and abs(j - index) > 1:
                if sequence[index] == 'H' and sequence[j] == 'H':
                    dE -= 1
        return dE

    for step in range(steps):
        t = t_start * (t_end / t_start) ** (step / steps)
        
        i = rng.randint(1, n - 2)
        old_pos = tuple(current[i])
        
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        moved = False
        for new_pos in neighbors:
            if new_pos in occupied:
                continue
            
            dE = delta_energy(i, new_pos)
            
            if dE < 0 or rng.random() < math.exp(-dE / t):
                del occupied[old_pos]
                occupied[new_pos] = i
                current[i] = list(new_pos)
                current_energy += dE
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best = [list(p) for p in current]
                
                moved = True
                break
        
        if not moved:
            continue
            
    return best, best_energy

def fold(sequence, dimension):
    """Best conformation found via growth and local search."""
    start_time = time.time()
    deadline = start_time + 18.0
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    n = len(sequence)
    
    while time.time() < deadline:
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        steps = 200 + n
        refined, energy = _local_search(sequence, chain, rng, steps=steps, t_start=1.0, t_end=0.05)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = refined

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    else:
        # Final validity check
        occupied = set()
        valid = True
        for i, p in enumerate(best_chain):
            if tuple(p) in occupied:
                valid = False
                break
            occupied.add(tuple(p))
            if i > 0:
                prev = best_chain[i-1]
                dist = sum(abs(a-b) for a, b in zip(p, prev))
                if dist != 1:
                    valid = False
                    break
        if not valid:
             best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
