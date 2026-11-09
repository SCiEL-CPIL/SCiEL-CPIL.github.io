"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _rotate_point(point, center, rot):
    x, y = point[0] - center[0], point[1] - center[1]
    if rot == 0:
        nx, ny = x, y
    elif rot == 90:
        nx, ny = -y, x
    elif rot == 180:
        nx, ny = -x, -y
    else: # 270
        nx, ny = y, -x
    return (center[0] + nx, center[1] + ny)

def fold(sequence, dimension):
    """Best conformation using Simulated Annealing with Pivot Moves."""
    rng = random.Random(12345 + len(sequence))
    
    # 1. Generate initial valid chain
    best_chain = None
    best_energy = 0
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
        if best_energy <= -len(sequence) * 0.5: # Heuristic: good enough
            break

    if best_chain is None:
        return [list(p) for p in [(i, 0) for i in range(len(sequence))]]

    chain = best_chain
    current_energy = best_energy
    N = len(chain)
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    # Time check
    while time.time() < _DEADLINE:
        # Pick a pivot
        k = rng.randint(1, N - 2)
        rot = rng.choice([90, 180, 270])
        
        # Rotate segment k..N-1 around chain[k-1]
        pivot = chain[k-1]
        new_segment = []
        for i in range(k, N):
            new_segment.append(_rotate_point(chain[i], pivot, rot))
        
        # Check collision with fixed part (0..k-2)
        # Note: chain[k-1] is the pivot, it stays. 
        # The segment starts at k.
        # We need to check if any point in new_segment is in chain[0..k-2]
        
        fixed_set = set(chain[:k-1])
        
        valid = True
        for p in new_segment:
            if p in fixed_set:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate new energy
        new_chain = chain[:k] + new_segment
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance
        if delta_e < 0 or (T > 0 and rng.random() < pow(2.71828, -delta_e / T)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain
        
        T *= cooling
        if T < T_min:
            T = 5.0 # Restart temperature
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
