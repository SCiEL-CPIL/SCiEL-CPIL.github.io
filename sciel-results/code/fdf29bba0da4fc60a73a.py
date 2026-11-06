"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 190.0  # wall-clock safety margin (evaluator kills at 210s)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _local_search(sequence, chain, rng, time_limit):
    """Local search using single-residue moves and 2-residue reversals."""
    n = len(chain)
    if n < 2:
        return chain, _energy(sequence, chain)
    
    current = [list(p) for p in chain]
    current_energy = _energy(sequence, current)
    best = [list(p) for p in current]
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Pick a random move type
        if rng.random() < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)
            orig = current[i]
            for site in _neighbours(orig):
                if any(site == current[j] for j in range(n) if j != i):
                    continue
                # Check unit steps to neighbors
                if abs(site[0] - current[i-1][0]) + abs(site[1] - current[i-1][1]) != 1:
                    continue
                if abs(site[0] - current[i+1][0]) + abs(site[1] - current[i+1][1]) != 1:
                    continue
                current[i] = site
                new_energy = _energy(sequence, current)
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = [list(p) for p in current]
                    break
                else:
                    current[i] = orig
        else:
            # 2-residue reversal
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            # Reverse segment from i to i+1
            current[i], current[i+1] = current[i+1], current[i]
            # Check self-avoidance
            valid = True
            seen = set()
            for j in range(n):
                p = tuple(current[j])
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                current[i], current[i+1] = current[i+1], current[i]
                continue
            # Check unit steps
            valid = True
            for j in range(n-1):
                if abs(current[j][0] - current[j+1][0]) + abs(current[j][1] - current[j+1][1]) != 1:
                    valid = False
                    break
            if not valid:
                current[i], current[i+1] = current[i+1], current[i]
                continue
            new_energy = _energy(sequence, current)
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best = [list(p) for p in current]
            else:
                current[i], current[i+1] = current[i+1], current[i]
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation found within the time budget."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    start_time = time.time()
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growth with multiple restarts
    budget_per_phase = (_DEADLINE - (time.time() - start_time)) / 3.0
    phase1_end = start_time + budget_per_phase
    
    while time.time() < phase1_end:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = [list(p) for p in chain]
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 5:
        improved, improved_energy = _local_search(sequence, best_chain, rng, remaining - 2)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    return best_chain
# EVOLVE-BLOCK-END
