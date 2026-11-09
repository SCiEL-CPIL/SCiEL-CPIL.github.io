"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0  # seconds, with safety margin before 330s kill


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


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with temperature-controlled exploration."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
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
            # Score: strong preference for H-H contacts, mild preference for free space
            score = gain * 5.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_moves=200, rng=None):
    """Simple local search: try moving individual residues to improve energy."""
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_moves):
        i = rng.randint(1, len(best_chain) - 2)  # Don't move endpoints
        old_pos = best_chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            if new_pos == best_chain[i-1] or new_pos == best_chain[i+1]:
                continue  # Must maintain unit steps
            
            # Check if move is valid (doesn't create collisions)
            valid = True
            for j, p in enumerate(best_chain):
                if j != i and p == new_pos:
                    valid = False
                    break
            if not valid:
                continue
            
            # Try the move
            best_chain[i] = new_pos
            occupied.pop(old_pos)
            occupied[new_pos] = i
            new_energy = _energy(sequence, best_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                moved = True
            else:
                # Revert
                best_chain[i] = old_pos
                occupied.pop(new_pos)
                occupied[old_pos] = i
        
        if not moved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence using multiple strategies within time budget."""
    start_time = time.time()
    n = len(sequence)
    
    # Strategy 1: Multiple greedy grows with different temperatures
    best_chain = None
    best_energy = 1  # Start with unfolded (energy 0 means 0 contacts, but we use -contacts)
    
    # Try different temperatures and seeds
    temperatures = [0.1, 0.5, 1.0, 2.0]
    seeds = [12345, 54321, 99999, 11111, 22222]
    
    for temp in temperatures:
        for seed in seeds:
            if time.time() - start_time > _DEADLINE - 5:
                break
            rng = random.Random(seed + len(sequence))
            chain = _grow(sequence, dimension, rng, temperature=temp)
            if chain is not None:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
        
        if time.time() - start_time > _DEADLINE - 5:
            break
    
    # Strategy 2: Local search on best chain found
    if best_chain is not None and time.time() - start_time < _DEADLINE - 10:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_moves=500, rng=random.Random(777))
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Strategy 3: Additional greedy grows if time permits
    if time.time() - start_time < _DEADLINE - 20:
        additional_seeds = [33333, 44444, 55555, 66666, 77777, 88888]
        for seed in additional_seeds:
            if time.time() - start_time > _DEADLINE - 10:
                break
            rng = random.Random(seed + n * 1000)
            chain = _grow(sequence, dimension, rng, temperature=0.3)
            if chain is not None:
                energy = _energy(sequence, chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
                # Quick local search
                if time.time() - start_time < _DEADLINE - 15:
                    lc, le = _local_search(sequence, chain, max_moves=100, rng=random.Random(seed))
                    if le < best_energy:
                        best_chain = lc
                        best_energy = le
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
