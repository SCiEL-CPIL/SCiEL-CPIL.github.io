"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline: 330s wall-clock kill, use 315s as safety margin
_DEADLINE = time.time() + 315.0


def _neighbours(point):
    """Yield all 6 neighbours of a 3D lattice point."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: minus the number of H-H contacts (non-consecutive)."""
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
    """
    Grow the chain greedily, preferring sites that create new H-H contacts.
    Returns a self-avoiding walk (list of tuples) or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score this site
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Small bonus for free space to avoid trapping
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.5 + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(key=lambda x: -x[0])
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_steps, rng):
    """
    Simple local search: try moving each residue to a neighbouring empty site
    if it improves energy. Returns improved chain or original.
    """
    n = len(sequence)
    if n == 0:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    occupied = {p: i for i, p in enumerate(current)}
    
    improved = True
    steps = 0
    while improved and steps < max_steps:
        improved = False
        steps += 1
        for i in range(1, n - 1):  # don't move ends
            for new_pos in _neighbours(current[i]):
                if new_pos in occupied and occupied[new_pos] != i:
                    continue
                # Check if moving i to new_pos is valid
                # Must not collide with neighbours in chain
                valid = True
                for j in (i - 1, i + 1):
                    if current[j] == new_pos:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute energy change
                old_pos = current[i]
                # Remove old pos
                del occupied[old_pos]
                occupied[new_pos] = i
                
                # Check if new_pos collides with any other
                # (already checked above for i-1, i+1, but need full check)
                # Actually we need to ensure no other residue is at new_pos
                # occupied[new_pos] = i, but was there something else?
                # We checked new_pos not in occupied (or occupied[new_pos] == i)
                # But we deleted old_pos first, so if new_pos was old_pos of another... 
                # Let's just compute full energy for simplicity in local search
                test_chain = list(current)
                test_chain[i] = new_pos
                test_energy = _energy(sequence, test_chain)
                
                # Restore
                del occupied[new_pos]
                occupied[old_pos] = i
                
                if test_energy < current_energy:
                    current[i] = new_pos
                    current_energy = test_energy
                    improved = True
                    break
            if improved:
                break
    
    return current


def _simulated_annealing(sequence, chain, steps, rng):
    """
    Simulated annealing on the chain. Tries to move individual residues.
    """
    n = len(sequence)
    if n == 0:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    occupied = {p: i for i, p in enumerate(current)}
    
    best = list(current)
    best_energy = current_energy
    
    t_start = time.time()
    t_end = t_start + 5.0  # max 5 seconds per SA run
    
    T = 2.0
    cooling = 0.995
    
    for step in range(steps):
        if time.time() > t_end:
            break
        
        # Pick a random residue to move (not the ends)
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        # Pick a random neighbour
        neighbours = list(_neighbours(old_pos))
        new_pos = rng.choice(neighbours)
        
        # Check if new_pos is occupied by another residue
        if new_pos in occupied and occupied[new_pos] != i:
            continue
        
        # Check if new_pos is same as old_pos
        if new_pos == old_pos:
            continue
        
        # Check if new_pos collides with chain neighbours
        if new_pos == current[i - 1] or new_pos == current[i + 1]:
            continue
        
        # Compute energy change
        # Remove old_pos, add new_pos
        del occupied[old_pos]
        occupied[new_pos] = i
        
        test_chain = list(current)
        test_chain[i] = new_pos
        test_energy = _energy(sequence, test_chain)
        
        delta = test_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current[i] = new_pos
            current_energy = test_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = list(current)
        else:
            # Revert
            del occupied[new_pos]
            occupied[old_pos] = i
        
        T *= cooling
    
    return best


def fold(sequence, dimension):
    """
    Fold a sequence on the 3D cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # energy is negative, so 0 is worst
    
    # Phase 1: Multiple greedy grows
    num_grows = 50
    for _ in range(num_grows):
        if time.time() > _DEADLINE - 30:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Phase 2: Local search on the best chain
    if time.time() < _DEADLINE - 20:
        improved = _local_search(sequence, best_chain, max_steps=100, rng=rng)
        improved_energy = _energy(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    # Phase 3: Simulated annealing for further improvement
    if time.time() < _DEADLINE - 10:
        sa_chain = _simulated_annealing(sequence, best_chain, steps=2000, rng=rng)
        sa_energy = _energy(sequence, sa_chain)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Return as list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
