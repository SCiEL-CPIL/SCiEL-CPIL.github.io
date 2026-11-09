"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
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


def _local_search(sequence, chain, max_iters=5000):
    """Local search using single-residue moves (shift one residue to an adjacent free site)."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    
    # Build occupied set
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        orig = best_chain[i]
        
        # Try moving to each neighbor
        moved = False
        for site in _neighbours(orig):
            if site in occupied and occupied[site] != i:
                continue
            # Check that the new position doesn't conflict with neighbors in the chain
            # (positions i-1 and i+1)
            if site == best_chain[i-1] or site == best_chain[i+1]:
                continue
            
            # Apply move
            best_chain[i] = site
            occupied[site] = i
            del occupied[orig]
            
            new_energy = _energy(sequence, best_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                moved = True
                break
            else:
                # Revert
                best_chain[i] = orig
                occupied[orig] = i
                del occupied[site]
        
        if not moved and rng.random() < 0.1:
            # Random restart within local search to escape local minima
            # Try a random valid move even if it doesn't improve
            i = rng.randint(1, n - 2)
            orig = best_chain[i]
            neighbors = [s for s in _neighbours(orig) if not (s in occupied and occupied[s] != i)
                         and s != best_chain[i-1] and s != best_chain[i+1]]
            if neighbors:
                site = rng.choice(neighbors)
                best_chain[i] = site
                occupied[site] = i
                del occupied[orig]
                best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # For very short sequences, exhaustive or near-exhaustive search
    if n <= 8:
        # Use more restarts
        n_restarts = 500
    elif n <= 20:
        n_restarts = 300
    elif n <= 50:
        n_restarts = 200
    else:
        n_restarts = 100
    
    best_chain = None
    best_energy = 0  # Start at 0 (unfolded)
    
    # Phase 1: Greedy growth with multiple restarts
    for _ in range(n_restarts):
        if _remaining_time() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best solution
    if _remaining_time() > 10.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=10000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: More restarts if time permits, with local search on each
    while _remaining_time() > 30.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
            # Refine the new best
            if _remaining_time() > 15.0:
                improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=5000)
                if improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy
        elif _remaining_time() > 60.0:
            # Even if not better than best, try local search on this chain
            # in case it leads to a better basin
            improved_chain, improved_energy = _local_search(sequence, chain, max_iters=2000)
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Final polish: one more local search pass on the best
    if _remaining_time() > 5.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, max_iters=20000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
