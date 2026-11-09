"""HP lattice protein folding on the cubic lattice.
Uses randomized greedy growth with tail re-growth refinement and deadline guard.
"""
import random
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_tail(sequence, chain, start_idx, dimension, rng, max_iters=50):
    """Regrow the tail of the chain from start_idx to improve energy."""
    if start_idx >= len(chain) - 1:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Try regrowing from start_idx with some randomness
        prefix = chain[:start_idx + 1]
        occupied = {p: i for i, p in enumerate(prefix)}
        
        current_tail = [tuple(p) for p in prefix]
        success = True
        
        for i in range(start_idx + 1, len(sequence)):
            options = []
            for site in _neighbours(current_tail[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # More randomness in tail regrowth
                options.append((gain + 0.05 * free + rng.random() * 2.0, site))
            if not options:
                success = False
                break
            site = max(options)[1]
            occupied[site] = i
            current_tail.append(site)
        
        if not success:
            continue
        
        new_chain = current_tail
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized growth + tail re-growth with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    # Phase 1: Multiple randomized greedy growths
    num_initial = 100
    for _ in range(num_initial):
        if time.time() > _DEADLINE - 30:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Tail re-growth refinement on the best chain
    if time.time() < _DEADLINE - 10:
        # Try regrowing from different positions
        n = len(sequence)
        # Try several tail lengths
        tail_starts = [max(0, n - 20), max(0, n - 40), max(0, n - 60)]
        for ts in tail_starts:
            if time.time() > _DEADLINE - 5:
                break
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, ts, dimension, rng, max_iters=30
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Phase 3: Full local search - try regrowing entire tail from various points
    if time.time() < _DEADLINE - 5:
        # Try regrowing from mid-chain positions
        n = len(sequence)
        mid_positions = [n // 3, 2 * n // 3]
        for mp in mid_positions:
            if time.time() > _DEADLINE - 2:
                break
            improved_chain, improved_energy = _regrow_tail(
                sequence, best_chain, mp, dimension, rng, max_iters=20
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
