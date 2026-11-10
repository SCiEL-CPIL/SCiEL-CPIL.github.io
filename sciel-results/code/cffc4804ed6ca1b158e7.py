# EVOLVE-BLOCK-START
import random
import time


_DEADLINE_START = time.time()
_DEADLINE_SECONDS = 310.0  # safety margin before 330s hard kill


def _remaining_time():
    return _DEADLINE_START + _DEADLINE_SECONDS - time.time()


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


def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Improve a valid chain via single-residue moves with delta energy calculation."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    # Pre-compute H positions for faster contact checking?
    # Or just check neighbors directly.
    
    for _ in range(max_iters):
        if _remaining_time() < 5.0:
            break

        # Pick a random internal residue (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = best_chain[i]
        
        # Generate candidate positions: all neighbours of old_pos
        candidates = list(_neighbours(old_pos))
        
        improved = False
        for new_pos in candidates:
            if new_pos == old_pos:
                continue
            if new_pos in occupied:
                continue
            
            # Check connectivity to neighbors i-1 and i+1
            prev_pos = best_chain[i-1]
            next_pos = best_chain[i+1]
            
            # Manhattan distance must be 1 for unit steps
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) + abs(new_pos[2] - prev_pos[2]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) + abs(new_pos[2] - next_pos[2]) != 1:
                continue

            # Calculate Delta Energy
            # Remove contacts involving i at old_pos
            delta = 0
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This contact is lost
                        delta += 1 # Energy increases (becomes less negative) by 1, so delta E = +1 in terms of "cost"
                        # Wait, Energy = -contacts. Losing a contact increases energy by 1.
            
            # Add contacts involving i at new_pos
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This contact is gained
                        delta -= 1 # Energy decreases by 1.

            new_energy = best_energy + delta
            
            if new_energy < best_energy:
                # Accept move
                occupied.pop(old_pos)
                occupied[new_pos] = i
                best_chain[i] = new_pos
                best_energy = new_energy
                improved = True
                break # Restart iteration with new state? Or continue checking other candidates?
                # Usually better to break and restart the random selection to avoid bias
            # else: reject
            
        if not improved:
            # Try a few random perturbations? No, keep it simple.
            pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growth + local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)

    # 1. Multiple Greedy Restarts
    # Limit restarts to ensure we have time for local search
    max_restarts = 50
    restarts_done = 0
    
    while _remaining_time() > 30.0 and restarts_done < max_restarts:
        chain = _grow(sequence, dimension, rng)
        restarts_done += 1
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

    # 2. Local Search on the best chain found
    if best is not None and _remaining_time() > 10.0:
        best, best_energy = _local_search(sequence, best, dimension, rng)

    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
