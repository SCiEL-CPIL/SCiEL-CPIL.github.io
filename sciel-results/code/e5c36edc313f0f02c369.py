"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
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
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, max_iter=5000):
    """Improve conformation via single-residue moves that preserve self-avoidance."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        # Try moving each residue (skip endpoints for simplicity, or try all)
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                # Check if new position is valid
                if new_pos in occupied:
                    continue
                # Check if moving would create chain adjacency issues
                # The residue at i must not be adjacent to residues at i-1 or i+1 in the new position
                # Actually, it's already adjacent to i-1 and i+1, so we just need self-avoidance
                # and that it doesn't become adjacent to itself (impossible) or create new adjacencies
                # Wait: after moving, residue i is at new_pos. It should not be adjacent to any other residue
                # except i-1 and i+1. But since we're only moving one residue, we need to check:
                # 1. new_pos is not occupied (already checked)
                # 2. new_pos is not adjacent to any residue j where |j-i| > 1 (would create invalid adjacency)
                
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute energy change
                old_energy = best_energy
                # Remove old contacts involving residue i
                new_energy = old_energy
                # This is complex; let's just compute full energy for simplicity (slower but correct)
                
                # Temporarily move
                best_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    # Keep the move
                else:
                    # Revert
                    best_chain[i] = current_pos
                    occupied.pop(new_pos)
                    occupied[current_pos] = i
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growths with local optimization."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Adjust number of restarts based on sequence length
    if n <= 30:
        num_restarts = 500
    elif n <= 60:
        num_restarts = 300
    else:
        num_restarts = 150
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        chain, energy = _local_search(sequence, chain, max_iter=min(5000, max(100, 10000 // n)))
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    if best is None:  # a straight line is always self-avoiding
        best = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
