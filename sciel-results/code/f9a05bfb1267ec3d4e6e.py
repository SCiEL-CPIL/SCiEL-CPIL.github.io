"""HP lattice protein folding using greedy growth with local refinement."""
# EVOLVE-BLOCK-START
import random
import time


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


def _grow(sequence, dimension, rng, free_weight=0.1):
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
            options.append((gain + free_weight * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_refine(sequence, chain, rng, max_steps=200):
    """Local search: try single-residue moves and small reversals to improve energy."""
    if chain is None:
        return None
    
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {tuple(p): i for i, p in enumerate(best_chain)}
    
    def try_move(idx, new_pos):
        """Try moving residue idx to new_pos, return new energy if valid."""
        old_pos = tuple(best_chain[idx])
        if new_pos == old_pos:
            return None
        # Check if new_pos is occupied by another residue
        if new_pos in occupied:
            other_idx = occupied[new_pos]
            if other_idx == idx:
                return None
            return None
        # Check adjacency with neighbors
        if idx > 0:
            if abs(best_chain[idx-1][0] - new_pos[0]) + abs(best_chain[idx-1][1] - new_pos[1]) != 1:
                return None
        if idx < n - 1:
            if abs(best_chain[idx+1][0] - new_pos[0]) + abs(best_chain[idx+1][1] - new_pos[1]) != 1:
                return None
        # Try the move
        new_chain = best_chain[:]
        new_chain[idx] = new_pos
        new_occupied = {tuple(p): i for i, p in enumerate(new_chain)}
        # Check self-avoidance
        if len(new_occupied) != n:
            return None
        new_energy = _energy(sequence, new_chain)
        return new_energy
    
    for step in range(max_steps):
        improved = False
        # Try single-residue moves
        for idx in range(1, n - 1):  # Don't move endpoints
            for new_pos in _neighbours(best_chain[idx]):
                result = try_move(idx, new_pos)
                if result is not None and result < best_energy:
                    old_pos = tuple(best_chain[idx])
                    best_chain[idx] = new_pos
                    best_energy = result
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try a small reversal
            if n >= 4:
                start = rng.randint(1, n - 3)
                end = rng.randint(start + 1, n - 1)
                # Check if reversal is valid
                new_chain = best_chain[:]
                new_chain[start:end+1] = reversed(new_chain[start:end+1])
                # Check self-avoidance
                if len(set(tuple(p) for p in new_chain)) == n:
                    # Check adjacency
                    valid = True
                    for i in range(n - 1):
                        if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                            valid = False
                            break
                    if valid:
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < best_energy:
                            best_chain = new_chain
                            best_energy = new_energy
                            improved = True
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth with local refinement."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 180s limit
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Allocate budget: more sequences get fewer restarts
    if n <= 20:
        max_growths = 300
        refine_steps = 300
    elif n <= 40:
        max_growths = 200
        refine_steps = 200
    elif n <= 60:
        max_growths = 150
        refine_steps = 150
    else:
        max_growths = 100
        refine_steps = 100
    
    for attempt in range(max_growths):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local refinement
        refined = _local_refine(sequence, chain, rng, max_steps=refine_steps)
        if refined is None:
            continue
        
        value = _energy(sequence, refined)
        if value < best_energy:
            best, best_energy = refined, value
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
