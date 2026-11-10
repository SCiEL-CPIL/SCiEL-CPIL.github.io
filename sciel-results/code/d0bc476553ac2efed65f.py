"""HP lattice protein folding on the square lattice using simulated annealing."""
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain, where):
    """Compute energy given pre-built position map."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _try_move(chain, where, pos, direction, sequence):
    """Try to move residue at position pos in given direction.
    Returns (new_chain, new_where, delta_energy) or None if move invalid."""
    old_pos = chain[pos]
    new_pos = tuple(old_pos[i] + direction[i] for i in range(len(old_pos)))
    
    # Check self-avoidance
    if new_pos in where:
        if where[new_pos] == pos:
            return None
        return None
    
    # Check adjacency: new_pos must be adjacent to chain[pos-1] and chain[pos+1]
    if pos > 0:
        prev = chain[pos - 1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return None
    if pos < len(chain) - 1:
        nxt = chain[pos + 1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return None
    
    # Compute delta energy
    # Remove old contacts involving pos
    old_contacts = 0
    if sequence[pos] == 'H':
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and j != pos:
                if sequence[j] == 'H':
                    if j > pos + 1 or (j < pos - 1):
                        old_contacts += 1
    
    # Add new contacts involving pos
    new_contacts = 0
    if sequence[pos] == 'H':
        for q in _neighbours(new_pos):
            j = where.get(q)
            if j is not None and j != pos:
                if sequence[j] == 'H':
                    if j > pos + 1 or (j < pos - 1):
                        new_contacts += 1
    
    delta_energy = -(new_contacts - old_contacts)
    
    # Build new chain and where
    new_chain = list(chain)
    new_chain[pos] = new_pos
    new_where = dict(where)
    del new_where[old_pos]
    new_where[new_pos] = pos
    
    return (new_chain, new_where, delta_energy)


def _initial_chain(sequence, dimension):
    """Create a simple spiral or zigzag initial conformation."""
    n = len(sequence)
    # Start with a simple pattern
    chain = [(0, 0)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d_idx = 0
    for i in range(1, n):
        # Try each direction, prefer turning to make compact shape
        placed = False
        for offset in range(4):
            d = directions[(d_idx + offset) % 4]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in chain:
                chain.append(new_pos)
                if offset > 0:
                    d_idx = (d_idx + offset) % 4
                placed = True
                break
        if not placed:
            # Fallback: continue in current direction
            d = directions[d_idx]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            chain.append(new_pos)
    return chain


def _anneal(sequence, dimension, rng, max_iters=50000):
    """Simulated annealing to minimize energy."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    # Initialize
    chain = _initial_chain(sequence, dimension)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain, where)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Annealing schedule
    T_start = 2.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        T *= cooling_rate
        
        # Pick a random internal residue to move
        pos = rng.randint(1, n - 2)
        
        # Try a random direction
        direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        
        result = _try_move(chain, where, pos, direction, sequence)
        if result is None:
            continue
        
        new_chain, new_where, delta_energy = result
        
        if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / T):
            chain = new_chain
            where = new_where
            current_energy += delta_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain


def fold(sequence, dimension):
    """Find best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Multiple restarts with different seeds
    num_restarts = 10
    iters_per_restart = 30000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        chain = _anneal(sequence, dimension, rng, max_iters=iters_per_restart)
        where = {p: i for i, p in enumerate(chain)}
        energy = _compute_energy(sequence, chain, where)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # If no valid chain found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
