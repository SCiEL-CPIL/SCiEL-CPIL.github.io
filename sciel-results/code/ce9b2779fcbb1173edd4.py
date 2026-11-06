"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 310  # safety margin before 330s kill


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


def _pivot_move(chain, i, rng, dimension):
    """Try a pivot move at position i: rotate the tail segment by 90 degrees around an axis."""
    if i <= 1 or i >= len(chain) - 1:
        return None
    
    n = len(chain)
    head = chain[:i+1]
    tail = chain[i+1:]
    
    # Choose a random axis for rotation
    axis = rng.randint(0, dimension - 1)
    # Choose rotation direction
    sign = rng.choice([-1, 1])
    
    # Rotate tail around the pivot point (chain[i])
    pivot = chain[i]
    new_tail = []
    for pos in tail:
        new_pos = list(pos)
        # Vector from pivot to current position
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        # Rotate 90 degrees around chosen axis
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = [pivot[k] + new_vec[k] for k in range(dimension)]
        new_tail.append(tuple(new_pos))
    
    new_chain = head + new_tail
    
    # Check self-avoidance
    occupied = set(new_chain)
    if len(occupied) != n:
        return None
    
    # Check unit steps
    for k in range(n - 1):
        diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
        if diff != 1:
            return None
    
    return new_chain


def _anneal(sequence, chain, dimension, rng, max_iters=500):
    """Simulated annealing refinement of a chain."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        i = rng.randint(2, len(current) - 2)
        new_chain = _pivot_move(current, i, rng, dimension)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        T *= alpha
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    # Determine number of restarts based on chain length
    if n <= 48:
        n_restarts = 150
    elif n <= 80:
        n_restarts = 100
    else:
        n_restarts = 60
    
    for _ in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Refine with simulated annealing
        refined, refined_energy = _anneal(sequence, chain, dimension, rng, max_iters=300)
        
        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
