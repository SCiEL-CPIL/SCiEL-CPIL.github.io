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


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
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


def _pivot_move(chain, dimension, rng):
    """Pivot move: rotate a suffix of the chain around a pivot."""
    n = len(chain)
    if n < 3:
        return None
    pivot = rng.randint(1, n - 2)
    # Choose a random rotation axis and direction
    axis = rng.randint(0, dimension - 1)
    sign = rng.choice((-1, 1))
    
    # The pivot point is chain[pivot], the suffix starts at chain[pivot+1]
    pivot_point = chain[pivot]
    suffix = chain[pivot + 1:]
    
    def rotate90(point, axis, sign):
        x, y, z = point
        if axis == 0:
            if sign == 1:
                return (x, -z, y)
            else:
                return (x, z, -y)
        elif axis == 1:
            if sign == 1:
                return (z, y, -x)
            else:
                return (-z, y, x)
        else:  # axis 2
            if sign == 1:
                return (y, -x, z)
            else:
                return (-y, x, z)
    
    # Rotate each suffix point relative to pivot
    new_suffix = []
    occupied = set(chain[:pivot + 1])
    valid = True
    for p in suffix:
        rel = (p[0] - pivot_point[0], p[1] - pivot_point[1], p[2] - pivot_point[2])
        rel = list(rel)
        # Rotate
        if axis == 0:
            if sign == 1:
                new_rel = (rel[0], -rel[2], rel[1])
            else:
                new_rel = (rel[0], rel[2], -rel[1])
        elif axis == 1:
            if sign == 1:
                new_rel = (rel[2], rel[1], -rel[0])
            else:
                new_rel = (-rel[2], rel[1], rel[0])
        else:
            if sign == 1:
                new_rel = (rel[1], -rel[0], rel[2])
            else:
                new_rel = (-rel[1], rel[0], rel[2])
        new_point = (pivot_point[0] + new_rel[0], pivot_point[1] + new_rel[1], pivot_point[2] + new_rel[2])
        if new_point in occupied:
            valid = False
            break
        occupied.add(new_point)
        new_suffix.append(new_point)
    
    if not valid:
        return None
    return chain[:pivot + 1] + new_suffix


def _local_move(chain, dimension, rng):
    """Try to move a single interior residue to an adjacent empty site."""
    n = len(chain)
    if n < 3:
        return None
    # Pick a random interior index
    i = rng.randint(1, n - 2)
    site = chain[i]
    occupied = set(chain)
    occupied.remove(site)
    moves = [q for q in _neighbours(site) if q not in occupied]
    if not moves:
        return None
    new_site = rng.choice(moves)
    new_chain = list(chain)
    new_chain[i] = new_site
    return new_chain


def _mc_fold(sequence, dimension, rng, max_steps, start_temp):
    """Monte Carlo folding with cooling."""
    n = len(sequence)
    chain = _grow(sequence, dimension, rng)
    if chain is None or len(chain) != n:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.5:
            new_chain = _pivot_move(chain, dimension, rng)
        else:
            new_chain = _local_move(chain, dimension, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or (start_temp > 0 and rng.random() < min(1.0, max(1e-10, 2.718281828 ** (-delta / start_temp)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        if step % max(1, max_steps // 10) == 0 and start_temp > 0:
            start_temp = max(0.1, start_temp * 0.7)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via Monte Carlo search with time budget."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (straight line)
    
    # First, get a baseline with a few greedy growths
    for _ in range(10):
        chain = _grow(sequence, dimension, rng)
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Now do Monte Carlo refinement
    # Determine how many MC steps to do based on remaining time
    start_time = time.time()
    time_budget = max(1.0, min(12.0, (_DEADLINE - start_time - 5) / 1.0))
    
    # Estimate steps per second - do a quick calibration
    test_steps = 500
    t0 = time.time()
    for _ in range(test_steps):
        _pivot_move(best_chain, dimension, rng)
        _local_move(best_chain, dimension, rng)
    elapsed = time.time() - t0
    if elapsed > 0:
        steps_per_sec = test_steps / elapsed
    else:
        steps_per_sec = 1000
    
    max_steps = int(time_budget * steps_per_sec * 0.8)
    max_steps = max(100, min(max_steps, 50000))
    
    # Run MC
    chain, energy = _mc_fold(sequence, dimension, rng, max_steps, 2.0)
    if energy < best_energy:
        best_chain = chain
        best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
